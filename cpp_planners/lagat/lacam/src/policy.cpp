#include "../include/policy.hpp"

#include <torch/nn/functional.h>
#include <torch/torch.h>

using namespace Common;

std::string AgentPolicy::MODEL_FILEPATH = "";
int AgentPolicy::OBSERVATION_RAD = 5;
float AgentPolicy::SAMPLING_TEMPERTURE = 1.0;
AgentPolicy::SamplingStrategy AgentPolicy::SAMPLING_STRATEGY =
    SamplingStrategy::Deterministic;

AgentPolicy::AgentPolicy(const Instance* _ins, DistTable* _D, int seed)
    : ins(_ins),
      MT(seed),
      rrd(0, 1),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      occupied_now(V_size, NO_AGENT),
      use_model(std::filesystem::exists(MODEL_FILEPATH)),
      fov_size(OBSERVATION_RAD * 2 + 1),
      inputs(1),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      preferences(N, std::vector<std::pair<Vertex*, ActionCost>>(5))
{
  if (use_model) {
    auto timer = Deadline();
    torch::jit::getProfilingMode() = false;
    torch::manual_seed(seed);
    info(1, "use learned model for preference construction, ",
         "cuda availability:", torch::cuda::is_available());
    model = torch::jit::load(MODEL_FILEPATH);
    model.to(device);
    model.eval();
    info(1, "model loaded");
    set_preferences_learned(ins->starts);
    info(1, "finish first inference");
    MODEL_LOAD_MS = timer.elapsed_ms();
  }
}

AgentPolicy::~AgentPolicy() {}

void AgentPolicy::set_preferences(const Config& Q_from,
                                  const std::set<int>& default_policy_agents)
{
  if (use_model) {
    set_preferences_learned(Q_from);
    if (!default_policy_agents.empty())
      set_preferences_naive(Q_from, default_policy_agents);
  } else {
    set_preferences_naive(Q_from);
  }
}

void AgentPolicy::set_preferences_naive(const Config& Q_from,
                                        const std::set<int>& A)
{
  auto get_cost = [&](const int i, const Vertex* u) {
    return std::make_tuple(D->get(i, u), rrd(MT));
  };

  auto set = [&](const int i) {
    const auto K = Q_from[i]->neighbor.size();

    // set candidate actions
    for (size_t k = 0; k <= K; ++k) {
      auto u = Q_from[i]->actions[k];
      preferences[i][k] = std::make_pair(u, get_cost(i, u));
    }

    // sort, note: K + 1 is sufficient
    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  };

  if (A.empty()) {
    for (int i = 0; i < N; ++i) set(i);
  } else {
    for (auto i : A) set(i);
  }
}

int get_policy_action_index_from_vertex(Vertex* v_from, Vertex* v_to)
{
  if (v_from->x == v_to->x && v_from->y > v_to->y) return 1;  // north
  if (v_from->x == v_to->x && v_from->y < v_to->y) return 2;  // south
  if (v_from->x > v_to->x && v_from->y == v_to->y) return 3;  // west
  if (v_from->x < v_to->x && v_from->y == v_to->y) return 4;  // east
  return 0;                                                   // stay
}

static int inference_cnt = 0;

void AgentPolicy::set_preferences_learned(const Config& Q)
{
  c10::InferenceMode guard(true);

  auto itr = known_config_table.find(Q);
  if (itr == known_config_table.end()) {
    // model inference
    set_features(Q);
    auto&& actions = model.forward(inputs).toTensor();
    ++inference_cnt;
    actions = actions.to(torch::kCPU);
    itr = std::get<0>(known_config_table.emplace(Q, actions));
  }
  auto&& actions = itr->second;

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    auto acc = actions.accessor<float, 2>();
    auto v_val = -acc[i][v_idx];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rrd(MT);
    }

    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      /* CS-PIBT paper  */
      return std::make_pair(v_to,
                            std::make_tuple((float)D->get(i, v_to), v_val));
    } else {
      /* deterministic or probabilistic */
      return std::make_pair(v_to, std::make_tuple(v_val, rrd(MT)));
    }
  };

  // set preference
  for (int i = 0; i < N; ++i) {
    // set candidate actions
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }

    // sort, note: K + 1 is sufficient
    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

void AgentPolicy::set_features(const Config& Q)
{
  c10::InferenceMode guard(true);

  const float d_invalid = 2 * OBSERVATION_RAD;

  // for edge construction
  for (int i = 0; i < N; ++i) occupied_now[Q[i]->id] = i;

  static torch::Tensor node_feature =
      torch::zeros({N, 2, fov_size, fov_size}, torch::kFloat32);
  static auto X = node_feature.accessor<float, 4>();

  int edge_ptr = 0;
  std::vector<std::pair<int, int>> vec_edge_index;
  std::vector<std::tuple<int, int, int>> vec_edge_attr;
  auto add_edge = [&](int src, int dst, float x_rel = 0, float y_rel = 0) {
    vec_edge_index.emplace_back(src, dst);
    vec_edge_attr.emplace_back(y_rel, x_rel, std::abs(x_rel) + std::abs(y_rel));
    ++edge_ptr;
  };

  auto&& W = ins->G->width;
  auto&& H = ins->G->height;

  // feature and edge_index construction
  for (int i = 0; i < N; ++i) {
    const auto v_i = Q[i];
    const auto d_base = D->get(i, v_i);

    // local, global
    for (auto y_l = 0; y_l < fov_size; ++y_l) {
      const auto y_g = y_l + v_i->y - OBSERVATION_RAD;
      for (auto x_l = 0; x_l < fov_size; ++x_l) {
        const auto x_g = x_l + v_i->x - OBSERVATION_RAD;

        // set default value
        X[i][0][y_l][x_l] = 1.0f;  // cost-to-go
        X[i][1][y_l][x_l] = 0;     // no agent

        if (0 <= x_g && x_g < W && 0 <= y_g && y_g < H) {
          const auto u = ins->G->U[W * y_g + x_g];
          if (u != nullptr) {
            // check neighbor agent
            const auto j = occupied_now[u->id];
            if (j != NO_AGENT) {
              const auto pos_diff_x = v_i->x - u->x;
              const auto pos_diff_y = v_i->y - u->y;
              add_edge(i, j, pos_diff_x, pos_diff_y);
              // other agent
              X[i][1][y_l][x_l] = 1;
            }

            // cost-to-go
            X[i][0][y_l][x_l] = std::max(
                std::min((D->get(i, u) - d_base) / d_invalid, 1.0f), -1.0f);
          }
        }
      }
    }
  }

  torch::Tensor edge_index = torch::empty({2, edge_ptr}, torch::kInt64);
  torch::Tensor edge_attr = torch::empty({edge_ptr, 3}, torch::kFloat32);
  auto edge_index_acc = edge_index.accessor<int64_t, 2>();
  auto edge_attr_acc = edge_attr.accessor<float, 2>();

  for (int64_t i = 0; i < edge_ptr; ++i) {
    edge_index_acc[0][i] = vec_edge_index[i].first;
    edge_index_acc[1][i] = vec_edge_index[i].second;
    edge_attr_acc[i][0] = std::get<0>(vec_edge_attr[i]);
    edge_attr_acc[i][1] = std::get<1>(vec_edge_attr[i]);
    edge_attr_acc[i][2] = std::get<2>(vec_edge_attr[i]);
  }

  auto kwargs = torch::Dict<std::string, torch::Tensor>();
  kwargs.insert("x", node_feature.to(device));
  kwargs.insert("edge_index", edge_index.to(device));
  kwargs.insert("edge_attr", edge_attr.to(device));
  inputs[0] = kwargs;

  // cleanup
  for (int i = 0; i < N; ++i) occupied_now[Q[i]->id] = NO_AGENT;
}

Vertex* AgentPolicy::get(const int i, const int k)
{
  return std::get<0>(preferences[i][k]);
}
