#pragma once
#include <torch/script.h>

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

using ActionCost = std::tuple<float, float>;
using Preference = std::vector<std::pair<Vertex *, ActionCost>>;
using Preferences = std::vector<Preference>;

constexpr int NO_AGENT = -1;

struct AgentPolicy {
  enum SamplingStrategy {
    Deterministic,
    Probablistic,
    Tiebreaking,
  };

  const Instance *ins;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;
  std::vector<int> occupied_now;  // for quick location check
  bool use_model;

  // inference
  const int fov_size;
  std::vector<torch::jit::IValue> inputs;
  torch::Device device;
  std::unordered_map<Config, torch::Tensor, ConfigHasher> known_config_table;

  // main
  Preferences preferences;
  torch::jit::script::Module model;

  // hyper parameters
  static std::string MODEL_FILEPATH;
  static int OBSERVATION_RAD;
  static SamplingStrategy SAMPLING_STRATEGY;
  static float SAMPLING_TEMPERTURE;

  AgentPolicy(const Instance *_ins, DistTable *_D, int seed = 0);
  ~AgentPolicy();

  void set_preferences(const Config &Q_from, const std::set<int> &A = {});
  void set_preferences_naive(const Config &Q_from, const std::set<int> &A = {});
  void set_preferences_learned(const Config &Q_from);

  // for model inference
  void set_features(const Config &Q);

  ActionCost get_action_cost(const int i, const Vertex *u);
  Vertex *get(const int i, const int k);
};
