#include <filesystem>
#include <fstream>
#include <lacam.hpp>

const auto tmpdir = std::filesystem::temp_directory_path();
const auto map_name = tmpdir / "tmp.map";
const auto scene_name = tmpdir / "tmp.scene";

extern "C" {
const char* run_lacam(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec, int seed, int verbose,
                      int flg_no_star, int pibt_num, int refiner_num,
                      int flg_no_scatter, int scatter_margin,
                      float random_insert_prob1, float random_insert_prob2,
                      int flg_random_insert_init_node, float recursive_rate,
                      int recursive_time_limit);
}

const char* run_lacam(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec, int seed, int verbose,
                      int flg_no_star, int pibt_num, int refiner_num,
                      int flg_no_scatter, int scatter_margin,
                      float random_insert_prob1, float random_insert_prob2,
                      int flg_random_insert_init_node, float recursive_rate,
                      int recursive_time_limit)
{
  std::string map_content(map_content_cstr);
  std::string scene_content(scene_content_cstr);

  const bool flg_no_star_bool = flg_no_star != 0;
  const bool flg_no_scatter_bool = flg_no_scatter != 0;
  const bool flg_random_insert_init_node_bool =
      flg_random_insert_init_node != 0;

  // setup instance
  std::ofstream ins_file;
  ins_file.open(scene_name, std::ios::out);
  ins_file << scene_content;
  ins_file.close();

  // setup instance
  std::ofstream map_file;
  map_file.open(map_name, std::ios::out);
  map_file << map_content;
  map_file.close();

  const auto ins = Instance(scene_name, map_name, N);
  if (!ins.is_valid(1)) return "ERROR_SCENE";

  // solver parameters
  Planner::FLG_SWAP = true;
  Planner::FLG_STAR = !flg_no_star_bool;
  Planner::FLG_MULTI_THREAD = true;
  Planner::PIBT_NUM = pibt_num;
  Planner::FLG_REFINER = true;
  Planner::REFINER_NUM = refiner_num;
  Planner::FLG_SCATTER = !flg_no_scatter_bool;
  Planner::SCATTER_MARGIN = scatter_margin;
  Planner::RANDOM_INSERT_PROB1 = random_insert_prob1;
  Planner::RANDOM_INSERT_PROB2 = random_insert_prob2;
  Planner::FLG_RANDOM_INSERT_INIT_NODE =
      flg_random_insert_init_node_bool;
  Planner::RECURSIVE_RATE = recursive_rate;
  Planner::RECURSIVE_TIME_LIMIT = recursive_time_limit;

  // solve
  const auto deadline = Deadline(time_limit_sec * 1000);
  const auto solution = solve(ins, verbose - 1, &deadline, seed);

  // failure
  if (solution.empty()) {
    info(1, verbose, &deadline, "failed to solve");
    return "ERROR_EMPTY";
  }

  // check feasibility
  if (!is_feasible_solution(ins, solution, verbose)) {
    info(0, verbose, &deadline, "invalid solution");
    return "ERROR_SOLUTION";
  }

  auto get_x = [&](int k) { return k % ins.G->width; };
  auto get_y = [&](int k) { return k / ins.G->width; };

  std::ostringstream result_string;
  for (size_t t = 0; t < solution.size(); ++t) {
    auto C = solution[t];
    for (auto v : C) {
      result_string << get_x(v->index) << "," << get_y(v->index) << "|";
    }
    result_string << "\n";
  }

  static std::string result;
  result = result_string.str();

  return result.c_str();
}
