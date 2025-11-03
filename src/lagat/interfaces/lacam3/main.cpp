#include <fstream>
#include <iostream>
#include <lacam.hpp>
#include <filesystem>

const auto tmpdir = std::filesystem::temp_directory_path();
const auto map_name = tmpdir / "tmp.map";
const auto scene_name = tmpdir / "tmp.scene";

extern "C" {
const char* run_lacam(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec);
}

const char* run_lacam(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec)
{
  std::string map_content(map_content_cstr);
  std::string scene_content(scene_content_cstr);

  const int seed = 0;
  const int verbose = 0;

  // Solver parameters
  const bool flg_no_all = false;
  const bool flg_no_star = false;
  const bool flg_no_swap = false;
  const bool flg_no_multi_thread = false;
  const int pibt_num = 10;
  const bool flg_no_refiner = false;
  const int refiner_num = 4;
  const bool flg_no_scatter = false;
  const int scatter_margin = 10;
  const float random_insert_prob1 = 0.001f;
  const float random_insert_prob2 = 0.01f;
  const bool flg_random_insert_init_node = false;
  const float recursive_rate = 0.2f;
  const int recursive_time_limit = 1000;
  const int checkpoints_duration = 5000;

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
  Planner::FLG_SWAP = !flg_no_swap && !flg_no_all;
  Planner::FLG_STAR = !flg_no_star && !flg_no_all;
  Planner::FLG_MULTI_THREAD = !flg_no_multi_thread && !flg_no_all;
  Planner::PIBT_NUM = flg_no_all ? 1 : pibt_num;
  Planner::FLG_REFINER = !flg_no_refiner && !flg_no_all;
  Planner::REFINER_NUM = refiner_num;
  Planner::FLG_SCATTER = !flg_no_scatter && !flg_no_all;
  Planner::SCATTER_MARGIN = scatter_margin;
  Planner::RANDOM_INSERT_PROB1 = flg_no_all ? 0 : random_insert_prob1;
  Planner::RANDOM_INSERT_PROB2 = flg_no_all ? 0 : random_insert_prob2;
  Planner::FLG_RANDOM_INSERT_INIT_NODE =
      flg_random_insert_init_node && !flg_no_all;
  Planner::RECURSIVE_RATE = flg_no_all ? 0 : recursive_rate;
  Planner::RECURSIVE_TIME_LIMIT = flg_no_all ? 0 : recursive_time_limit;
  Planner::CHECKPOINTS_DURATION = checkpoints_duration;

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
