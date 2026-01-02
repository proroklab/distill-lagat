#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <planner.hpp>
#include <policy.hpp>
#include <utils.hpp>

const auto tmpdir = std::filesystem::temp_directory_path();
const auto map_name = tmpdir / "tmp.map";
const auto scene_name = tmpdir / "tmp.scene";

extern "C" {
const char* run_lagat(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec, int seed,
                      const char* model_path_cstr, int deadlock_detection,
                      int deadlock_depth, int lns, int plns_num_refiners);
}

const char* run_lagat(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec, int seed,
                      const char* model_path_cstr, int deadlock_detection,
                      int deadlock_depth, int lns, int plns_num_refiners)
{
  std::string map_content(map_content_cstr);
  std::string scene_content(scene_content_cstr);
  std::string model_path = model_path_cstr ? model_path_cstr : "";

  const int verbose = 10;

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

  AgentPolicy::MODEL_FILEPATH = model_path;

  LaCAM::DEADLOCK_DETECTION = deadlock_detection != 0;
  LaCAM::DEADLOCK_DEPTH = deadlock_depth;
  LNS::ON = lns != 0;
  PLNS::NUM_REFINERS = plns_num_refiners;

  // solve
  auto deadline = Deadline(time_limit_sec * 1000);
  set_print_options(deadline, verbose);
  const auto solution = solve(ins, verbose - 1, &deadline, seed);

  // failure
  if (solution.empty()) return "ERROR_EMPTY";

  // check feasibility
  if (!is_feasible_solution(ins, solution)) return "ERROR_SOLUTION";

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
