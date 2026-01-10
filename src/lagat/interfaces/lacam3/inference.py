import ctypes
import numpy as np
from pathlib import Path
from typing import Literal

from lagat.interfaces.cpp_build_config import ensure_lib_exists, resolve_cpp_lib_path
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig

import platform

if platform.system() == "Darwin":  # macOS
    LACAM_LIB_FILENAME = "liblacam.dylib"
elif platform.system() == "Linux":
    LACAM_LIB_FILENAME = "liblacam.so"
else:
    raise RuntimeError("Unsupported OS")


calling_script_dir = Path(__file__).parent
lib_lacam_path = resolve_cpp_lib_path(
    calling_script_dir, "lacam3", "interfaces/lacam3", LACAM_LIB_FILENAME
)
ensure_lib_exists(lib_lacam_path, "lacam3")


class LacamLib:
    def __init__(self):
        self.load_library()

    def load_library(self):
        self._lacam_lib = ctypes.CDLL(lib_lacam_path)

        self._lacam_lib.run_lacam.argtypes = [
            ctypes.c_char_p,  # map_name
            ctypes.c_char_p,  # scene_name
            ctypes.c_int,  # N
            ctypes.c_float,  # time_limit_sec
            ctypes.c_int,  # seed
            ctypes.c_int,  # verbose
            ctypes.c_int,  # flg_no_star
            ctypes.c_int,  # pibt_num
            ctypes.c_int,  # refiner_num
            ctypes.c_int,  # flg_no_scatter
            ctypes.c_int,  # scatter_margin
            ctypes.c_float,  # random_insert_prob1
            ctypes.c_float,  # random_insert_prob2
            ctypes.c_int,  # flg_random_insert_init_node
            ctypes.c_float,  # recursive_rate
            ctypes.c_int,  # recursive_time_limit
        ]
        self._lacam_lib.run_lacam.restype = ctypes.c_char_p

    def run_lacam(self, map_file_content, scene_file_content, num_agents, cfg):
        map_file_bytes = map_file_content.encode("utf-8")
        scenario_file_bytes = scene_file_content.encode("utf-8")

        for time_limit_sec in cfg.timeouts:
            result = self._lacam_lib.run_lacam(
                map_file_bytes,
                scenario_file_bytes,
                ctypes.c_int(num_agents),
                ctypes.c_float(time_limit_sec),
                ctypes.c_int(cfg.seed),
                ctypes.c_int(cfg.verbose),
                ctypes.c_int(int(cfg.flg_no_star)),
                ctypes.c_int(cfg.pibt_num),
                ctypes.c_int(cfg.refiner_num),
                ctypes.c_int(int(cfg.flg_no_scatter)),
                ctypes.c_int(cfg.scatter_margin),
                ctypes.c_float(cfg.random_insert_prob1),
                ctypes.c_float(cfg.random_insert_prob2),
                ctypes.c_int(int(cfg.flg_random_insert_init_node)),
                ctypes.c_float(cfg.recursive_rate),
                ctypes.c_int(cfg.recursive_time_limit),
            )

            try:
                result_str = result.decode("utf-8")
            except Exception as e:
                print(f"Exception occured while running LaCAM: {e}")
                raise e

            if "ERROR" in result_str:
                print(
                    "LaCAM failed to find path with "
                    f"time_limit_sec={time_limit_sec} | {result_str}"
                )
                continue

            return True, result_str

        return False, None


class Lacam3InferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["LaCAM"] = "LaCAM"
    timeouts: list[float] = [1.0, 5.0, 10.0, 60.0]
    seed: int = 0
    verbose: int = 0
    flg_no_star: bool = False
    pibt_num: int = 10
    refiner_num: int = 4
    flg_no_scatter: bool = False
    scatter_margin: int = 10
    random_insert_prob1: float = 0.001
    random_insert_prob2: float = 0.01
    flg_random_insert_init_node: bool = False
    recursive_rate: float = 0.0  # for CPU management
    recursive_time_limit: int = 1


class Lacam3Inference:
    def __init__(self, cfg: Lacam3InferenceConfig):
        self.cfg = cfg
        self.lacam_lib = LacamLib()
        self.output_data = None
        self.step = 1
        self.timed_out = False
        self._moves = np.array(GridConfig().MOVES)

    def _parse_data(self, data):
        if data is None:
            return None
        lines = data.strip().split("\n")
        columns = None

        for line in lines:
            tuples = [
                tuple(map(int, item.split(",")))
                for item in line.strip().split("|")
                if item
            ]
            if len(tuples) == 0:
                return None
            if columns is None:
                columns = [[] for _ in range(len(tuples))]
            for i, t in enumerate(tuples):
                columns[i].append(t[::-1])

        return columns

    def _get_next_move_single_agent(self, agent_id, step):
        agent_path = self.output_data[agent_id]
        if step >= len(agent_path):
            return 0

        old_pos = agent_path[step - 1]
        new_pos = agent_path[step]
        move = np.array(new_pos) - np.array(old_pos)
        return np.nonzero(np.all(self._moves == move, axis=-1))[0].item()

    def _get_next_move(self, step, num_agents):
        return [
            self._get_next_move_single_agent(agent_id, step)
            for agent_id in range(num_agents)
        ]

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        num_agents = len(observations)
        if self.output_data is None:
            if self.timed_out:
                return [0] * num_agents

            map_array = np.array(observations[0]["global_obstacles"])
            agent_starts_xy = [obs["global_xy"] for obs in observations]
            agent_targets_xy = [obs["global_target_xy"] for obs in observations]

            def map_row(row):
                return "".join("@" if x else "." for x in row)

            map_content = "\n".join(map_row(row) for row in map_array)
            map_file_content = "type octile"
            map_file_content += f"\nheight {map_array.shape[0]}"
            map_file_content += f"\nwidth {map_array.shape[1]}"
            map_file_content += f"\nmap\n{map_content}"

            task_file_content = "version 1\n"
            for idx, (start_xy, target_xy) in enumerate(
                zip(agent_starts_xy, agent_targets_xy)
            ):
                task_file_content += (
                    f"{idx}	tmp.map	{map_array.shape[0]}	{map_array.shape[1]}	"
                )
                task_file_content += (
                    f"{start_xy[1]}	{start_xy[0]}	"
                    f"{target_xy[1]}	{target_xy[0]}	1\n"
                )

            solved, lacam_results = self.lacam_lib.run_lacam(
                map_file_content,
                task_file_content,
                num_agents,
                self.cfg,
            )
            if solved:
                self.output_data = self._parse_data(lacam_results)
            if self.output_data is None:
                self.timed_out = True
                return [0] * num_agents

        actions = self._get_next_move(self.step, num_agents)
        self.step += 1
        return actions

    def reset_states(self):
        self.output_data = None
        self.step = 1
        self.timed_out = False
