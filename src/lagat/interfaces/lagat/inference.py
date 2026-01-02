import ctypes
import numpy as np
from pathlib import Path
from typing import Literal, Optional

from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig

import platform
import subprocess

if platform.system() == "Darwin":  # macOS
    LAGAT_LIB_FILENAME = "libplanner.dylib"
elif platform.system() == "Linux":
    LAGAT_LIB_FILENAME = "libplanner.so"
else:
    raise RuntimeError("Unsupported OS")


calling_script_dir = Path(__file__).parent
lib_lagat_path = calling_script_dir / "build" / LAGAT_LIB_FILENAME

if not lib_lagat_path.exists():
    cmake_cmd = ["cmake", "-B", "build"]
    subprocess.run(cmake_cmd, check=True, cwd=calling_script_dir)
    make_cmd = ["make", "-C", "build", "-j8"]
    subprocess.run(make_cmd, check=True, cwd=calling_script_dir)


class LagatLib:
    def __init__(self):
        self.load_library()

    def load_library(self):
        self._lagat_lib = ctypes.CDLL(lib_lagat_path)

        self._lagat_lib.run_lagat.argtypes = [
            ctypes.c_char_p,  # map_name
            ctypes.c_char_p,  # scene_name
            ctypes.c_int,  # N
            ctypes.c_float,  # time_limit_sec
            ctypes.c_int,  # seed
            ctypes.c_char_p,  # model_path
            ctypes.c_int,  # deadlock_detection
            ctypes.c_int,  # deadlock_depth
            ctypes.c_int,  # lns
            ctypes.c_int,  # plns_num_refiners
        ]
        self._lagat_lib.run_lagat.restype = ctypes.c_char_p

    def run_lagat(
        self,
        map_file_content,
        scene_file_content,
        num_agents,
        cfg,
    ):
        map_file_bytes = map_file_content.encode("utf-8")
        scenario_file_bytes = scene_file_content.encode("utf-8")
        model_path = str(cfg.model_path) if cfg.model_path else ""

        result = self._lagat_lib.run_lagat(
            map_file_bytes,
            scenario_file_bytes,
            ctypes.c_int(num_agents),
            ctypes.c_float(cfg.time_limit),
            ctypes.c_int(cfg.seed),
            model_path.encode("utf-8"),
            ctypes.c_int(int(cfg.deadlock_detection)),
            ctypes.c_int(cfg.deadlock_depth),
            ctypes.c_int(int(cfg.lns)),
            ctypes.c_int(cfg.plns_num_refiners),
        )

        try:
            result_str = result.decode("utf-8")
        except Exception as e:
            print(f"Exception occured while running LaGAT: {e}")
            raise e

        if "ERROR" in result_str:
            print(f"LaGAT failed to find path | {result_str}")
            return False, None

        return True, result_str


class LagatInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["LaGAT"] = "LaGAT"
    time_limit: float = 3.0
    seed: int = 0
    model_path: Optional[Path] = None
    deadlock_detection: bool = True
    deadlock_depth: int = 3
    lns: bool = True
    plns_num_refiners: int = 4


class LagatInference:
    def __init__(self, cfg: LagatInferenceConfig):
        self.cfg = cfg
        self.lagat_lib = LagatLib()
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

            solved, lagat_results = self.lagat_lib.run_lagat(
                map_file_content,
                task_file_content,
                num_agents,
                self.cfg,
            )
            if solved:
                self.output_data = self._parse_data(lagat_results)
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
