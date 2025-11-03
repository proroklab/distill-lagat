from typing import Union
from pathlib import Path

from dataclasses import dataclass


import numpy as np
from pogema import GridConfig
from pogema_toolbox.generators.random_generator import MapRangeSettings, generate_map
from pogema_toolbox.generators.maze_generator import MazeGenerator, MazeRangeSettings


@dataclass
class PredefinedMapRangeSettings:
    map_fpaths: list[str]

    def sample(self, seed: int) -> Path:
        rng = np.random.default_rng(seed)
        return dict(map_fpath=Path(rng.choice(self.map_fpaths)))


RangeSettings = Union[MapRangeSettings, MazeRangeSettings, PredefinedMapRangeSettings]


def generate_grid_config_from_env(env, max_episode_steps=None):
    config = env.grid.config
    if max_episode_steps is None:
        max_episode_steps = config.max_episode_steps

    return GridConfig(
        num_agents=config.num_agents,  # number of agents
        size=config.size,  # size of the grid
        density=config.density,  # obstacle density
        seed=config.seed,
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=config.obs_radius,  # defines field of view
        observation_type=config.observation_type,
        collision_system=config.collision_system,
        on_target=config.on_target,
        map=env.grid.get_obstacles(ignore_borders=True).tolist(),
        agents_xy=env.grid.get_agents_xy(ignore_borders=True),
        targets_xy=env.grid.get_targets_xy(ignore_borders=True),
    )


def get_obstacles_from_mapfile(map_fpath: Path | str) -> list[list[int]]:
    with open(map_fpath) as f:
        lines = f.readlines()

    map_start = None
    for i, line in enumerate(lines):
        if line.strip() == "map":
            map_start = i + 1
            break

    if map_start is None:
        raise ValueError(f"No 'map' section found in {map_fpath}")

    lines = lines[map_start:]
    obstacles = []
    for line in lines:
        obs_line = []
        for cell in line.strip():
            if cell == "@" or cell == "T" or cell == "O":
                obs_line.append(1)
            else:
                obs_line.append(0)
        obstacles.append(obs_line)

    return obstacles


def grid_config_generator_factory(
    cands_num_agents: list[int],
    map_type_probs: list[float],
    map_types: list[RangeSettings],
    **kwargs,
):
    def fn(seed):
        rng = np.random.default_rng(seed)
        num_agents = rng.choice(cands_num_agents)
        range_setting = rng.choice(map_types)
        setting = range_setting.sample(seed)
        if isinstance(range_setting, MazeRangeSettings):
            grid_map = MazeGenerator.generate_maze(**setting)
        elif isinstance(range_setting, MapRangeSettings):
            grid_map = generate_map(setting)
        elif isinstance(range_setting, PredefinedMapRangeSettings):
            grid_map = get_obstacles_from_mapfile(**setting)
        else:
            raise ValueError()

        return GridConfig(
            seed=seed,
            num_agents=num_agents,
            observation_type="MAPF",
            on_target="nothing",
            map=grid_map,
            **kwargs,
        )

    return fn
