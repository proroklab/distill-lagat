import argparse
import math
from pathlib import Path

import numpy as np
from pogema import GridConfig
from pogema.grid import Grid
from loguru import logger
from tqdm import tqdm

from lagat.grid_config_generator import get_obstacles_from_mapfile


def sample_starts_goals(
    obstacles,
    free_positions: list[list[int]],
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    num_agents = len(free_positions)
    grid_config = GridConfig(
        map=obstacles,
        num_agents=num_agents,
        seed=seed,
        observation_type="MAPF",
        on_target="nothing",
        possible_agents_xy=[pos[:] for pos in free_positions],
        possible_targets_xy=[pos[:] for pos in free_positions],
    )
    grid = Grid(grid_config)
    starts = grid.get_agents_xy(ignore_borders=True)
    goals = grid.get_targets_xy(ignore_borders=True)
    return starts, goals


def get_output_paths(output: Path, map_stem: str, num_instances: int) -> list[Path]:
    if num_instances == 1:
        return [output]
    if output.suffix == ".scen":
        return [
            output.with_name(f"{output.stem}-{i:03d}.scen")
            for i in range(num_instances)
        ]
    output.mkdir(parents=True, exist_ok=True)
    return [output / f"{map_stem}-{i:03d}.scen" for i in range(num_instances)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map",
        type=Path,
        required=True,
        help="MAPF benchmark format .map file",
    )
    parser.add_argument(
        "-n",
        "--num-instances",
        type=int,
        default=1,
        help="Number of .scen files to generate",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/scen"),
        help="Output directory for .scen files",
    )

    args = parser.parse_args()
    if not args.map.exists():
        raise FileNotFoundError(f"{args.map} does not exist")

    obstacles = get_obstacles_from_mapfile(args.map)
    if not obstacles:
        raise ValueError(f"Failed to parse map data from {args.map}")

    height = len(obstacles)
    width = len(obstacles[0])
    free_positions = [
        [y, x]
        for y, row in enumerate(obstacles)
        for x, cell in enumerate(row)
        if cell == 0
    ]
    free_cells = len(free_positions)

    num_instances = args.num_instances
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")

    rng = np.random.default_rng(args.seed)
    output_paths = get_output_paths(args.output_dir, args.map.stem, num_instances)
    logger.info(f"map={args.map} size={width}x{height} free_cells={free_cells}")
    logger.info(f"num_instances={num_instances} outputs={len(output_paths)}")
    logger.info("generating scenarios with num_agents=free_cells")
    for output_path in tqdm(output_paths, desc="generating .scen files"):
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        seed = int(rng.integers(np.iinfo(np.int64).max))
        starts, goals = sample_starts_goals(
            obstacles,
            free_positions=free_positions,
            seed=seed,
        )
        if len(starts) != free_cells or len(goals) != free_cells:
            raise RuntimeError(
                "Failed to sample starts/goals for all free cells. "
                f"starts={len(starts)} goals={len(goals)} free_cells={free_cells}"
            )
        with open(output_path, "w") as f:
            f.write("version 1\n")
            for idx, (start, goal) in enumerate(zip(starts, goals)):
                sy, sx = start
                gy, gx = goal
                d = math.hypot(sy - gy, sx - gx)
                row = (
                    f"{idx}\t{args.map.name}\t{width}\t{height}\t"
                    f"{sx}\t{sy}\t{gx}\t{gy}\t{d}\n"
                )
                f.write(row)

    logger.info(f"saved in {args.output_dir}")

if __name__ == "__main__":
    main()
