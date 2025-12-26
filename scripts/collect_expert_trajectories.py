import pickle
import numpy as np
from tqdm import tqdm
from loguru import logger


import hydra
from hydra.utils import instantiate
from pathlib import Path

from lagat.run_planners import run_full_horizon_planner as run_expert


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="collect_expert_trajectories",
)
def main(cfg):
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    if cfg.save_animation:
        (log_dir / "animation").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    expert = instantiate(cfg.expert)
    grid_config_generator = instantiate(cfg.grid_config)

    logger.info(f"start solving {cfg.num_samples} instances")
    dataset = []
    result_file_num = 0
    num_data = 0
    num_success = 0

    def save():
        nonlocal result_file_num, dataset
        with open(log_dir / f"data_{result_file_num:08d}.pkl", "wb") as f:
            pickle.dump(dataset, f)
            result_file_num += 1
            dataset = []

    num_instances = 0
    with tqdm(total=cfg.num_samples) as pbar:
        while num_success < cfg.num_samples:
            num_instances += 1
            seed = rng.integers(np.iinfo(np.int64).max)
            grid_config = grid_config_generator(seed)
            env, all_actions, all_observations, all_terminated = run_expert(
                expert,
                grid_config=grid_config,
                animation=cfg.save_animation,
            )
            if cfg.save_animation:
                env.save_animation(log_dir / f"animation/{num_instances:06d}.svg")
            success = all(all_terminated[-1])
            if success:
                num_data += len(all_observations)
                dataset.append((all_observations, all_actions, all_terminated))
                num_success += 1
            if len(dataset) >= cfg.dataset_size:
                save()
            pbar.set_postfix(
                success_rate=f"{num_success / num_instances:.2f}",
                num_data=num_data,
                num_instances=num_instances,
                result_file_num=result_file_num,
            )
            pbar.update(1 if success else 0)

    logger.info(
        f"{cfg.num_samples} samples ({num_data} configurations) were successfully added to the dataset"
    )
    save()
    logger.info(f"save data in {log_dir}")


if __name__ == "__main__":
    main()
