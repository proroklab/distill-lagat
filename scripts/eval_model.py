from pathlib import Path
from loguru import logger
import hydra
from tqdm import tqdm
import numpy as np
import yaml
import time
import pandas as pd

from lagat.run_planners import run_neural_policy as run_model
from lagat.utils import set_global_seeds
from hydra.utils import instantiate
from lagat.run_planners import get_solution_quality


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="eval_model",
)
def main(cfg):
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    set_global_seeds(cfg.seed)
    logger.info(f"device={cfg.device}, seed={cfg.seed}, logdir={log_dir}")

    if cfg.save_animation:
        (log_dir / "animation").mkdir(parents=True, exist_ok=True)

    # problem definition
    grid_config_generator = instantiate(cfg.grid_config)

    # load model
    if cfg.model.fpath is None:
        target_dir = Path(__file__).parents[1] / "outputs/train"
        model_dir = sorted(
            [p for p in target_dir.iterdir() if p.is_dir() and p.stem.startswith("20")]
        )[-1]
        cfg.model.fpath = model_dir / "loss_best.pt"
        logger.info(f"model fpath is not specified, so use {cfg.model.fpath}")
    model = instantiate(cfg.model).to(cfg.device)
    model.eval()

    stats = dict()
    stats["num_samples"] = cfg.num_samples
    stats["num_success"] = 0
    stats["success_rate"] = 0
    results = []

    with tqdm(desc="online expert", total=cfg.num_samples) as pbar:
        for i in range(cfg.num_samples):
            t_s = time.perf_counter()
            env, all_actions, all_observations, all_terminated = run_model(
                model,
                grid_config_generator(np.random.randint(np.iinfo(np.int64).max)),
                **cfg.run_model,
            )
            # compute result
            t_elapsed_sec = time.perf_counter() - t_s
            if all(all_terminated[-1]):
                stats["num_success"] += 1
            stats["success_rate"] = stats["num_success"] / (i + 1)
            results.append(
                {
                    "index": i,
                    "num_agents": len(all_observations[0]),
                    "elapsed_sec": t_elapsed_sec,
                    **get_solution_quality(
                        all_actions, all_observations, all_terminated
                    ),
                }
            )
            if cfg.save_animation:
                env.save_animation(log_dir / f"animation/{i:06d}.svg")
            pbar.set_postfix(**stats)
            pbar.update(1)

    df = pd.DataFrame(results)
    df.to_csv(log_dir / "results.csv", index=False)
    stats["makespan_subopt"] = float(df.query("solved")["makespan_subopt"].mean())
    stats["sum_of_costs_subopt"] = float(
        df.query("solved")["sum_of_costs_subopt"].mean()
    )
    stats["elapsed_sec"] = float(df.query("solved")["elapsed_sec"].mean())

    logger.info("stats:")
    for key, val in stats.items():
        logger.info(f"  {key}\t{val}")
    # save results
    with open(log_dir / "stats.yaml", "w") as f:
        yaml.dump(stats, f)
    logger.info(f"save results in {log_dir}")


if __name__ == "__main__":
    main()
