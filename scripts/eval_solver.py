from pathlib import Path
from loguru import logger
import hydra
from tqdm import tqdm
import pandas as pd
import subprocess
import re
from itertools import product
from joblib import Parallel, delayed
import shutil


def print_summary(csv_file: Path):
    df = pd.read_csv(csv_file)
    num_total = len(df)
    num_solved = (df["solved"] == 1).sum()
    if num_total == 0 or num_solved == 0:
        logger.info(f"solved: {num_solved}/{num_total}={num_solved / num_total:.3f}")
        return
    comp = df.loc[df.solved == 1, "comp_time"]
    logger.info(
        f"solved: {num_solved}/{num_total}={num_solved / num_total:.3f}\t"
        f"comp_time(ms): max={comp.max()}\tmean={comp.mean()}\tmed={comp.median()}"
    )


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="eval_solver",
)
def main(cfg):
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    logger.info(f"logdir={log_dir}")
    tmp_dir = log_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    scen_dir = Path(cfg.scen_dir)
    map_dir = Path(cfg.map_dir)

    result_keys = (
        "solved",
        "comp_time",
        "soc",
        "soc_lb",
        "makespan",
        "makespan_lb",
        "sum_of_loss",
        "sum_of_loss_lb",
    )

    def worker(k, map_file, scen_file, N, seed):
        out_file = tmp_dir / f"res-{k}.txt"
        cmd = [
            cfg.exec_file,
            "-m",
            str(map_file),
            "-i",
            str(scen_file),
            "-N",
            str(N),
            "-o",
            str(out_file),
            "-t",
            str(cfg["time_limit_sec"]),
            "-s",
            str(seed),
        ] + cfg.get("solver_options", [])
        try:
            subprocess.run(cmd)
        except Exception:
            pass

        row = dict(
            solver=cfg.solver_name,
            num_agents=N,
            map_name=map_file.name,
            scen=scen_file.name,
            seed=seed,
        )
        for key in result_keys:
            row[key] = 0

        if out_file.is_file():
            with open(out_file) as f:
                for line in f:
                    for key in result_keys:
                        m = re.search(rf"{key}=(\d+(?:\.\d+)?)", line)
                        if m:
                            row[key] = int(m.group(1))
        return row

    params_lst = []
    for map_name in cfg.maps:
        map_file = map_dir / f"{map_name}.map"
        scen_files = sorted(scen_dir.glob(f"{map_name}*.scen"))
        num_agents = list(
            range(cfg.num_min_agents, cfg.num_max_agents + 1, cfg.num_interval_agents)
        )
        seeds = list(range(cfg.seed_start, cfg.seed_end + 1))
        params_lst.extend(product([map_file], scen_files, num_agents, seeds))

    params_lst = [(k, *params) for k, params in zip(range(len(params_lst)), params_lst)]
    logger.info(f"test {len(params_lst)} instances")

    results = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=cfg.n_jobs)(
                [delayed(worker)(*p) for p in params_lst]
            ),
            desc="benchmarking",
            total=len(params_lst),
        )
    ]

    csv_file = log_dir / "results.csv"
    pd.DataFrame(results).to_csv(csv_file, index=False)
    print_summary(csv_file)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"save results in {log_dir}")


if __name__ == "__main__":
    main()
