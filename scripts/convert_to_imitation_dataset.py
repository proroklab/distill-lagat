import pickle
from pathlib import Path
import hydra
import yaml
from glob import glob
from loguru import logger
from tqdm import tqdm
from joblib import Parallel, delayed


from lagat.features import get_pyg_datalist_from_one_instance


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="convert_to_imitation_dataset",
)
def main(cfg):
    logger.info("convert raw data to imitation learning dataset")
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    # retrieve instance files
    if cfg.dataset_dir is None:
        target_dir = Path(__file__).parents[1] / "outputs/raw_expert_trajectories"
        dataset_dir = sorted(
            [p for p in target_dir.iterdir() if p.is_dir() and p.stem.startswith("20")]
        )[-1]
        logger.info(f"dataset_dir is not specified, so use {dataset_dir}")
    else:
        dataset_dir = Path(cfg.dataset_dir)
    dataset_fpaths = list(
        sorted(map(Path, glob(f"{dataset_dir}/**/*.pkl", recursive=True)))
    )
    logger.info(f"found {len(dataset_fpaths)} files")
    for fpath in dataset_fpaths:
        logger.info(f"  {fpath}")

    # retrieve parameters
    dataset_cfg_fpath = dataset_fpaths[0].parent / ".hydra/config.yaml"
    logger.info(f"retrieve dataset config from {dataset_cfg_fpath}")
    with open(dataset_cfg_fpath) as f:
        dataset_cfg = yaml.safe_load(f)
    obs_radius = dataset_cfg["grid_config"]["obs_radius"]
    logger.info(f"{obs_radius=}")

    # convert raw data into pyg data
    logger.info("create pyg dataset")
    total_ins = 0
    total_config = 0
    for k, fpath in enumerate(dataset_fpaths):
        desc = f"{(k + 1):2d}/{len(dataset_fpaths)}: convert {fpath}"
        with open(fpath, "rb") as f:
            raw_dataset = pickle.load(f)
            total_ins += len(raw_dataset)

            def worker(ins):
                return get_pyg_datalist_from_one_instance(ins, obs_radius=obs_radius)

            pyg_dataset_list = [
                r
                for r in tqdm(
                    Parallel(return_as="generator", n_jobs=cfg.n_jobs)(
                        [delayed(worker)(ins) for ins in raw_dataset]
                    ),
                    desc=desc,
                    total=len(raw_dataset),
                )
            ]
            pyg_dataset = []
            for d in pyg_dataset_list:
                pyg_dataset.extend(d)

            total_config += len(pyg_dataset)
            dataset_fpath = log_dir / f"data_{k:08d}.pkl"
            with open(dataset_fpath, "wb") as f:
                pickle.dump(pyg_dataset, f)
                logger.info(f"save data in {dataset_fpath}")
    logger.info(
        f"retrieved {total_ins} instances, {total_config} configurations in total"
    )


if __name__ == "__main__":
    main()
