import os
import pickle
from loguru import logger
from glob import glob
import random

import hydra
from hydra.utils import call
from omegaconf import OmegaConf
from pathlib import Path
import wandb
from tqdm import tqdm
import torch
import numpy as np

from torch_geometric.nn import summary
from lagat.utils import set_global_seeds
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from lagat.features import get_pyg_datalist_from_one_instance
from hydra.utils import instantiate
from lagat.grid_config_generator import generate_grid_config_from_env
from lagat.run_planners import run_neural_policy as run_model
from lagat.run_planners import run_full_horizon_planner as run_expert


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="train",
)
def main(cfg):
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    # setup wandb
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project=cfg.project,
        dir=log_dir,
        name=str(log_dir).split("/")[-1],
        mode="online" if cfg.wandb else "disabled",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    set_global_seeds(cfg.seed)
    logger.info(f"device={cfg.device}, seed={cfg.seed}, logdir={log_dir}")

    if cfg.save_animation:
        (log_dir / "animation").mkdir(parents=True, exist_ok=True)

    # load dataset
    if cfg.dataset_dir is None:
        target_dir = Path(__file__).parents[1] / "outputs/imitation_learning_dataset"
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

    with open(dataset_fpaths[0], "rb") as f:
        dataset_example = pickle.load(f)
    jit_trace_data = dict(
        [
            (k, dataset_example[0][k].clone().to("cpu"))
            for k in ["x", "edge_index", "edge_attr"]
        ]
    )

    obs_radius = (dataset_example[0].x.size(-1) - 1) // 2
    logger.info(f"{obs_radius=}")

    # define model
    model = call(cfg.model, obs_radius=obs_radius).to(cfg.device)
    print(summary(model, dataset_example[0].clone().to(cfg.device)))
    model = torch.compile(model)  # speed up

    # learning metrics
    optimizer = call(cfg.optimizer, model.parameters())
    scheduler = call(cfg.scheduler, optimizer)

    # data aggregation by online expert
    oe = cfg.online_evaluation
    expert = instantiate(cfg.expert)
    grid_config_generator = instantiate(cfg.grid_config)

    # stats
    stats = dict()
    stats["loss/best"] = np.inf
    stats["acc/best"] = 0
    stats["success/best"] = 0
    stats["dataset_size/train"] = 0

    pyg_dataset_oe = []

    def step_online_evaluation(epoch: int):
        if epoch % oe.every_epoch != 0 or oe.use is False:
            return

        nonlocal pyg_dataset_oe
        stats["oe/success_rate"] = 0
        stats["oe/num_model_success"] = 0
        stats["oe/num_new_instances"] = 0
        stats["oe/num_new_configurations"] = 0
        stats["oe/num_expert_failure"] = 0
        with tqdm(desc="online evaluation", total=oe.num_samples, leave=False) as pbar:
            for i in range(oe.num_samples):
                env, all_actions, all_observations, all_terminated = run_model(
                    model,
                    grid_config_generator(np.random.randint(np.iinfo(np.int64).max)),
                    **oe.run_model,
                )
                if cfg.save_animation:
                    target_dir = log_dir / f"animation/{epoch:06d}"
                    target_dir.mkdir(exist_ok=True)
                    env.save_animation(target_dir / f"{i:06d}.svg")

                if all(all_terminated[-1]):
                    stats["oe/num_model_success"] += 1
                elif oe.dataset_aggregation:
                    # online dataset aggregation by running the expert planner
                    env, all_actions, all_observations, all_terminated = run_expert(
                        expert, generate_grid_config_from_env(env)
                    )
                    if all(all_terminated[-1]):
                        pyg_data = get_pyg_datalist_from_one_instance(
                            (all_observations, all_actions, all_terminated),
                            obs_radius=model.obs_radius,
                        )
                        pyg_dataset_oe.extend(pyg_data)
                        stats["oe/num_new_instances"] += 1
                        stats["oe/num_new_configurations"] += len(pyg_data)
                    else:
                        stats["oe/num_expert_failure"] += 1
                stats["oe/success_rate"] = stats["oe/num_model_success"] / (i + 1)
                msg = {k: v for k, v in stats.items() if k.startswith("oe")}
                pbar.set_postfix(**msg)
                pbar.update(1)
        stats["success/val"] = stats["oe/num_model_success"] / oe.num_samples

    @torch.no_grad()
    def update_best():
        model.eval()
        for metric in ["loss", "acc", "success"]:
            if (
                metric in ["acc", "success"]
                and stats.get(f"{metric}/val", 0) > stats[f"{metric}/best"]
            ) or (
                metric in ["loss"]
                and stats.get(f"{metric}/val", np.inf) < stats[f"{metric}/best"]
            ):
                stats[f"{metric}/best"] = stats[f"{metric}/val"]
                model.save(log_dir / f"{metric}_best.pt")
                model_copy = model.reconstruct(log_dir / f"{metric}_best.pt")
                model_copy.eval()
                jit_model = torch.jit.trace(model_copy, jit_trace_data)
                jit_model.save(log_dir / f"{metric}_best_jit.pt")

    # main loop
    logger.info(f"start training for {cfg.num_epochs} epochs")
    with tqdm(desc="training", total=cfg.num_epochs) as pbar:
        for epoch in range(1, cfg.num_epochs + 1):
            losses_train, accuracies_train = [], []
            losses_val, accuracies_val = [], []
            random.shuffle(dataset_fpaths)
            random.shuffle(pyg_dataset_oe)
            stats["dataset_size/train"] = 0
            for dataset_idx, fpath in enumerate(dataset_fpaths):
                idx_str = f"{dataset_idx + 1:6d}/{len(dataset_fpaths)}"

                # dataset preparation
                with open(fpath, "rb") as f:
                    pyg_dataset = pickle.load(f)
                dataset_train, dataset_val = call(
                    cfg.dataset_split,
                    pyg_dataset,
                    # prevent data leak
                    generator=torch.Generator().manual_seed(cfg.seed),
                )
                if len(dataset_train) == 0 or len(dataset_val) == 0:
                    continue
                # add online aggregation data to training dataset
                L1, L2 = len(dataset_fpaths), len(pyg_dataset_oe)
                oe_idx_from = int((dataset_idx / L1) * L2)
                oe_idx_to = int(((dataset_idx + 1) / L1) * L2)
                # create data loader
                dataloader_train = call(
                    cfg.dataloader,
                    ConcatDataset(
                        [dataset_train, pyg_dataset_oe[oe_idx_from:oe_idx_to]]
                    ),
                )
                dataloader_val = call(cfg.dataloader, dataset_val)

                # training
                desc = f"{idx_str} step_train, dataset_size={len(dataset_train)}"
                for batch in tqdm(dataloader_train, desc=desc, leave=False):
                    batch = batch.to(cfg.device)
                    logits = model(batch)
                    loss = F.cross_entropy(logits, batch.y)
                    acc = (torch.argmax(logits, dim=1) == batch.y).float().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses_train.append(loss.item())
                    accuracies_train.append(acc.item())
                    stats["dataset_size/train"] += len(batch)

                # validation
                desc = f"{idx_str} step_val, dataset_size={len(dataset_val)}"
                with torch.no_grad():
                    for batch in tqdm(dataloader_val, desc=desc, leave=False):
                        batch = batch.to(cfg.device)
                        logits = model(batch)
                        loss = F.cross_entropy(logits, batch.y)
                        acc = (torch.argmax(logits, dim=1) == batch.y).float().mean()
                        losses_val.append(loss.item())
                        accuracies_val.append(acc.item())

            # online evaluation
            step_online_evaluation(epoch)

            stats["loss/train"] = np.mean(losses_train)
            stats["acc/train"] = np.mean(accuracies_train)
            stats["loss/val"] = np.mean(losses_val)
            stats["acc/val"] = np.mean(accuracies_val)

            scheduler.step()
            update_best()
            # update log
            wandb.log(stats)
            msg = {k: v for k, v in stats.items() if not k.startswith("oe")}
            msg["lr"] = scheduler.get_last_lr()[0]
            pbar.set_postfix(**msg)
            pbar.update(1)

    logger.info(f"save results in {log_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
