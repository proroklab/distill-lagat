from pathlib import Path
import yaml

from lagat.features import get_cost_to_go_matrices, get_pyg_data_from_observations
from pogema import pogema_v0, GridConfig
import torch
from hydra.utils import instantiate
from torch_geometric.nn import summary

torch.set_printoptions(precision=2, linewidth=200)


@torch.no_grad()
def test_model():
    # data preparation
    obs_radius = 5
    grid_config = GridConfig(
        num_agents=2,
        obs_radius=obs_radius,
        seed=0,
        observation_type="MAPF",
        on_target="nothing",
        density=0.2,
    )
    env = pogema_v0(grid_config=grid_config)
    obs, infos = env.reset()
    cost_to_go_matrices = get_cost_to_go_matrices(obs)
    data = get_pyg_data_from_observations(
        obs,
        obs_radius=obs_radius,
        cost_to_go_matrices=cost_to_go_matrices,
    )

    model_cfg_file = Path(__file__).parents[1] / "scripts/conf/model/gnn_planner.yaml"
    with open(model_cfg_file) as f:
        model_cfg = yaml.safe_load(f)

    model = instantiate(model_cfg)
    model.eval()
    print(model_cfg)
    print(summary(model, data))
