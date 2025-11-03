from lagat.features import get_cost_to_go_matrices, get_pyg_data_from_observations
from pogema import pogema_v0, GridConfig, AnimationMonitor
import torch

torch.set_printoptions(precision=2, linewidth=200)


def test_feature():
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
    env = AnimationMonitor(env)
    obs, infos = env.reset()
    print()
    env.render()
    cost_to_go_matrices = get_cost_to_go_matrices(obs)
    data = get_pyg_data_from_observations(
        obs,
        obs_radius=obs_radius,
        cost_to_go_matrices=cost_to_go_matrices,
    )
    print(data)
    print("node_feature=")
    print(data.x)
    print("edge_index=")
    print(data.edge_index)
    print("edge_attr=")
    print(data.edge_attr)
