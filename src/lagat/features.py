import numpy as np
from collections import deque
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from scipy.spatial.distance import squareform, pdist


def get_cost_to_go_matrix(
    global_obstacles: np.ndarray, goal: tuple[int, int]
) -> np.ndarray:
    cost_to_go = np.full(global_obstacles.shape, np.inf)
    cost_to_go[goal] = 0
    Q = deque([np.array(goal)])
    action_primitives = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    while len(Q) > 0:
        u = Q.popleft()
        d = cost_to_go[tuple(u)]
        for v in u + action_primitives:
            x, y = v
            if (
                0 <= x < cost_to_go.shape[0]
                and 0 <= y < cost_to_go.shape[1]
                and global_obstacles[x, y] == 0
                and d + 1 < cost_to_go[x, y]
            ):
                cost_to_go[x, y] = d + 1
                Q.append(v)
    return cost_to_go


def get_cost_to_go_matrices(onestep_observations) -> np.ndarray:
    return np.array(
        [
            get_cost_to_go_matrix(obs["global_obstacles"], obs["global_target_xy"])
            for obs in onestep_observations
        ]
    )


def get_node_feature_from_observations(
    observations,
    obs_radius: int,
    cost_to_go_matrices: np.ndarray,
) -> np.ndarray:
    fov_size = obs_radius * 2 + 1
    node_features = []
    for agent_idx, observation in enumerate(observations):
        obs_agent = np.array(observation["agents"])
        obs_cost_to_go = np.ones((fov_size, fov_size))

        centre_x, centre_y = obs_radius, obs_radius
        g_agent_x, g_agent_y = observation["global_xy"]
        g_goal_x, g_goal_y = observation["global_target_xy"]

        # cost to go
        cost_to_go_matrix = cost_to_go_matrices[agent_idx]
        d_norm = cost_to_go_matrix[(g_agent_x, g_agent_y)]
        x_min = max(0, g_agent_x - obs_radius)
        x_max = min(cost_to_go_matrix.shape[0] - 1, g_agent_x + obs_radius + 1)
        y_min = max(0, g_agent_y - obs_radius)
        y_max = min(cost_to_go_matrix.shape[1] - 1, g_agent_y + obs_radius + 1)
        C = np.clip(
            (cost_to_go_matrix[x_min:x_max, y_min:y_max] - d_norm) / (2 * obs_radius),
            None,
            1,
        )
        x_d = centre_x - (g_agent_x - x_min)
        y_d = centre_y - (g_agent_y - y_min)
        obs_cost_to_go[x_d : x_d + C.shape[0], y_d : y_d + C.shape[1]] = C

        node_features.append(np.stack([obs_cost_to_go, obs_agent]))
    node_features = np.stack(node_features)
    return node_features


def get_pyg_data_from_observations(
    observations,
    obs_radius: int,
    actions=None,
    cost_to_go_matrices: np.ndarray | None = None,
) -> np.ndarray:
    if cost_to_go_matrices is None:
        cost_to_go_matrices = get_cost_to_go_matrices(observations)

    # node feature
    node_features = get_node_feature_from_observations(
        observations=observations,
        obs_radius=obs_radius,
        cost_to_go_matrices=cost_to_go_matrices,
    )
    node_features = torch.from_numpy(node_features).float()

    # communication graph
    global_xys = np.array([obs["global_xy"] for obs in observations])
    Adj = squareform(pdist(global_xys, "euclidean"))
    mask = Adj <= obs_radius
    Adj = Adj * mask
    Adj += np.eye(Adj.shape[0])  # self-loop

    # edge features
    edge_index, _ = dense_to_sparse(torch.tensor(Adj))

    def ef(i, j):
        pos_diff = global_xys[i] - global_xys[j]
        return (*pos_diff, np.abs(pos_diff).sum())

    edge_attr = torch.from_numpy(np.array([ef(i, j) for i, j in edge_index.T])).float()

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    if actions is not None:
        data.y = torch.from_numpy(np.array(actions))
    return data


def get_pyg_datalist_from_one_instance(
    dataset_ins,
    obs_radius: int,
) -> list[Data]:
    observations_ins, actions_ins, _ = dataset_ins
    cost_to_go_matrices = get_cost_to_go_matrices(observations_ins[0])
    params = dict(
        obs_radius=obs_radius,
        cost_to_go_matrices=cost_to_go_matrices,
    )
    pyg_data = [
        get_pyg_data_from_observations(observations=o, actions=a, **params)
        for o, a in zip(observations_ins, actions_ins)
    ]
    return pyg_data
