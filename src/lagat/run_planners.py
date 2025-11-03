import torch
import torch.nn.functional as F
from pogema import pogema_v0, AnimationMonitor
from typing import Any

from .features import (
    get_cost_to_go_matrices,
    get_pyg_data_from_observations,
)
from .utils import suppress_stdout


@suppress_stdout()
def run_full_horizon_planner(planner, grid_config, animation=False):
    env = pogema_v0(grid_config=grid_config)
    if animation:
        env = AnimationMonitor(env)
    obs, infos = env.reset()
    planner.reset_states()

    all_actions = []
    all_observations = []
    all_terminated = []

    while True:
        actions = planner.act(obs)
        all_observations.append(obs)
        all_actions.append(actions)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        all_terminated.append(terminated)

        if all(terminated) or all(truncated):
            break

    return env, all_actions, all_observations, all_terminated


@torch.no_grad()
def run_neural_policy(
    model,
    grid_config,
    animation: bool = False,
    max_episodes: int = 128,
    device: str = "cpu",
    sampling_temperature: float = 1.0,
    **_,
):
    model.eval()
    env = pogema_v0(grid_config=grid_config)
    if animation:
        env = AnimationMonitor(env)

    obs, infos = env.reset()
    cost_to_go_matrices = get_cost_to_go_matrices(obs)
    params = dict(
        obs_radius=model.obs_radius,
        cost_to_go_matrices=cost_to_go_matrices,
    )

    all_actions = []
    all_observations = []
    all_terminated = []

    for _ in range(max_episodes):
        # sample action
        data = get_pyg_data_from_observations(obs, **params).to(device)
        probs = F.softmax(model(data) / sampling_temperature, dim=-1).detach().cpu()
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1).numpy()

        all_observations.append(obs)
        all_actions.append(actions)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        all_terminated.append(terminated)

        if all(terminated) or all(truncated):
            break

    return env, all_actions, all_observations, all_terminated


def get_sum_of_costs(all_actions, all_observations, all_terminated) -> int:
    if not all(all_terminated[-1]):
        return 0
    cost = 0
    N = len(all_observations[0])
    T = len(all_observations)
    for i in range(N):
        c = T
        for t in reversed(range(T)):
            obs = all_observations[t][i]
            if obs["global_xy"] != obs["global_target_xy"]:
                break
            c = t
        cost += c
    return cost


def get_makespan(all_actions, all_observations, all_terminated) -> int:
    if not all(all_terminated[-1]):
        return 0
    return len(all_actions)


def get_solution_quality(
    all_actions, all_observations, all_terminated, **_
) -> dict[str, Any]:
    res = (all_actions, all_observations, all_terminated)
    D = get_cost_to_go_matrices(all_observations[0])
    dist_start_goal = [D[i][o["global_xy"]] for i, o in enumerate(all_observations[0])]
    sum_of_costs_lb = sum(dist_start_goal)
    makespan_lb = max(dist_start_goal)

    sum_of_costs = get_sum_of_costs(*res)
    makespan = get_makespan(*res)

    return dict(
        solved=all(all_terminated[-1]),
        sum_of_costs=get_sum_of_costs(*res),
        sum_of_costs_lb=sum_of_costs_lb,
        sum_of_costs_subopt=sum_of_costs / sum_of_costs_lb,
        makespan=makespan,
        makespan_lb=makespan_lb,
        makespan_subopt=makespan / makespan_lb,
    )
