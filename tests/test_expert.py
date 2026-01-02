from pathlib import Path
import platform

from pogema import GridConfig
from lagat.interfaces.lacam3.inference import Lacam3Inference, Lacam3InferenceConfig
from lagat.interfaces.lagat.inference import LagatInference, LagatInferenceConfig
from lagat.run_planners import run_full_horizon_planner as run_expert


def _make_grid_config():
    obs_radius = 5
    grid_config = GridConfig(
        num_agents=5,
        obs_radius=obs_radius,
        seed=0,
        observation_type="MAPF",
        on_target="nothing",
        density=0.1,
        collision_system="soft",
        max_episode_steps=128,
    )
    return grid_config


def _run_and_save(planner, grid_config, output_name):
    env, all_actions, all_observations, all_terminated = run_expert(
        planner,
        grid_config=grid_config,
        animation=True,
    )
    success = all(all_terminated[-1])
    target_dir = Path(__file__).parent / ".local"
    target_dir.mkdir(parents=True, exist_ok=True)
    env.save_animation(target_dir / output_name)
    return success


def test_lacam3():
    grid_config = _make_grid_config()
    cfg = Lacam3InferenceConfig(timeouts=[5.0])
    planner = Lacam3Inference(cfg)
    success = _run_and_save(planner, grid_config, "solution_example_lacam3.svg")

    assert success, "Failed to solve"


def test_lagat():
    grid_config = _make_grid_config()
    if platform.system() == "Darwin":
        model_path = None
    else:
        model_path = Path(__file__).parents[1] / "assets/pretrained/loss_best_jit.pt"
    cfg = LagatInferenceConfig(time_limit=5.0, model_path=model_path)
    planner = LagatInference(cfg)
    success = _run_and_save(planner, grid_config, "solution_example_lagat.svg")

    assert success, "Failed to solve"
