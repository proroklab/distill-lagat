# AGENTS.md

Quick guide for working in this repo.

## Project summary
- distill-lagat: simplified LaGAT (GNN policy + LaCAM search) for MAPF.
- Python 3.10 + PyTorch (GNN), C++17 (search).

## Key paths
- `src/`: Python source code.
- `scripts/`: CLI entry points for data, training, evaluation.
- `cpp_planners/`: C++ planners (LaCAM/LaGAT).
- `assets/`: maps, pretrained models, demo artifacts.
- `outputs/`: generated datasets and run outputs.
- `tests/`: pytest-based checks for features/model.

## Typical workflow
```sh
# 1) Collect expert trajectories
uv run scripts/collect_expert_trajectories.py num_samples=10 save_animation=True

# 2) Convert to imitation dataset
uv run scripts/convert_to_imitation_dataset.py dataset_dir=/path/to/dataset/

# 3) Train
uv run scripts/train.py dataset_dir=/path/to/imitation_learning_dataset/ num_epochs=10

# 4) Evaluate pretrained model
uv run scripts/eval_model.py model.fpath=assets/pretrained/success_best.jit save_animation=True
```
