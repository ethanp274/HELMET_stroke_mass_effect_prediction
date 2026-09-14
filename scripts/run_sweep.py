"""Launch a Weights & Biases hyperparameter sweep for the MGB training runs.

Originally compiled by Ethan Phillips (Univ of Oxford), 12 August 2024.
"""

import argparse
import os

import wandb
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(REPO_ROOT, "configs", "sweeps")
DEFAULT_COUNT = 50


def resolve_config(parser, config):
    """Accept a bare sweep name, a name with .yaml, or an explicit path."""
    if os.path.exists(config):
        return config
    candidate = os.path.join(SWEEP_DIR, config)
    if not candidate.endswith(".yaml"):
        candidate += ".yaml"
    if os.path.exists(candidate):
        return candidate
    available = sorted(f for f in os.listdir(SWEEP_DIR) if f.endswith(".yaml"))
    parser.error(f"no sweep config {config!r}; available: {', '.join(available)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="sweep name in configs/sweeps, or a path to a YAML file"
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help="runs to execute"
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", "edema_pred_ml"),
        help="W&B entity (defaults to $WANDB_ENTITY)",
    )
    args = parser.parse_args()

    with open(resolve_config(parser, args.config), "r") as yaml_file:
        sweep_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, entity=args.entity)
    wandb.agent(sweep_id, count=args.count)


if __name__ == "__main__":
    main()
