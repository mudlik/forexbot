"""
Train baseline MLP on engineered features.

Run from project root:
  python -m src.main_train
  python -m src.main_train --config-dir configs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Forex SMC baseline model")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing data.yaml, smc.yaml, train.yaml",
    )
    args = parser.parse_args()
    run_dir = run_training(config_dir=args.config_dir)
    print(f"Training finished. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
