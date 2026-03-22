"""
Run inference using a training run directory (checkpoint + scaler + copied configs).

Run from project root:
  python -m src.main_infer --run-dir models/run_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.ml.infer import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with saved Forex SMC model")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to models/run_* containing checkpoint.pt and scaler.joblib",
    )
    args = parser.parse_args()
    probs, ys = run_inference(args.run_dir)
    pred = (probs >= 0.5).astype(np.int64)
    acc = float((pred == ys).mean()) if len(ys) else 0.0
    print(f"samples={len(ys)} accuracy={acc:.4f} mean_p_up={probs.mean():.4f}")


if __name__ == "__main__":
    main()
