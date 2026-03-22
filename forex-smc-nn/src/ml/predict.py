from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.ml.datasets import build_window_features_only
from src.ml.model import MLPClassifier
from src.ml.preprocessing import load_scaler


def load_model_from_run(run_dir: str | Path, map_location: str = "cpu") -> tuple[MLPClassifier, dict[str, Any]]:
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoint.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=map_location)
    model = MLPClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        num_classes=int(ckpt.get("num_classes", 2)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict_p_win_per_bar(
    feat_np: np.ndarray,
    run_dir: str | Path,
    lookback: int,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    P(win) for `trade_win` label definition, aligned to bar indices.

    Returns (p_win_full, bar_indices) where p_win_full has length n with np.nan for bars
    without a full window or invalid features.
    """
    run_dir = Path(run_dir)
    n = feat_np.shape[0]
    X, bar_idx = build_window_features_only(feat_np, lookback)
    scaler = load_scaler(run_dir / "scaler.joblib")
    Xs = scaler.transform(X)

    model, _ = load_model_from_run(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(Xs), batch_size):
            batch = torch.from_numpy(Xs[start : start + batch_size]).to(device)
            logits = model(batch)
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
    p_flat = np.concatenate(probs, axis=0)

    out = np.full(n, np.nan, dtype=np.float64)
    out[bar_idx] = p_flat
    return out, bar_idx
