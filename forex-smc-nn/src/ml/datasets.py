from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int,
    *,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack sliding windows of `lookback` rows; label is taken at the end bar of each window.

    features: (n_samples, n_features)
    labels: (n_samples,) with NaN for invalid positions
    """
    n, f_dim = features.shape
    xs: list[np.ndarray] = []
    ys: list[int] = []
    idxs: list[int] = []
    for i in range(lookback, n):
        lab = labels[i]
        if np.isnan(lab):
            continue
        window = features[i - lookback : i]
        if not np.isfinite(window).all():
            continue
        xs.append(window.reshape(-1))
        ys.append(int(lab))
        idxs.append(i)
    if not xs:
        raise ValueError("No valid training windows; check data length, lookback, and NaNs.")
    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    if return_indices:
        return X, y, np.array(idxs, dtype=np.int64)
    return X, y


def build_window_features_only(features: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    All valid windows for inference (no label filter). Returns (X_flat, bar_indices).
    """
    n = features.shape[0]
    xs: list[np.ndarray] = []
    idxs: list[int] = []
    for i in range(lookback, n):
        window = features[i - lookback : i]
        if not np.isfinite(window).all():
            continue
        xs.append(window.reshape(-1))
        idxs.append(i)
    if not xs:
        raise ValueError("No valid windows for inference.")
    return np.stack(xs, axis=0).astype(np.float32), np.array(idxs, dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
