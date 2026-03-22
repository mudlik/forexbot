from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.ensure import ensure_yahoo_data, merge_data_cfg_defaults
from src.data.loaders import load_ohlcv_csv
from src.ml.datasets import WindowDataset, build_windows
from src.ml.features_spec import FEATURE_NAMES
from src.ml.labels import compute_direction_labels, compute_long_trade_win_labels
from src.trading.analysis import analysis_config_from_yaml, analyze_chart
from src.ml.model import MLPClassifier
from src.ml.preprocessing import load_scaler
from src.smc.features import compute_feature_frame
from src.utils.paths import project_root


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(
    run_dir: str | Path,
    *,
    limit_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load checkpoint + scaler from `run_dir`, score all valid windows from the configured CSV.
    Returns (probs_class1, y_true_or_minus1).
    """
    run_dir = Path(run_dir)
    root = project_root()
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    data_cfg = merge_data_cfg_defaults(_load_yaml(run_dir / "data.yaml"), root)
    smc_cfg = _load_yaml(run_dir / "smc.yaml")
    train_cfg = _load_yaml(run_dir / "train.yaml")
    tr_path = run_dir / "trading.yaml"
    trading_cfg = _load_yaml(tr_path) if tr_path.is_file() else _load_yaml(root / "configs" / "trading.yaml")

    ensure_yahoo_data(root, data_cfg)
    raw_path = root / data_cfg["raw_csv"]
    tz = data_cfg.get("timezone")
    df = load_ohlcv_csv(raw_path, timezone=tz)

    feat_df = compute_feature_frame(df, ma_period=int(smc_cfg["ma_period"]))
    label_mode = str(train_cfg.get("label_mode", "direction")).lower()
    if label_mode == "trade_win":
        acfg = analysis_config_from_yaml(smc_cfg, trading_cfg)
        analysis = analyze_chart(df, ma_period=acfg["ma_period"], atr_period=acfg["atr_period"])
        labels = compute_long_trade_win_labels(df, analysis, trading_cfg)
    else:
        labels = compute_direction_labels(df["close"], horizon=int(smc_cfg["label_horizon"]))
    feat_np = feat_df[list(FEATURE_NAMES)].values.astype(np.float32)
    lookback = int(train_cfg["lookback"])
    X_flat, y = build_windows(feat_np, labels, lookback=lookback)

    scaler = load_scaler(run_dir / "scaler.joblib")
    X_s = scaler.transform(X_flat)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    hidden_dims = list(ckpt["hidden_dims"])
    num_classes = int(ckpt.get("num_classes", 2))

    model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = WindowDataset(X_s, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=int(train_cfg.get("batch_size", 32)), shuffle=False)

    probs_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []
    batches = 0
    for xb, yb in loader:
        logits = model(xb)
        prob = torch.softmax(logits, dim=-1)[:, 1].numpy()
        probs_list.append(prob)
        ys_list.append(yb.numpy())
        batches += 1
        if limit_batches is not None and batches >= limit_batches:
            break

    probs = np.concatenate(probs_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    return probs, ys
