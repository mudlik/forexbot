from __future__ import annotations

import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.ensure import ensure_yahoo_data
from src.data.loaders import load_ohlcv_csv
from src.ml.datasets import WindowDataset, build_windows
from src.ml.features_spec import FEATURE_NAMES
from src.ml.labels import compute_direction_labels, compute_long_trade_win_labels
from src.ml.model import MLPClassifier
from src.ml.preprocessing import fit_scaler, save_scaler
from src.smc.features import compute_feature_frame
from src.trading.analysis import analysis_config_from_yaml, analyze_chart
from src.utils.paths import project_root


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    n = len(y)
    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())
    if c0 == 0 or c1 == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    w0 = n / (2 * max(c0, 1))
    w1 = n / (2 * max(c1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def _tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """
    Pick P(win) threshold that maximizes **precision** on validation (win rate when model says "take").

    Among ties, prefer a **higher** threshold (stricter filter, fewer trades, usually higher live WR).
    If nothing qualifies, default to 0.55 for downstream backtests.
    """
    n = len(y_true)
    candidates: list[tuple[float, float, int]] = []
    for t in np.linspace(0.2, 0.95, 36):
        mask = probs >= t
        k = int(mask.sum())
        if k < 5:
            continue
        prec = float(y_true[mask].mean())
        candidates.append((prec, float(t), k))

    if not candidates:
        return 0.55, float(np.mean(y_true)) if n else 0.0

    candidates.sort(key=lambda x: (-x[0], -x[1]))
    prec, t, _ = candidates[0]
    return t, prec


def run_training(
    *,
    config_dir: Path | None = None,
) -> Path:
    root = project_root()
    cfg_dir = config_dir or (root / "configs")
    data_cfg = _load_yaml(cfg_dir / "data.yaml")
    smc_cfg = _load_yaml(cfg_dir / "smc.yaml")
    train_cfg = _load_yaml(cfg_dir / "train.yaml")
    trading_cfg = _load_yaml(cfg_dir / "trading.yaml")

    _set_seed(int(train_cfg["random_seed"]))

    ensure_yahoo_data(root, data_cfg)
    raw_path = root / data_cfg["raw_csv"]
    tz = data_cfg.get("timezone")
    df = load_ohlcv_csv(raw_path, timezone=tz)

    label_mode = str(train_cfg.get("label_mode", "direction")).lower()
    acfg = analysis_config_from_yaml(smc_cfg, trading_cfg)
    analysis = analyze_chart(df, ma_period=acfg["ma_period"], atr_period=acfg["atr_period"])

    feat_df = compute_feature_frame(df, ma_period=int(smc_cfg["ma_period"]))
    if label_mode == "trade_win":
        labels = compute_long_trade_win_labels(df, analysis, trading_cfg)
    else:
        labels = compute_direction_labels(df["close"], horizon=int(smc_cfg["label_horizon"]))

    feat_np = feat_df[list(FEATURE_NAMES)].values.astype(np.float32)
    X_flat, y, _ = build_windows(feat_np, labels, lookback=int(train_cfg["lookback"]), return_indices=True)

    seed = int(train_cfg["random_seed"])
    val_frac = float(train_cfg["val_fraction"])
    strat = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat,
            y,
            test_size=val_frac,
            random_state=seed,
            stratify=strat,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat,
            y,
            test_size=val_frac,
            random_state=seed,
        )

    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = WindowDataset(X_train_s, y_train)
    val_ds = WindowDataset(X_val_s, y_val)

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    input_dim = X_train_s.shape[1]
    hidden_dims = list(train_cfg["hidden_dims"])
    model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_weights = bool(train_cfg.get("use_class_weights", True))
    w = _class_weights(y_train).to(device) if use_weights else None
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))

    best_val = float("inf")
    best_state = None
    epochs = int(train_cfg["epochs"])

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
        avg_val = total_loss / max(1, len(val_ds))
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()
    val_probs: list[np.ndarray] = []
    val_ys: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            val_probs.append(p)
            val_ys.append(yb.numpy())
    probs_v = np.concatenate(val_probs)
    y_v = np.concatenate(val_ys)
    pred_v = (probs_v >= 0.5).astype(np.int64)
    val_acc = float(np.mean(pred_v == y_v))
    best_th, val_prec_at_th = _tune_threshold(probs_v, y_v)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / train_cfg.get("checkpoint_dir", "models") / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": best_state,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "num_classes": 2,
            "feature_names": list(FEATURE_NAMES),
            "lookback": int(train_cfg["lookback"]),
            "best_val_loss": best_val,
            "label_mode": label_mode,
            "val_accuracy": val_acc,
            "min_prob_threshold": best_th,
            "val_precision_at_threshold": val_prec_at_th,
            "base_win_rate": float(y_train.mean()),
        },
        ckpt_path,
    )
    save_scaler(scaler, run_dir / "scaler.joblib")

    for name in ("data.yaml", "smc.yaml", "train.yaml", "trading.yaml"):
        shutil.copy2(cfg_dir / name, run_dir / name)

    print(
        f"[train] label_mode={label_mode} val_acc@0.5={val_acc:.4f} "
        f"min_prob_threshold={best_th:.3f} val_precision@{best_th:.2f}={val_prec_at_th:.4f} "
        f"base_win_rate_train={float(y_train.mean()):.4f}"
    )

    return run_dir
