"""
Ensure raw OHLCV exists: optional automatic Yahoo Finance download when the file is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.yahoo import save_yahoo_csv


def merge_data_cfg_defaults(data_cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Fill Yahoo-related keys from project configs/data.yaml (e.g. old model run dirs)."""
    out = dict(data_cfg)
    base_path = root / "configs" / "data.yaml"
    if not base_path.is_file():
        return out
    with base_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if not isinstance(base, dict):
        return out
    for k in ("auto_yahoo_download", "yahoo_ticker", "yahoo_period", "yahoo_interval"):
        if k not in out and k in base:
            out[k] = base[k]
    return out


def ensure_yahoo_data(root: Path, data_cfg: dict[str, Any]) -> None:
    """
    If `auto_yahoo_download` is true and `yahoo_ticker` is set, download CSV when missing or empty.
    """
    cfg = merge_data_cfg_defaults(data_cfg, root)
    if not cfg.get("auto_yahoo_download"):
        return
    ticker = cfg.get("yahoo_ticker")
    if not ticker:
        return
    rel = cfg.get("raw_csv")
    if not rel:
        return
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    min_bytes = 100
    if path.is_file() and path.stat().st_size > min_bytes:
        return
    period = str(cfg.get("yahoo_period", "2y"))
    interval = str(cfg.get("yahoo_interval", "1h"))
    save_yahoo_csv(path, ticker=ticker, period=period, interval=interval)
    print(f"[data] Auto-downloaded Yahoo Finance {ticker} -> {path}")
