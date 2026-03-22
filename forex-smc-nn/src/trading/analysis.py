"""
Chart analysis: OHLCV + baseline indicators used for signals and risk distances.

Extend with SMC structure/liquidity/FVG modules as they are implemented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.smc.features import compute_feature_frame


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = _true_range(df["high"], df["low"], prev_close)
    return tr.rolling(period, min_periods=period).mean()


def analyze_chart(df: pd.DataFrame, *, ma_period: int = 20, atr_period: int = 14) -> pd.DataFrame:
    """
    Return an analysis frame row-aligned with `df` (same index).

    Columns include baseline features, ATR, and simple trend score for signal hooks.
    """
    feats = compute_feature_frame(df, ma_period=ma_period)
    atr = compute_atr(df, period=atr_period)
    close = df["close"]
    ma = close.rolling(ma_period, min_periods=1).mean()
    trend_score = np.sign(close - ma)

    out = feats.copy()
    out["atr"] = atr
    out["trend_score"] = trend_score
    return out


def analysis_config_from_yaml(smc_cfg: dict[str, Any], trading_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "ma_period": int(smc_cfg.get("ma_period", 20)),
        "atr_period": int(trading_cfg.get("atr_period", 14)),
    }
