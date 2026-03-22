"""
Entry signals from chart analysis.

Baseline: dist_to_ma + trend alignment. Replace with SMC triggers (BOS, sweep, FVG) later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_signals(analysis: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """
    Per-bar discrete action: -1 short, 0 flat, +1 long.

    Rules (configurable thresholds on dist_to_ma):
    - Long: trend_score >= 0 and dist_to_ma <= long_threshold (oversold vs MA in uptrend/neutral)
    - Short: trend_score <= 0 and dist_to_ma >= short_threshold
    """
    long_th = float(cfg.get("signal_dist_to_ma_long", 0.35))
    short_th = float(cfg.get("signal_dist_to_ma_short", 0.35))

    ts = analysis["trend_score"].fillna(0.0)
    dist = analysis["dist_to_ma"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    long_mask = (ts >= 0) & (dist <= long_th)
    short_mask = (ts <= 0) & (dist >= short_th)

    sig = np.zeros(len(analysis), dtype=np.int8)
    sig[long_mask.to_numpy()] = 1
    sig[short_mask.to_numpy()] = -1
    # If both (rare), prefer flat
    sig[(long_mask & short_mask).to_numpy()] = 0
    return pd.Series(sig, index=analysis.index, name="signal")
