"""
Targets for supervised learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.trading.simulator import simulate_long_trade_outcome


def compute_direction_labels(close: pd.Series, horizon: int) -> np.ndarray:
    """
    Per-bar label: 1 if close[t+horizon] > close[t], else 0. NaN where future is unavailable.
    """
    future_ret = close.shift(-horizon) / close - 1
    y = np.where(future_ret.isna(), np.nan, (future_ret > 0).astype(np.float64))
    return y


def compute_long_trade_win_labels(
    ohlc: pd.DataFrame,
    analysis: pd.DataFrame,
    trading_cfg: dict[str, Any],
) -> np.ndarray:
    """
    Label at bar `i`: outcome of a long entered at `open[i+1]` (same as backtest: signal on `i` -> entry next bar).

    1 = TP before SL; 0 = SL or timeout. NaN = invalid ATR or no room to simulate.
    """
    n = len(ohlc)
    y = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - 1):
        out = simulate_long_trade_outcome(ohlc, analysis, i + 1, trading_cfg)
        if out is None:
            continue
        reason, _ = out
        y[i] = 1.0 if reason == "tp" else 0.0
    return y
