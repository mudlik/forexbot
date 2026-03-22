"""
Tune thresholds so backtest win rate meets a target (e.g. 50%+).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.trading.signals import compute_signals
from src.trading.simulator import BacktestResult, run_backtest


def _win_rate(result: BacktestResult) -> tuple[float, int, int]:
    wins = sum(1 for t in result.trades if t.exit_reason == "tp")
    n = len(result.trades)
    return (wins / n if n else 0.0, wins, n)


def find_min_prob_for_target_wr(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    base_signals: pd.Series,
    p_win: np.ndarray,
    trading_cfg: dict[str, Any],
    reward_cfg: Any,
    *,
    target_wr: float = 0.5,
    min_trades: int = 8,
) -> tuple[float | None, BacktestResult | None, list[tuple[float, float, int]]]:
    """
    Grid-search min P(win) for longs. Prefer the **most permissive** threshold (lowest bar)
    among those with WR >= target and trades >= min_trades; if none, pick threshold with max WR then max trades.

    Returns (chosen_threshold, result, scan_log as list of (th, wr, n)).
    """
    scan: list[tuple[float, float, int]] = []
    feasible: list[tuple[float, float, int, BacktestResult]] = []

    for th in np.linspace(0.30, 0.95, 66):
        sig = base_signals.to_numpy(dtype=np.int8).copy()
        for i in range(len(sig)):
            if sig[i] == 1:
                if not np.isfinite(p_win[i]) or float(p_win[i]) < float(th):
                    sig[i] = 0
        signals = pd.Series(sig, index=base_signals.index, name="signal")
        result = run_backtest(df, analysis, signals, trading_cfg, reward_cfg)
        wr, wins, n = _win_rate(result)
        scan.append((float(th), wr, n))
        if n >= min_trades and wr + 1e-9 >= target_wr:
            feasible.append((float(th), wr, n, result))

    if feasible:
        # Maximize number of trades; tie-break: lower threshold (more permissive among same n is rare)
        feasible.sort(key=lambda x: (-x[2], x[0]))
        th, wr, n, res = feasible[0]
        return th, res, scan

    # Fallback: best WR, then most trades
    best: tuple[float, float, int, BacktestResult] | None = None
    for th in np.linspace(0.30, 0.95, 66):
        sig = base_signals.to_numpy(dtype=np.int8).copy()
        for i in range(len(sig)):
            if sig[i] == 1:
                if not np.isfinite(p_win[i]) or float(p_win[i]) < float(th):
                    sig[i] = 0
        signals = pd.Series(sig, index=base_signals.index, name="signal")
        result = run_backtest(df, analysis, signals, trading_cfg, reward_cfg)
        wr, wins, n = _win_rate(result)
        if n == 0:
            continue
        if best is None or wr > best[1] + 1e-9 or (abs(wr - best[1]) < 1e-9 and n > best[2]):
            best = (float(th), wr, n, result)

    if best is None:
        return None, None, scan
    return best[0], best[3], scan


def find_long_threshold_for_target_wr(
    analysis: pd.DataFrame,
    df: pd.DataFrame,
    trading_cfg: dict[str, Any],
    reward_cfg: Any,
    *,
    target_wr: float = 0.5,
    min_trades: int = 8,
    long_grid: np.ndarray | None = None,
) -> tuple[float | None, BacktestResult | None, pd.Series | None]:
    """
    Tighten long entry by lowering signal_dist_to_ma_long (more negative = fewer longs, often higher WR).
    """
    if long_grid is None:
        long_grid = np.linspace(0.35, -1.8, 45)

    best_feasible: tuple[float, BacktestResult] | None = None
    best_n = -1

    base = dict(trading_cfg)
    for long_th in long_grid:
        cfg = dict(base)
        cfg["signal_dist_to_ma_long"] = float(long_th)
        signals = compute_signals(analysis, cfg)
        result = run_backtest(df, analysis, signals, trading_cfg, reward_cfg)
        wr, wins, n = _win_rate(result)
        if n >= min_trades and wr + 1e-9 >= target_wr:
            if n > best_n:
                best_n = n
                best_feasible = (float(long_th), result)

    if best_feasible is not None:
        th, res = best_feasible
        cfg = dict(base)
        cfg["signal_dist_to_ma_long"] = float(th)
        sig = compute_signals(analysis, cfg)
        return float(th), res, sig

    # Fallback: maximize WR
    best_wr = -1.0
    best_n = 0
    best_t = None
    best_res = None
    best_sig = None
    for long_th in long_grid:
        cfg = dict(base)
        cfg["signal_dist_to_ma_long"] = float(long_th)
        signals = compute_signals(analysis, cfg)
        result = run_backtest(df, analysis, signals, trading_cfg, reward_cfg)
        wr, wins, n = _win_rate(result)
        if n == 0:
            continue
        if wr > best_wr + 1e-9 or (abs(wr - best_wr) < 1e-9 and n > best_n):
            best_wr = wr
            best_n = n
            best_t = float(long_th)
            best_res = result
            best_sig = signals

    return best_t, best_res, best_sig
