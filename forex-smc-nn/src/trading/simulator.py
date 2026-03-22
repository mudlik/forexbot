"""
Walk-forward simulation: entries from signals, ATR-based SL/TP, rewards per trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.trading.rewards import RewardConfig, compute_trade_reward


@dataclass(frozen=True)
class TradeRecord:
    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    exit_reason: str
    realized_r: float
    reward: float


@dataclass(frozen=True)
class BacktestResult:
    trades: list[TradeRecord]
    total_reward: float
    equity_curve: pd.Series


def _exit_long_bar(
    low: float,
    high: float,
    sl: float,
    tp: float,
) -> tuple[str, float] | None:
    """Return (reason, exit_price) if SL/TP hit this bar; None if no exit."""
    hit_sl = low <= sl
    hit_tp = high >= tp
    if hit_sl and hit_tp:
        return "sl", sl
    if hit_sl:
        return "sl", sl
    if hit_tp:
        return "tp", tp
    return None


def simulate_long_trade_outcome(
    ohlc: pd.DataFrame,
    analysis: pd.DataFrame,
    entry_idx: int,
    trading_cfg: dict[str, Any],
) -> tuple[str, float] | None:
    """
    Hypothetical long opened at open[entry_idx] using ATR from the same bar.

    Returns (exit_reason, exit_price) or None if ATR invalid / out of range.
    """
    n = len(ohlc)
    if entry_idx < 0 or entry_idx >= n:
        return None

    atr_mult = float(trading_cfg.get("atr_stop_mult", 1.5))
    rr = float(trading_cfg.get("risk_reward", 2.0))
    max_hold = int(trading_cfg.get("max_hold_bars", 80))

    open_px = ohlc["open"].to_numpy(dtype=float)
    high = ohlc["high"].to_numpy(dtype=float)
    low = ohlc["low"].to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    atr = analysis["atr"].to_numpy(dtype=float)

    entry_price = float(open_px[entry_idx])
    a = float(atr[entry_idx])
    if not np.isfinite(a) or a <= 0:
        return None

    risk_dist = a * atr_mult
    sl = entry_price - risk_dist
    tp = entry_price + rr * risk_dist

    exit_reason = "timeout"
    exit_price = float(close[min(entry_idx + max_hold, n - 1)])

    for j in range(entry_idx, min(entry_idx + max_hold + 1, n)):
        hit = _exit_long_bar(low[j], high[j], sl, tp)
        if hit is not None:
            exit_reason, exit_price = hit
            break
    else:
        exit_idx = min(entry_idx + max_hold, n - 1)
        exit_price = float(close[exit_idx])
        exit_reason = "timeout"

    return exit_reason, exit_price


def _exit_short_bar(
    low: float,
    high: float,
    sl: float,
    tp: float,
) -> tuple[str, float] | None:
    hit_sl = high >= sl
    hit_tp = low <= tp
    if hit_sl and hit_tp:
        return "sl", sl
    if hit_sl:
        return "sl", sl
    if hit_tp:
        return "tp", tp
    return None


def run_backtest(
    ohlc: pd.DataFrame,
    analysis: pd.DataFrame,
    signals: pd.Series,
    trading_cfg: dict[str, Any],
    reward_cfg: RewardConfig,
) -> BacktestResult:
    """
    Execute at most one open position at a time.

    Entry: next bar open after signal (signal on bar i -> enter at open[i+1]).
    Stop: ATR * atr_stop_mult from entry; target: RR * risk distance.
    """
    atr_period = int(trading_cfg.get("atr_period", 14))
    atr_mult = float(trading_cfg.get("atr_stop_mult", 1.5))
    rr = float(trading_cfg.get("risk_reward", 2.0))
    max_hold = int(trading_cfg.get("max_hold_bars", 80))

    open_px = ohlc["open"].to_numpy(dtype=float)
    high = ohlc["high"].to_numpy(dtype=float)
    low = ohlc["low"].to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    atr = analysis["atr"].to_numpy(dtype=float)
    sig = signals.to_numpy(dtype=np.int8)

    n = len(ohlc)
    trades: list[TradeRecord] = []
    bar_rewards = np.zeros(n, dtype=float)
    total_reward = 0.0

    i = 0
    while i < n - 2:
        s = int(sig[i])
        if s == 0:
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = float(open_px[entry_idx])
        a = float(atr[entry_idx])
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue

        risk_dist = a * atr_mult
        if s == 1:
            side = "long"
            sl = entry_price - risk_dist
            tp = entry_price + rr * risk_dist
        else:
            side = "short"
            sl = entry_price + risk_dist
            tp = entry_price - rr * risk_dist

        exit_idx = entry_idx
        exit_reason = "timeout"
        exit_price = float(close[min(entry_idx + max_hold, n - 1)])

        for j in range(entry_idx, min(entry_idx + max_hold + 1, n)):
            if side == "long":
                hit = _exit_long_bar(low[j], high[j], sl, tp)
            else:
                hit = _exit_short_bar(low[j], high[j], sl, tp)
            if hit is not None:
                exit_reason, exit_price = hit
                exit_idx = j
                break
        else:
            exit_idx = min(entry_idx + max_hold, n - 1)
            exit_price = float(close[exit_idx])
            exit_reason = "timeout"

        if side == "long":
            realized_r = (exit_price - entry_price) / max(risk_dist, 1e-12)
        else:
            realized_r = (entry_price - exit_price) / max(risk_dist, 1e-12)

        won = exit_reason == "tp"
        lost = exit_reason == "sl"
        timeout = exit_reason == "timeout"

        rw = compute_trade_reward(
            won=won,
            lost=lost,
            timeout=timeout,
            realized_r=realized_r if (won or lost) else None,
            rc=reward_cfg,
        )

        trades.append(
            TradeRecord(
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=sl,
                target_price=tp,
                exit_reason=exit_reason,
                realized_r=float(realized_r),
                reward=float(rw),
            )
        )
        bar_rewards[exit_idx] += rw
        total_reward += rw

        i = exit_idx + 1

    ts = ohlc["timestamp"] if "timestamp" in ohlc.columns else pd.RangeIndex(n)
    equity = pd.Series(np.cumsum(bar_rewards), index=ts, name="cum_reward")
    return BacktestResult(trades=trades, total_reward=float(total_reward), equity_curve=equity)
