"""
Reward shaping for closed trades: positive for success, negative for failure.

Designed for backtesting and future RL training (same reward definition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    reward_win: float = 1.0
    penalty_loss: float = 1.0
    penalty_timeout: float = 0.25
    risk_reward: float = 2.0


def reward_config_from_yaml(cfg: dict[str, Any]) -> RewardConfig:
    return RewardConfig(
        reward_win=float(cfg.get("reward_win", 1.0)),
        penalty_loss=float(cfg.get("penalty_loss", 1.0)),
        penalty_timeout=float(cfg.get("penalty_timeout", 0.25)),
        risk_reward=float(cfg.get("risk_reward", 2.0)),
    )


def compute_trade_reward(
    *,
    won: bool,
    lost: bool,
    timeout: bool,
    realized_r: float | None,
    rc: RewardConfig,
) -> float:
    """
    Map trade outcome to scalar reward.

    - TP / favorable exit: +reward_win scaled by realized R vs planned RR (capped).
    - SL: -penalty_loss (optionally scale by |realized_r| if provided).
    - Timeout / flat exit: -penalty_timeout
    """
    if won:
        if realized_r is None:
            return rc.reward_win
        scale = min(max(realized_r, 0.0), rc.risk_reward * 1.25)
        return rc.reward_win * (scale / max(rc.risk_reward, 1e-9))
    if lost:
        mag = abs(realized_r) if realized_r is not None else 1.0
        return -rc.penalty_loss * min(mag, 2.0)
    if timeout:
        return -rc.penalty_timeout
    return 0.0
