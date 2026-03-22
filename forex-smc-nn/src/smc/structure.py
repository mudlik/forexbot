"""
Market structure (HH/HL/LL/LH, BOS, MSS, MSR).

TODO: Implement per methodology — confirmations on candle *body* close, not wick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Trend = Literal["bullish", "bearish", "range", "unknown"]


@dataclass
class StructureSnapshot:
    trend: Trend = "unknown"


def detect_trend_stub() -> StructureSnapshot:
    """Placeholder until swing/BOS/MSS logic is implemented."""
    return StructureSnapshot(trend="unknown")
