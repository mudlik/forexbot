"""
Fair Value Gap detection and lifecycle (open / mitigated / invalid).

TODO: Implement 3-candle FVG, midpoint 0.5, IOFED / fulfil rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FVGZone:
    low: float
    high: float
    direction: str


def detect_fvg_stub() -> list[FVGZone]:
    return []
