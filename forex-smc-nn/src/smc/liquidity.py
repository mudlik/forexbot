"""
Liquidity pools, equal highs/lows, sweeps, open vs closed liquidity.

TODO: Rank pools (fractals, session extremes, PDH/PDL, ...), validate sweep rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiquidityPool:
    price: float
    label: str = "stub"


def list_pools_stub() -> list[LiquidityPool]:
    return []
