"""
Session and kill-zone helpers (Kyiv timezone in configs).

TODO: Map bars to Asia/London/NY, kill zones, session-boundary filters.
"""

from __future__ import annotations

import pandas as pd


def session_id_stub(ts: pd.Timestamp) -> str:
    """Placeholder: return a coarse session label."""
    return "unknown"
