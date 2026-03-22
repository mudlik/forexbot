"""
Single source of truth for feature column names and order (train/infer contract).
"""

from __future__ import annotations

FEATURE_NAMES: tuple[str, ...] = (
    "return_1",
    "log_return_1",
    "dist_to_ma",
)
