"""
Feature engineering bridge: OHLCV -> per-bar feature columns for ML.

Baseline uses simple price-action stats; replace/extend with SMC rules in sibling modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ml.features_spec import FEATURE_NAMES


def compute_feature_frame(df: pd.DataFrame, *, ma_period: int = 20) -> pd.DataFrame:
    """
    Compute a row-aligned feature frame indexed like `df`.

    Columns match `FEATURE_NAMES` order (see `src/ml/features_spec.py`).
    """
    close = df["close"]
    ma = close.rolling(ma_period, min_periods=1).mean()
    std = close.rolling(ma_period, min_periods=1).std()
    dist_to_ma = (close - ma) / (std.replace(0, np.nan))
    dist_to_ma = dist_to_ma.fillna(0.0)

    out = pd.DataFrame(
        {
            "return_1": close.pct_change(),
            "log_return_1": np.log(close).diff(),
            "dist_to_ma": dist_to_ma,
        },
        index=df.index,
    )
    return out[list(FEATURE_NAMES)]
