from __future__ import annotations

import pandas as pd


def resample_ohlc(df: pd.DataFrame, rule: str, *, label: str = "left") -> pd.DataFrame:
    """
    Resample OHLCV to a new timeframe (e.g. '1H', '4H', '1D').

    Assumes `timestamp` is datetime and sorted. Sets timestamp as index for resampling.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")

    work = df.set_index("timestamp")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = work.resample(rule, label=label).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    return out


def resample_multi_timeframe(
    df: pd.DataFrame,
    rules: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Build multiple timeframe frames for TDA-style workflows.

    TODO: align bars across timeframes (forward-fill HTF context) when SMC modules need it.
    """
    return {rule: resample_ohlc(df, rule) for rule in rules}
