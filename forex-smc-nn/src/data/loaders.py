from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")


def load_ohlcv_csv(
    path: str | Path,
    *,
    timezone: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV from CSV.

    Expected columns: timestamp, open, high, low, close; volume optional.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp values in CSV")

    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise ValueError("Non-numeric OHLC values")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    df = df.sort_values("timestamp").reset_index(drop=True)

    if timezone:
        ts = df["timestamp"]
        if ts.dt.tz is None:
            df["timestamp"] = ts.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
        else:
            df["timestamp"] = ts.dt.tz_convert(timezone)

    return df
