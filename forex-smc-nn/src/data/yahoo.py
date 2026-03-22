"""
Download OHLCV from Yahoo Finance into project CSV format (timestamp, open, high, low, close, volume).

Ticker for EUR/USD spot on Yahoo: EURUSD=X
https://finance.yahoo.com/quote/EURUSD%3DX/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("Install yfinance: pip install yfinance") from e


def download_yahoo_ohlcv(
    ticker: str,
    *,
    period: str = "2y",
    interval: str = "1h",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance.

    Parameters match yfinance: period e.g. 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max;
    interval e.g. 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo.
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker!r} (period={period}, interval={interval})")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    df = df.reset_index()
    first = df.columns[0]
    df = df.rename(columns={first: "timestamp"})

    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in Yahoo response")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
        }
    )
    if "volume" in df.columns:
        out["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    return out


def save_yahoo_csv(
    output: str | Path,
    *,
    ticker: str = "EURUSD=X",
    period: str = "2y",
    interval: str = "1h",
) -> Path:
    """Download and write CSV; returns path written."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = download_yahoo_ohlcv(ticker, period=period, interval=interval)
    df.to_csv(output, index=False)
    return output.resolve()
