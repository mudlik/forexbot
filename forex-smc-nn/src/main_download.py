"""
Download OHLCV from Yahoo Finance (e.g. EUR/USD EURUSD=X) into data/raw for training/backtest.

Usage (from project root):
  python -m src.main_download
  python -m src.main_download --ticker EURUSD=X --period 2y --interval 1h -o data/raw/eurusd_yahoo.csv
  python -m src.main_download --period max --interval 1d

Reference: https://finance.yahoo.com/quote/EURUSD%3DX/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.yahoo import save_yahoo_csv
from src.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV to CSV")
    parser.add_argument("--ticker", default="EURUSD=X", help="Yahoo symbol, e.g. EURUSD=X")
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance period. Note: 1h bars are only available for ~last 730 days on Yahoo.",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Bar size: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/raw/eurusd_yahoo.csv under project root)",
    )
    args = parser.parse_args()

    root = project_root()
    out = args.output if args.output is not None else (root / "data" / "raw" / "eurusd_yahoo.csv")
    if not out.is_absolute():
        out = root / out

    path = save_yahoo_csv(out, ticker=args.ticker, period=args.period, interval=args.interval)
    print(f"Saved {path} ({args.ticker} period={args.period} interval={args.interval})")


if __name__ == "__main__":
    main()
