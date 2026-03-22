# Forex SMC + baseline neural network

Skeleton for Smart Money Concept feature engineering (`src/smc/`) and PyTorch training (`src/ml/`).

## Setup

```bash
cd forex-smc-nn
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Yahoo Finance EUR/USD** (`EURUSD=X`): with `auto_yahoo_download: true` in `configs/data.yaml`, the first run of train/backtest/infer **downloads** `data/raw/eurusd_yahoo.csv` if it is missing (requires network + `yfinance`). Manual override: `python -m src.main_download`.

`1h` history on Yahoo is limited to about the last **730 days**; use e.g. `python -m src.main_download --period max --interval 1d` and point `raw_csv` at that file for longer daily history. Timestamps are naive UTC; `timezone: "UTC"` in `configs/data.yaml`.

## Train

From the project root (directory that contains `src/` and `configs/`):

```bash
python -m src.main_train
python -m src.main_train --config-dir configs
```

`configs/train.yaml` uses `label_mode: trade_win`: the target is “TP before SL” for a **long** entered at `open[i+1]` (same risk rules as `configs/trading.yaml`). Class weights help with imbalance. The run saves `min_prob_threshold` in `checkpoint.pt` (chosen on validation to maximize **precision** of “take” signals).

Checkpoints, scaler, and copied YAMLs are written to `models/run_<timestamp>/`.

## Infer

```bash
python -m src.main_infer --run-dir models/run_YYYYMMDD_HHMMSS
```

## Chart analysis and backtest (rewards / penalties)

Walk-forward simulation: baseline signals from `dist_to_ma` + trend, ATR stops, fixed RR targets. Each closed trade gets a **reward** on take-profit and a **penalty** on stop-loss (configurable in `configs/trading.yaml`).

```bash
python -m src.main_backtest
python -m src.main_backtest --config-dir configs
```

Filter **long** entries with a trained model. By default (`configs/trading.yaml`) the backtest **auto-tunes** `min_prob` on the same CSV to maximize trade count subject to `win_rate >= target_win_rate` (default 50%). Override with `--min-prob` or disable with `--no-auto-tune`.

```bash
python -m src.main_backtest --model-run models/run_YYYYMMDD_HHMMSS
python -m src.main_backtest --model-run models/run_YYYYMMDD_HHMMSS --min-prob 0.55
python -m src.main_backtest --no-auto-tune
```

Replace signal logic in `src/trading/signals.py` with SMC rules (BOS, sweeps, FVG) when ready.

## Layout

- `data/raw/` — input CSV
- `data/processed/` — optional cached features
- `models/` — checkpoints and `scaler.joblib`
- `configs/` — `data.yaml`, `smc.yaml`, `train.yaml`, `trading.yaml`
- `src/smc/` — SMC domain logic (stubs + `features.py` baseline)
- `src/ml/` — datasets, MLP, train/infer
- `src/trading/` — chart analysis, signals, simulator, reward shaping
