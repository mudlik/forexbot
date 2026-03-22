"""
Run chart analysis, generate signals, simulate positions with reward/penalty accounting.

Optional: filter long entries with a trained `trade_win` model and min probability.
Auto-tune thresholds toward a target win rate (see configs/trading.yaml).

Usage (from project root):
  python -m src.main_backtest
  python -m src.main_backtest --model-run models/run_YYYYMMDD_HHMMSS
  python -m src.main_backtest --model-run models/run_xxx --min-prob 0.55
  python -m src.main_backtest --no-auto-tune
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.ensure import ensure_yahoo_data
from src.data.loaders import load_ohlcv_csv
from src.ml.features_spec import FEATURE_NAMES
from src.ml.predict import predict_p_win_per_bar
from src.smc.features import compute_feature_frame
from src.trading.analysis import analysis_config_from_yaml, analyze_chart
from src.trading.rewards import reward_config_from_yaml
from src.trading.signals import compute_signals
from src.trading.simulator import BacktestResult, run_backtest
from src.trading.wr_tune import find_long_threshold_for_target_wr, find_min_prob_for_target_wr
from src.utils.paths import project_root


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_long_prob_filter(
    signals: pd.Series,
    p_win: np.ndarray,
    th: float,
) -> pd.Series:
    sig = signals.to_numpy(dtype=np.int8).copy()
    for i in range(len(sig)):
        if sig[i] == 1:
            if not np.isfinite(p_win[i]) or float(p_win[i]) < float(th):
                sig[i] = 0
    return pd.Series(sig, index=signals.index, name="signal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest with rewards/penalties")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--model-run",
        type=Path,
        default=None,
        help="Optional training run dir with checkpoint.pt (label_mode trade_win recommended)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=None,
        help="Min P(win) for long entries; overrides auto-tune when set",
    )
    parser.add_argument(
        "--no-auto-tune",
        action="store_true",
        help="Disable auto-tuning from configs/trading.yaml (use checkpoint or raw signals)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg_dir = args.config_dir
    data_cfg = _load_yaml(root / cfg_dir / "data.yaml")
    smc_cfg = _load_yaml(root / cfg_dir / "smc.yaml")
    trading_cfg = _load_yaml(root / cfg_dir / "trading.yaml")

    ensure_yahoo_data(root, data_cfg)
    raw_path = root / data_cfg["raw_csv"]
    df = load_ohlcv_csv(raw_path, timezone=data_cfg.get("timezone"))

    acfg = analysis_config_from_yaml(smc_cfg, trading_cfg)
    analysis = analyze_chart(df, ma_period=acfg["ma_period"], atr_period=acfg["atr_period"])
    signals = compute_signals(analysis, trading_cfg)

    rcfg = reward_config_from_yaml(trading_cfg)
    target_wr = float(trading_cfg.get("target_win_rate", 0.5))
    min_trades_tune = int(trading_cfg.get("min_trades_for_tune", 8))
    auto_tune_prob = bool(trading_cfg.get("auto_tune_min_prob", True)) and not args.no_auto_tune
    auto_tune_base = bool(trading_cfg.get("auto_tune_baseline_signals", True)) and not args.no_auto_tune

    result: BacktestResult | None = None
    notes: list[str] = []

    if args.model_run is not None:
        run_dir = args.model_run if args.model_run.is_absolute() else (root / args.model_run)
        ckpt_path = run_dir / "checkpoint.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        train_cfg_run = _load_yaml(run_dir / "train.yaml")
        lookback = int(train_cfg_run.get("lookback", 20))

        feat_df = compute_feature_frame(df, ma_period=int(smc_cfg["ma_period"]))
        feat_np = feat_df[list(FEATURE_NAMES)].values.astype(np.float32)
        p_win, _ = predict_p_win_per_bar(feat_np, run_dir, lookback)

        vp = ckpt.get("val_precision_at_threshold")
        vp_s = f"{float(vp):.4f}" if vp is not None else "n/a"
        notes.append(
            f"[model] run={run_dir.name} label_mode={ckpt.get('label_mode', '?')} val_prec={vp_s}"
        )

        if args.min_prob is not None:
            th = float(args.min_prob)
            signals = _apply_long_prob_filter(signals, p_win, th)
            notes.append(f"min_prob={th:.3f} (manual)")
        elif auto_tune_prob:
            th, result, scan = find_min_prob_for_target_wr(
                df,
                analysis,
                signals,
                p_win,
                trading_cfg,
                rcfg,
                target_wr=target_wr,
                min_trades=min_trades_tune,
            )
            if result is not None and th is not None:
                wn = sum(1 for t in result.trades if t.exit_reason == "tp")
                nt = len(result.trades)
                wr_now = wn / nt if nt else 0.0
                met = wr_now + 1e-9 >= target_wr and nt >= min_trades_tune
                notes.append(
                    f"auto_tune model: min_prob={th:.3f} WR={wr_now:.2%} trades={nt} "
                    f"({'met' if met else 'best-feasible'} target>={target_wr:.0%}; scan={len(scan)})"
                )
            else:
                th_fb = float(ckpt.get("min_prob_threshold", 0.55))
                signals = _apply_long_prob_filter(signals, p_win, th_fb)
                notes.append(f"auto_tune: no feasible WR>={target_wr:.0%}; fallback min_prob={th_fb:.3f}")
                result = None
        else:
            th = float(ckpt.get("min_prob_threshold", 0.5))
            signals = _apply_long_prob_filter(signals, p_win, th)
            notes.append(f"min_prob={th:.3f} (checkpoint, auto_tune off)")

    elif auto_tune_base:
        lt, result, sig = find_long_threshold_for_target_wr(
            analysis,
            df,
            trading_cfg,
            rcfg,
            target_wr=target_wr,
            min_trades=min_trades_tune,
        )
        if result is not None and sig is not None and lt is not None:
            signals = sig
            wn = sum(1 for t in result.trades if t.exit_reason == "tp")
            nt = len(result.trades)
            wr_now = wn / nt if nt else 0.0
            if wr_now + 1e-9 >= target_wr and nt >= min_trades_tune:
                notes.append(
                    f"auto_tune baseline: long signal_dist_to_ma_long={lt:.3f} "
                    f"WR={wr_now:.2%} (met target>={target_wr:.0%}, trades={nt})"
                )
            else:
                notes.append(
                    f"auto_tune baseline: long signal_dist_to_ma_long={lt:.3f} "
                    f"WR={wr_now:.2%} (best effort, target>={target_wr:.0%} not met, trades={nt})"
                )
        else:
            notes.append("auto_tune baseline: no threshold; using default signals")
            result = None

    if result is None:
        result = run_backtest(df, analysis, signals, trading_cfg, rcfg)

    for line in notes:
        print(line)

    wins = sum(1 for t in result.trades if t.exit_reason == "tp")
    losses = sum(1 for t in result.trades if t.exit_reason == "sl")
    timeouts = sum(1 for t in result.trades if t.exit_reason == "timeout")
    n = len(result.trades)
    decided = wins + losses

    print(f"trades={n} win_tp={wins} loss_sl={losses} timeout={timeouts}")
    if n:
        win_rate = 100.0 * wins / n
        print(f"win_rate={win_rate:.2f}%  (TP / all trades)")
        if decided > 0:
            win_rate_tp_sl = 100.0 * wins / decided
            print(f"win_rate_tp_sl={win_rate_tp_sl:.2f}%  (TP / (TP+SL), excl. timeout)")
    else:
        print("win_rate=n/a  (no trades)")

    print(f"total_reward={result.total_reward:.4f}")
    if result.trades:
        avg_r = sum(t.realized_r for t in result.trades) / len(result.trades)
        print(f"avg_realized_R={avg_r:.4f}")
        for k, t in enumerate(result.trades[:10]):
            print(
                f"  [{k}] {t.side} entry@{t.entry_price:.5f} exit@{t.exit_price:.5f} "
                f"{t.exit_reason} R={t.realized_r:.2f} reward={t.reward:.3f}"
            )
        if len(result.trades) > 10:
            print(f"  ... {len(result.trades) - 10} more trades")


if __name__ == "__main__":
    main()
