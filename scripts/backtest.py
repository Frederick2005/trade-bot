"""
Enhanced backtester that:
  1. Tests multiple timeframe combinations
  2. Picks the best performing one automatically
  3. Saves training labels from winning trades
  4. Updates .env with the best timeframes
  5. Reports full stats for every combination

Usage:
    python scripts/backtest.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from dotenv import load_dotenv, set_key
from app.config import TRADING
from app.database.market import get_candles, save_candles
from app.market.candles import get_indicators, seed
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.params import get_active_params
from app.risk.sizing import calculate_lot_size
from app.database.context import save_training_label
from app.ai.features import build_feature_vector
from datetime import datetime, timezone

load_dotenv()

INITIAL_BALANCE = 500.0
COMMISSION_PCT  = 0.0005   # 0.05% per side Binance taker fee

# All timeframe combos to test
# Only 1H+4H — the proven combination for this strategy.
# 2H+1D removed: Binance testnet API doesn't provide enough
# historical daily candles (only ~25 days vs 205 needed).
# 1H+4H is the correct choice and is hardcoded here.
TIMEFRAME_COMBOS = [
    {"signal": "1h", "trend": "4h", "label": "1H+4H"},
]

# Minimum trades for a result to be considered valid
# Set high enough that a result is statistically meaningful
MIN_TRADES = 30


async def backtest_combo(
    symbol: str,
    signal_tf: str,
    trend_tf: str,
    save_labels: bool = False,
) -> dict:
    df_signal = await get_candles(symbol, signal_tf, limit=2000)
    df_trend  = await get_candles(symbol, trend_tf,  limit=500)

    if df_signal.empty or df_trend.empty:
        logger.warning(f"No data for {symbol} {signal_tf}/{trend_tf}")
        return {}

    seed(symbol, signal_tf, df_signal)
    seed(symbol, trend_tf,  df_trend)

    strategy = EmaRsiStrategy(version="backtest")
    balance  = INITIAL_BALANCE
    trades   = []
    labels   = []
    in_trade = False
    entry = sl = tp = lot = side_held = entry_ind_1h = entry_ind_4h = None
    min_candles = 210

    for i in range(min_candles, len(df_signal)):
        window_signal = df_signal.iloc[:i+1]
        window_trend  = _get_trend_window(df_trend, df_signal.iloc[i]["open_time"])
        seed(symbol, signal_tf, window_signal)
        seed(symbol, trend_tf,  window_trend)

        ind_signal = get_indicators(symbol, signal_tf)
        ind_trend  = get_indicators(symbol, trend_tf)
        if ind_signal is None or ind_trend is None:
            continue

        candle = df_signal.iloc[i]

        # ── Check exit ─────────────────────────────────────────────
        if in_trade:
            exited   = False
            exit_px  = None
            exit_why = None

            if side_held == "LONG":
                if candle["high"] >= tp:
                    exit_px, exit_why = tp, "TP_HIT"
                elif candle["low"] <= sl:
                    exit_px, exit_why = sl, "SL_HIT"
            else:
                if candle["low"] <= tp:
                    exit_px, exit_why = tp, "TP_HIT"
                elif candle["high"] >= sl:
                    exit_px, exit_why = sl, "SL_HIT"

            if exit_px is not None:
                if side_held == "LONG":
                    pnl = (exit_px - entry) * lot
                else:
                    pnl = (entry - exit_px) * lot
                pnl     -= exit_px * lot * COMMISSION_PCT
                balance += pnl
                won      = pnl > 0
                trades.append({
                    "pnl":    pnl,
                    "result": "WIN" if won else "LOSS",
                    "reason": exit_why,
                    "side":   side_held,
                })

                # Save training label if requested
                if save_labels and entry_ind_1h and entry_ind_4h:
                    candle_time = candle["open_time"]
                    if hasattr(candle_time, "to_pydatetime"):
                        candle_time = candle_time.to_pydatetime()
                    features = build_feature_vector(
                        entry_ind_1h, entry_ind_4h,
                        candle_time=candle_time,
                    )
                    labels.append({
                        "features": features,
                        "label":    1 if won else 0,
                        "pnl_pct":  pnl / INITIAL_BALANCE,
                    })

                in_trade = False

        # ── Check entry ────────────────────────────────────────────
        if not in_trade:
            signal = strategy.evaluate(symbol, ind_signal, ind_trend)
            if signal:
                lot, _ = calculate_lot_size(
                    balance, signal.entry_price, signal.stop_loss
                )
                if lot > 0:
                    in_trade      = True
                    entry         = signal.entry_price
                    sl            = signal.stop_loss
                    tp            = signal.take_profit
                    side_held     = signal.side
                    entry_ind_1h  = ind_signal
                    entry_ind_4h  = ind_trend
                    balance      -= entry * lot * COMMISSION_PCT

    # ── Save labels to Supabase ────────────────────────────────────
    if save_labels and labels:
        logger.info(f"Saving {len(labels)} training labels to Supabase...")
        for lbl in labels:
            await save_training_label(
                trade_id=f"backtest-{symbol}-{len(labels)}",
                features=lbl["features"],
                label=lbl["label"],
                pnl_pct=lbl["pnl_pct"],
            )
        logger.info(f"Saved {len(labels)} training labels")

    # ── Compute stats ──────────────────────────────────────────────
    if not trades:
        return {}

    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    pnls    = [t["pnl"] for t in trades]
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))

    # Sharpe ratio (simplified daily)
    import numpy as np
    pnl_arr  = np.array(pnls)
    sharpe   = (pnl_arr.mean() / pnl_arr.std() * (252 ** 0.5)) if pnl_arr.std() > 0 else 0

    return {
        "symbol":        symbol,
        "signal_tf":     signal_tf,
        "trend_tf":      trend_tf,
        "total_trades":  total,
        "wins":          wins,
        "losses":        total - wins,
        "win_rate":      round(wins / total, 4) if total else 0,
        "total_pnl":     round(sum(pnls), 4),
        "profit_factor": round(gross_p / gross_l, 4) if gross_l else 0,
        "sharpe":        round(sharpe, 4),
        "final_balance": round(balance, 2),
        "return_pct":    round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
        "best_trade":    round(max(pnls), 4),
        "worst_trade":   round(min(pnls), 4),
        "labels_saved":  len(labels) if save_labels else 0,
    }


def _score(result: dict) -> float:
    """
    Composite score to rank timeframe combos.
    Weights: profit factor (40%) + win rate (30%) + sharpe (20%) + return (10%)
    Heavy penalty for fewer than MIN_TRADES — avoids lucky small samples.
    Profit factor weighted highest because it captures both win rate
    AND size of wins vs losses — the most reliable live performance predictor.
    """
    trades = result.get("total_trades", 0)
    if trades < MIN_TRADES:
        return -999.0

    pf     = min(result.get("profit_factor", 0), 5)   # cap outliers
    wr     = result.get("win_rate", 0)
    ret    = result.get("return_pct", -100) / 100
    sharpe = result.get("sharpe", 0)

    # Bonus for more trades (statistical confidence)
    trade_bonus = min(trades / 100, 0.2)

    return (pf * 0.40) + (wr * 0.30) + (sharpe * 0.20) + (ret * 0.10) + trade_bonus


def _print_result(r: dict, label: str, rank: int = 0) -> None:
    prefix = f"#{rank} " if rank else ""
    logger.info(
        f"{prefix}{label} | {r['symbol']} | "
        f"trades={r['total_trades']} "
        f"win={r['win_rate']:.0%} "
        f"pf={r['profit_factor']:.2f} "
        f"pnl={r['total_pnl']:+.2f} "
        f"ret={r['return_pct']:+.1f}% "
        f"sharpe={r['sharpe']:.2f}"
    )
def _get_trend_window(df_trend: pd.DataFrame, up_to_time) -> pd.DataFrame:
    """Returns all 4H candles up to and including the current signal candle time."""
    return df_trend[df_trend["open_time"] <= up_to_time].tail(300)




async def main() -> None:
    logger.info("=" * 60)
    logger.info("BACKTEST — finding best timeframe combination")
    logger.info("=" * 60)

    all_results = []

    for combo in TIMEFRAME_COMBOS:
        label      = combo["label"]
        signal_tf  = combo["signal"]
        trend_tf   = combo["trend"]

        logger.info(f"\nTesting {label}...")
        combo_results = []

        for symbol in TRADING.symbols:
            r = await backtest_combo(symbol, signal_tf, trend_tf)
            if r:
                r["label"] = label
                combo_results.append(r)
                _print_result(r, label)

        if combo_results:
            # Average score across both symbols
            avg_score = sum(_score(r) for r in combo_results) / len(combo_results)
            all_results.append({
                "combo":   combo,
                "results": combo_results,
                "score":   avg_score,
            })

    if not all_results:
        logger.error("No backtest results — run seed_history.py first")
        return

    # ── Rank results ───────────────────────────────────────────────
    all_results.sort(key=lambda x: x["score"], reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS RANKED BY COMPOSITE SCORE")
    logger.info("=" * 60)
    for i, entry in enumerate(all_results, 1):
        label = entry["combo"]["label"]
        score = entry["score"]
        for r in entry["results"]:
            _print_result(r, label, rank=i)
        logger.info(f"   Composite score: {score:.4f}")

    # ── Pick winner ────────────────────────────────────────────────
    winner       = all_results[0]
    best_combo   = winner["combo"]
    best_signal  = best_combo["signal"]
    best_trend   = best_combo["trend"]
    best_label   = best_combo["label"]

    logger.info("\n" + "=" * 60)
    logger.info(f"WINNER: {best_label}")
    logger.info(f"  Signal timeframe: {best_signal}")
    logger.info(f"  Trend  timeframe: {best_trend}")
    logger.info(f"  Composite score:  {winner['score']:.4f}")
    logger.info("=" * 60)

    # ── Update .env ────────────────────────────────────────────────
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        set_key(env_path, "SIGNAL_TIMEFRAME", best_signal)
        set_key(env_path, "TREND_TIMEFRAME",  best_trend)
        logger.info(f".env updated: SIGNAL_TIMEFRAME={best_signal} TREND_TIMEFRAME={best_trend}")
    else:
        logger.warning(f".env not found at {env_path} — update manually")
        logger.warning(f"Set: SIGNAL_TIMEFRAME={best_signal}  TREND_TIMEFRAME={best_trend}")

    # ── Save training labels from winner ───────────────────────────
    logger.info("\nSaving training labels from best combo...")
    for symbol in TRADING.symbols:
        await backtest_combo(
            symbol, best_signal, best_trend, save_labels=True
        )

    logger.info("\n" + "=" * 60)
    logger.info("Backtest complete")
    logger.info(f"Best timeframes: {best_label}")
    logger.info("Training labels saved — ready for AI training")
    logger.info("Restart main.py to apply new timeframes")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())