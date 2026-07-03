"""
OPTIMIZED backtester — pre-computes all indicators upfront.
10-20x faster than the previous candle-by-candle re-seeding approach.

Usage: python scripts/backtest.py
"""
import asyncio
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import ta
from loguru import logger
from dotenv import load_dotenv, set_key

from app.config import TRADING
from app.database.market import get_candles
from app.risk.sizing import calculate_lot_size
from app.database.context import save_training_label
from app.ai.features import build_feature_vector
from datetime import datetime, timezone

load_dotenv()

INITIAL_BALANCE = 500.0
COMMISSION_PCT  = 0.0005
MIN_TRADES      = 30
TIMEFRAME_COMBOS = [
    {"signal": "15m", "trend": "1h", "label": "15M+1H"},
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute all indicators for the entire dataframe at once.
    This is the key optimization — O(n) instead of O(n²).
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df = df.copy()
    df["ema50"]   = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema200"]  = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["rsi"]     = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["atr"]     = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    vol_avg            = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_avg

    df["price_vs_ema50"]  = (close - df["ema50"]) / df["ema50"] * 100
    df["trend_strength"]  = (df["ema50"] - df["ema200"]) / df["ema200"] * 100
    df["volatility_pct"]  = df["atr"] / close * 100
    df["ema_gap_pct"]     = (df["ema50"] - df["ema200"]) / df["ema200"] * 100

    candle_range         = high - low
    candle_body          = abs(close - df["open"])
    df["candle_body_pct"] = candle_body / candle_range.replace(0, np.nan)
    df["is_bullish"]      = (close > df["open"]).astype(float)
    df["ema50_slope"]     = (df["ema50"] - df["ema50"].shift(4)) / df["ema50"].shift(4) * 100

    return df


def row_to_indicators(row: pd.Series) -> dict:
    """Convert a precomputed dataframe row to the indicators dict format."""
    return {
        "price":            row["close"],
        "ema50":            row["ema50"],
        "ema200":           row["ema200"],
        "rsi":              row["rsi"],
        "atr":              row["atr"],
        "volume_ratio":     row["volume_ratio"],
        "price_vs_ema50":   row["price_vs_ema50"],
        "trend_strength":   row["trend_strength"],
        "volatility_pct":   row["volatility_pct"],
        "ema_gap_pct":      row["ema_gap_pct"],
        "candle_body_pct":  row.get("candle_body_pct", 0.0),
        "is_bullish_candle": row["is_bullish"] == 1.0,
        "ema50_slope":      row["ema50_slope"],
    }


def get_trend_indicators(df_trend: pd.DataFrame, signal_time) -> dict | None:
    """Find the most recent trend candle at or before signal_time."""
    mask = df_trend["open_time"] <= signal_time
    if not mask.any():
        return None
    row = df_trend[mask].iloc[-1]
    if pd.isna(row["ema200"]):
        return None
    return row_to_indicators(row)


def evaluate_signal(
    ind_signal: dict,
    ind_trend: dict,
    atr_multiplier: float = 1.5,
    tp_rr: float = 1.5,
) -> dict | None:
    """Pure function — no class needed, no params DB call."""
    price          = ind_signal.get("price")
    ema50_s        = ind_signal.get("ema50")
    ema200_s       = ind_signal.get("ema200")
    rsi_s          = ind_signal.get("rsi")
    atr_s          = ind_signal.get("atr")
    body_pct       = ind_signal.get("candle_body_pct", 0.0)
    bullish        = ind_signal.get("is_bullish_candle", False)
    price_vs_ema50 = ind_signal.get("price_vs_ema50", 0.0)
    ema50_slope    = ind_signal.get("ema50_slope", 0.0)
    ema50_t        = ind_trend.get("ema50")
    ema200_t       = ind_trend.get("ema200")

    if not all([price, ema50_s, ema200_s, ema50_t, ema200_t, atr_s]):
        return None
    if rsi_s is None or pd.isna(rsi_s):
        return None

    if not (ema50_t > ema200_t):        return None  # 1h uptrend
    if not (ema50_s > ema200_s):        return None  # 15m uptrend
    if not (-1.5 <= price_vs_ema50 <= 1.5): return None  # pullback zone
    if not (42.0 <= rsi_s <= 62.0):     return None  # RSI zone
    if not bullish:                      return None  # bullish candle
    if body_pct < 0.35:                  return None  # real body
    if ema50_slope <= 0:                 return None  # slope rising

    sl = price - (atr_s * atr_multiplier)
    if (price - sl) <= 0:
        return None

    tp = price + ((price - sl) * tp_rr)

    return {"side": "LONG", "entry": price, "sl": sl, "tp": tp}


async def backtest_combo(
    symbol: str,
    signal_tf: str,
    trend_tf: str,
    save_labels: bool = False,
) -> dict:
    logger.info(f"Loading data for {symbol}...")
    df_signal = await get_candles(symbol, signal_tf, limit=70200)
    df_trend  = await get_candles(symbol, trend_tf,  limit=17600)

    if df_signal.empty or df_trend.empty:
        logger.warning(f"No data for {symbol}")
        return {}

    logger.info(f"Computing indicators for {symbol} {signal_tf} ({len(df_signal)} candles)...")
    df_signal = compute_indicators(df_signal)
    df_trend  = compute_indicators(df_trend)

    # Drop rows where indicators aren't ready yet (first 200 candles)
    df_signal = df_signal.dropna(subset=["ema200", "rsi", "atr"]).reset_index(drop=True)
    df_trend  = df_trend.dropna(subset=["ema200"]).reset_index(drop=True)

    logger.info(f"Running backtest for {symbol} ({len(df_signal)} signal candles)...")

    balance  = INITIAL_BALANCE
    trades   = []
    labels   = []
    in_trade = False
    entry = sl = tp = entry_ind_s = entry_ind_t = None

    for i in range(len(df_signal)):
        row_s = df_signal.iloc[i]
        ind_s = row_to_indicators(row_s)
        ind_t = get_trend_indicators(df_trend, row_s["open_time"])

        if ind_t is None:
            continue

        # ── Check exit ──────────────────────────────────────────────
        if in_trade:
            exit_px  = None
            exit_why = None

            if row_s["high"] >= tp:
                exit_px, exit_why = tp, "TP_HIT"
            elif row_s["low"] <= sl:
                exit_px, exit_why = sl, "SL_HIT"

            if exit_px is not None:
                pnl      = (exit_px - entry) * lot
                pnl     -= exit_px * lot * COMMISSION_PCT
                balance += pnl
                won      = pnl > 0
                trades.append({
                    "pnl":    pnl,
                    "result": "WIN" if won else "LOSS",
                    "reason": exit_why,
                })

                if save_labels and entry_ind_s and entry_ind_t:
                    candle_time = row_s["open_time"]
                    if hasattr(candle_time, "to_pydatetime"):
                        candle_time = candle_time.to_pydatetime()
                    try:
                        features = build_feature_vector(
                            entry_ind_t, entry_ind_s, candle_time=candle_time
                        )
                        labels.append({
                            "trade_id": str(uuid.uuid4()),
                            "features": features,
                            "label":    1 if won else 0,
                            "pnl_pct":  pnl / INITIAL_BALANCE,
                        })
                    except Exception as e:
                        logger.debug(f"Feature build error: {e}")

                in_trade = False

        # ── Check entry ─────────────────────────────────────────────
        if not in_trade:
            sig = evaluate_signal(ind_s, ind_t)
            if sig:
                lot, _ = calculate_lot_size(balance, sig["entry"], sig["sl"])
                if lot > 0:
                    in_trade    = True
                    entry       = sig["entry"]
                    sl          = sig["sl"]
                    tp          = sig["tp"]
                    entry_ind_s = ind_s
                    entry_ind_t = ind_t
                    balance    -= entry * lot * COMMISSION_PCT

    # ── Save labels ─────────────────────────────────────────────────
    if save_labels and labels:
        logger.info(f"Saving {len(labels)} training labels...")
        saved = 0
        for lbl in labels:
            ok = await save_training_label(
                trade_id=lbl["trade_id"],
                features=lbl["features"],
                label=lbl["label"],
                pnl_pct=lbl["pnl_pct"],
            )
            if ok:
                saved += 1
        logger.info(f"Saved {saved}/{len(labels)} labels")

    if not trades:
        return {}

    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    pnls    = [t["pnl"] for t in trades]
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pnl_arr = np.array(pnls)
    sharpe  = (pnl_arr.mean() / pnl_arr.std() * (252 ** 0.5)) if pnl_arr.std() > 0 else 0

    return {
        "symbol":        symbol,
        "signal_tf":     signal_tf,
        "trend_tf":      trend_tf,
        "total_trades":  total,
        "wins":          wins,
        "losses":        total - wins,
        "win_rate":      round(wins / total, 4),
        "total_pnl":     round(sum(pnls), 4),
        "profit_factor": round(gross_p / gross_l, 4) if gross_l else 0,
        "sharpe":        round(sharpe, 4),
        "final_balance": round(balance, 2),
        "return_pct":    round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
        "best_trade":    round(max(pnls), 4),
        "worst_trade":   round(min(pnls), 4),
        "trades_per_day": round(total / 730, 1),
        "labels_saved":  len(labels) if save_labels else 0,
    }


def _print_result(r: dict, label: str) -> None:
    logger.info(
        f"{label} | {r['symbol']} | "
        f"trades={r['total_trades']} ({r['trades_per_day']}/day) "
        f"win={r['win_rate']:.0%} "
        f"pf={r['profit_factor']:.2f} "
        f"pnl={r['total_pnl']:+.2f} "
        f"ret={r['return_pct']:+.1f}% "
        f"sharpe={r['sharpe']:.2f}"
    )


async def main() -> None:
    logger.info("=" * 60)
    logger.info("OPTIMIZED BACKTEST — 15M+1H")
    logger.info("=" * 60)

    all_results = []

    for combo in TIMEFRAME_COMBOS:
        label     = combo["label"]
        signal_tf = combo["signal"]
        trend_tf  = combo["trend"]
        combo_results = []

        for symbol in TRADING.symbols:
            r = await backtest_combo(symbol, signal_tf, trend_tf)
            if r:
                r["label"] = label
                combo_results.append(r)
                _print_result(r, label)

        if combo_results:
            avg_score = sum(
                (r.get("profit_factor", 0) * 0.4 +
                 r.get("win_rate", 0) * 0.3 +
                 r.get("sharpe", 0) * 0.2 +
                 r.get("return_pct", -100) / 100 * 0.1)
                for r in combo_results
            ) / len(combo_results)
            all_results.append({
                "combo": combo, "results": combo_results, "score": avg_score
            })

    if not all_results:
        logger.error("No results — run seed_history.py first")
        return

    winner     = all_results[0]
    best_combo = winner["combo"]

    logger.info("\n" + "=" * 60)
    logger.info(f"WINNER: {best_combo['label']}")
    logger.info("=" * 60)

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        set_key(env_path, "SIGNAL_TIMEFRAME", best_combo["signal"])
        set_key(env_path, "TREND_TIMEFRAME",  best_combo["trend"])
        logger.info(f".env updated")

    logger.info("\nSaving training labels from best combo...")
    for symbol in TRADING.symbols:
        await backtest_combo(
            symbol, best_combo["signal"], best_combo["trend"], save_labels=True
        )

    logger.info("\n" + "=" * 60)
    logger.info("Backtest complete — training labels saved")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())