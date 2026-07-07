"""
scripts/backtest.py

Optimized backtester for up to 5 years of 15m + 1h data, per symbol.

What changed vs the previous version, and why it matters at 5-year scale:

1. THE BIG ONE — trend-candle lookup was O(n_signal x n_trend): for every
   single signal-timeframe row, it re-scanned the ENTIRE trend dataframe
   with a boolean mask to find "the most recent trend candle at or before
   this time". At 5 years of 15m+1h data that's ~175k x ~44k = ~7.7 BILLION
   comparisons per symbol, per run — measured at several minutes just for
   that step. Replaced with a single pd.merge_asof (sorted backward-join),
   which does the same join correctly in ~50ms.

2. Candle fetching is now cached to local parquet files under data_cache/.
   get_candles() pages through Supabase 1000 rows at a time — for 5 years of
   15m candles that's ~176 sequential network round trips, and the old
   script did this TWICE (once to score the combo, once again to save
   training labels). The cache makes every run after the first one instant,
   and only re-fetches when the cache is stale (older than a few candle
   intervals).

3. LONG *and* SHORT support, using the exact same app.strategy.signal_logic
   .evaluate_entry() function the live bot uses (via EmaRsiStrategy) — so
   backtest results now actually reflect what the live/paper bot will do.
   The previous version only ever tested LONG trades.

4. The per-candle simulation loop uses itertuples() instead of iloc[i],
   which avoids re-boxing each row into a pandas Series (a few x faster for
   this size of loop).

Usage:
    python scripts/backtest.py                  # default: 5 years, all configured symbols
    BACKTEST_YEARS=2 python scripts/backtest.py  # override lookback window
"""
import asyncio
import os
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import ta
from loguru import logger
from dotenv import load_dotenv, set_key

from app.config import TRADING
from app.database.market import get_candles
from app.database.context import save_training_label
from app.ai.features import build_feature_vector
from app.strategy.signal_logic import evaluate_entry

load_dotenv()

INITIAL_BALANCE  = 500.0
COMMISSION_PCT   = 0.0005

MAX_OPEN_TRADES  = 10       # allow up to 10 concurrent positions, matches app/engine.py
RISK_PER_TRADE   = 0.01     # 1% risk per trade
MIN_TRADES       = 30

# SIZING_MODE controls what "1% risk per trade" is 1% OF:
#
# "compounding" — 1% of the CURRENT (running) balance. Realistic: models
#   what an actual account would do, including the fact that a losing
#   streak shrinks future position sizes. Consequence: if the strategy has
#   negative expectancy, the account can decay toward zero well before your
#   data ends, and the account-blown guard below will stop the simulation
#   at that point — meaning you never see how the strategy would have
#   performed on the LATER part of your historical window. Correct for
#   modeling a real account; truncates your sample if the strategy loses.
#
# "fixed" — (default here) 1% of the ORIGINAL INITIAL_BALANCE, every trade,
#   regardless of running P&L. This decouples signal-quality measurement
#   from compounding/account-survival effects: every trade in your dataset
#   gets evaluated and counted, so win_rate/profit_factor/RR reflect your
#   FULL historical window. total_pnl and final_balance in "fixed" mode are
#   NOT a realistic equity curve anymore (a real compounding account would
#   behave differently) — read them as "sum of fixed-size R-outcomes", not
#   as "what your account balance would actually have been".
#
# Use "fixed" to answer "is there any edge here at all, across the whole
# dataset". Use "compounding" to answer "would a real account survive this".
# They answer different questions — neither is more "correct" in isolation.
SIZING_MODE = os.getenv("BACKTEST_SIZING_MODE", "fixed")

YEARS            = float(os.getenv("BACKTEST_YEARS", "5"))
CANDLES_PER_DAY  = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}
TF_SECONDS       = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

TIMEFRAME_COMBOS = [
    {"signal": "15m", "trend": "1h", "label": "15M+1H"},
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _candles_for_years(timeframe: str, years: float) -> int:
    per_day = CANDLES_PER_DAY.get(timeframe, 24)
    return int(years * 365 * per_day) + 250  # +250 bar buffer for EMA200 warm-up


# ── Cached candle fetching ──────────────────────────────────────────────────

async def get_candles_cached(symbol: str, timeframe: str, limit: int, max_age_candles: int = 3) -> pd.DataFrame:
    """
    Wraps app.database.market.get_candles() with a local parquet cache.
    Supabase paging is the slow part of a fresh fetch (~176 round trips for
    5 years of 15m data) — once fetched, reuse it across repeated backtest
    runs (e.g. while tuning thresholds) unless the cache looks stale.
    """
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_{timeframe}.parquet")

    if os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty and len(cached) >= limit:
                last_time = pd.Timestamp(cached["open_time"].max())
                if last_time.tzinfo is None:
                    last_time = last_time.tz_localize("UTC")
                age_seconds = (datetime.now(timezone.utc) - last_time.to_pydatetime()).total_seconds()
                tf_sec = TF_SECONDS.get(timeframe, 3600)
                if age_seconds < tf_sec * max_age_candles:
                    logger.info(
                        f"Cache hit: {symbol} {timeframe} "
                        f"({len(cached):,} rows, {age_seconds / 3600:.1f}h old)"
                    )
                    return cached.tail(limit).reset_index(drop=True)
                logger.info(f"Cache stale for {symbol} {timeframe} ({age_seconds / 3600:.1f}h old) — refetching")
        except Exception as e:
            logger.warning(f"Cache read failed for {symbol} {timeframe}: {e} — refetching")

    t0 = time.time()
    logger.info(f"Fetching from Supabase: {symbol} {timeframe} (limit={limit:,})...")
    df = await get_candles(symbol, timeframe, limit=limit)
    logger.info(f"Fetched {len(df):,} rows for {symbol} {timeframe} in {time.time() - t0:.1f}s")

    if not df.empty:
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Could not write cache for {symbol} {timeframe}: {e}")

    return df


# ── Vectorized indicator computation (unchanged shape, already O(n)) ───────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute all indicators once — O(n), not O(n^2)."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df = df.copy()
    df["ema50"]  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["rsi"]    = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["atr"]    = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    vol_avg              = volume.rolling(20).mean()
    df["volume_ratio"]   = volume / vol_avg
    df["price_vs_ema50"] = (close - df["ema50"]) / df["ema50"] * 100
    df["trend_strength"] = (df["ema50"] - df["ema200"]) / df["ema200"] * 100
    df["volatility_pct"] = df["atr"] / close * 100
    df["ema_gap_pct"]    = (df["ema50"] - df["ema200"]) / df["ema200"] * 100

    candle_range           = high - low
    candle_body            = (close - df["open"]).abs()
    df["candle_body_pct"]  = candle_body / candle_range.replace(0, np.nan)
    df["is_bullish_candle"] = close > df["open"]
    df["ema50_slope"]      = (df["ema50"] - df["ema50"].shift(4)) / df["ema50"].shift(4) * 100

    return df


# ── THE FIX: vectorized trend join instead of O(n x m) row-by-row scan ─────

def attach_trend_indicators(df_signal: pd.DataFrame, df_trend: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the old get_trend_indicators() row-by-row boolean-mask scan
    (O(n_signal x n_trend)) with a single pd.merge_asof backward join
    (O(n log n)). For each signal candle, attaches the most recent trend
    candle at or before that time — same semantics, ~4500x faster at
    5-year data volumes (measured: ~4 min -> ~50ms for the join itself).
    """
    trend_cols = [
        "open_time", "ema50", "ema200", "rsi", "atr", "volume_ratio",
        "price_vs_ema50", "trend_strength", "volatility_pct",
        "ema_gap_pct", "candle_body_pct", "is_bullish_candle", "ema50_slope",
    ]
    trend_slim = df_trend[trend_cols].rename(
        columns={c: f"{c}_trend" for c in trend_cols if c != "open_time"}
    )
    merged = pd.merge_asof(
        df_signal.sort_values("open_time"),
        trend_slim.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    return merged


def row_to_signal_indicators(row) -> dict:
    return {
        "price":             row.close,
        "ema50":             row.ema50,
        "ema200":            row.ema200,
        "rsi":               row.rsi,
        "atr":               row.atr,
        "volume_ratio":      row.volume_ratio,
        "price_vs_ema50":    row.price_vs_ema50,
        "trend_strength":    row.trend_strength,
        "volatility_pct":    row.volatility_pct,
        "ema_gap_pct":       row.ema_gap_pct,
        "candle_body_pct":   row.candle_body_pct if not pd.isna(row.candle_body_pct) else 0.0,
        "is_bullish_candle": bool(row.is_bullish_candle),
        "ema50_slope":       row.ema50_slope,
    }


def row_to_trend_indicators(row) -> dict:
    return {
        "price":             row.close,  # signal-tf close; trend join doesn't carry its own close
        "ema50":             row.ema50_trend,
        "ema200":            row.ema200_trend,
        "rsi":               row.rsi_trend,
        "atr":               row.atr_trend,
        "volume_ratio":      row.volume_ratio_trend,
        "price_vs_ema50":    row.price_vs_ema50_trend,
        "trend_strength":    row.trend_strength_trend,
        "volatility_pct":    row.volatility_pct_trend,
        "ema_gap_pct":       row.ema_gap_pct_trend,
        "candle_body_pct":   row.candle_body_pct_trend if not pd.isna(row.candle_body_pct_trend) else 0.0,
        "is_bullish_candle": bool(row.is_bullish_candle_trend),
        "ema50_slope":       row.ema50_slope_trend,
    }


# ── Backtest core ────────────────────────────────────────────────────────────

async def backtest_combo(
    symbol: str,
    signal_tf: str,
    trend_tf: str,
    save_labels: bool = False,
) -> dict:
    t_start = time.time()

    signal_limit = _candles_for_years(signal_tf, YEARS)
    trend_limit  = _candles_for_years(trend_tf, YEARS)

    df_signal = await get_candles_cached(symbol, signal_tf, signal_limit)
    df_trend  = await get_candles_cached(symbol, trend_tf, trend_limit)

    if df_signal.empty or df_trend.empty:
        logger.warning(f"No data for {symbol} — run scripts/seed_history.py first")
        return {}

    logger.info(f"Computing indicators for {symbol} ({len(df_signal):,} signal candles)...")
    df_signal = compute_indicators(df_signal)
    df_trend  = compute_indicators(df_trend)

    df_signal = df_signal.dropna(subset=["ema200", "rsi", "atr"]).reset_index(drop=True)
    df_trend  = df_trend.dropna(subset=["ema200"]).reset_index(drop=True)

    if df_signal.empty or df_trend.empty:
        logger.warning(f"Not enough warmed-up candles for {symbol} after indicator dropna")
        return {}

    t_join = time.time()
    merged = attach_trend_indicators(df_signal, df_trend)
    logger.info(f"Trend join for {symbol}: {time.time() - t_join:.3f}s ({len(merged):,} rows)")

    balance     = INITIAL_BALANCE
    open_trades = []
    trades      = []
    labels      = []

    t_sim = time.time()
    for row in merged.itertuples(index=False):
        if pd.isna(row.ema200_trend):
            continue

        ind_s = row_to_signal_indicators(row)
        ind_t = row_to_trend_indicators(row)

        current_high = row.high
        current_low  = row.low

        # ── Check exits for all open trades ─────────────────────────
        still_open = []
        for trade in open_trades:
            exit_px, exit_why = None, None
            if trade["side"] == "LONG":
                if current_high >= trade["tp"]:
                    exit_px, exit_why = trade["tp"], "TP_HIT"
                elif current_low <= trade["sl"]:
                    exit_px, exit_why = trade["sl"], "SL_HIT"
                pnl = ((exit_px - trade["entry"]) * trade["lot"]) if exit_px else None
            else:  # SHORT
                if current_low <= trade["tp"]:
                    exit_px, exit_why = trade["tp"], "TP_HIT"
                elif current_high >= trade["sl"]:
                    exit_px, exit_why = trade["sl"], "SL_HIT"
                pnl = ((trade["entry"] - exit_px) * trade["lot"]) if exit_px else None

            if exit_px is not None:
                pnl     -= exit_px * trade["lot"] * COMMISSION_PCT
                balance += pnl
                won      = pnl > 0
                trades.append({"pnl": pnl, "result": "WIN" if won else "LOSS", "reason": exit_why, "side": trade["side"]})

                if save_labels:
                    candle_time = row.open_time
                    if hasattr(candle_time, "to_pydatetime"):
                        candle_time = candle_time.to_pydatetime()
                    try:
                        features = build_feature_vector(trade["ind_s"], trade["ind_t"], candle_time=candle_time)
                        # Debug fields — these cost nothing (features is schemaless
                        # jsonb already) and would have let us diagnose the
                        # account-decay bug directly from training_labels instead
                        # of needing a code change + a fresh backtest run to find it.
                        features["_debug_symbol"] = symbol
                        features["_debug_side"] = trade["side"]
                        features["_debug_entry_price"] = trade["entry"]
                        features["_debug_exit_price"] = exit_px
                        features["_debug_exit_reason"] = exit_why
                        features["_debug_balance_at_close"] = round(balance, 6)
                        # Explicit, unambiguous fields — the generic "candle_body_pct"
                        # in the main feature vector reads from whichever indicator
                        # dict is passed FIRST to build_feature_vector(), which has
                        # been a source of real confusion (see signal_logic.py's
                        # MIN_BODY_PCT comment history). These two are always
                        # unambiguous regardless of argument order elsewhere.
                        features["_debug_signal_candle_body_pct"] = trade["ind_s"].get("candle_body_pct")
                        features["_debug_trend_candle_body_pct"]  = trade["ind_t"].get("candle_body_pct")
                        labels.append({
                            "trade_id": str(uuid.uuid4()),
                            "features": features,
                            "label": 1 if won else 0,
                            "pnl_pct": pnl / INITIAL_BALANCE,
                        })
                    except Exception as e:
                        logger.debug(f"Feature build error: {e}")
            else:
                still_open.append(trade)

        open_trades = still_open

        # ── Account-blown check ──────────────────────────────────────
        # Previously missing entirely: app/risk/sizing.py (used by the LIVE
        # engine) has a $5 minimum-notional floor, but this backtest's
        # inline sizing logic never got the same protection. In
        # "compounding" mode, negative expectancy + 1%-of-CURRENT-balance
        # sizing means balance shrinks every loss — over thousands of
        # trades it can decay toward a fraction of a cent while the
        # simulation keeps opening "trades" with dust-sized positions
        # forever. Confirmed on real data: ~65% of a 10,926-trade run were
        # economically meaningless (<0.0001% price-equivalent pnl) because
        # they happened after the account was already effectively wiped
        # out. In "fixed" mode this check can't trigger from compounding
        # decay (sizing is always relative to INITIAL_BALANCE, not the
        # running balance) — it only guards against INITIAL_BALANCE itself
        # being configured too small to trade at all.
        MIN_NOTIONAL = 5.0
        sizing_reference_balance = balance if SIZING_MODE == "compounding" else INITIAL_BALANCE
        if sizing_reference_balance * RISK_PER_TRADE <= 0 or sizing_reference_balance <= MIN_NOTIONAL:
            logger.warning(
                f"{symbol}: {'account balance' if SIZING_MODE == 'compounding' else 'INITIAL_BALANCE'} "
                f"(${sizing_reference_balance:.4f}) too small to continue trading at candle "
                f"{row.open_time} — stopping simulation here instead of generating "
                f"dust-sized phantom trades."
            )
            break

        # ── Open new trade if a slot is available ───────────────────
        if len(open_trades) < MAX_OPEN_TRADES:
            sig = evaluate_entry(ind_s, ind_t)
            if sig:
                risk_amount = sizing_reference_balance * RISK_PER_TRADE
                sl_dist = abs(sig["entry"] - sig["sl"])
                if sl_dist > 0:
                    lot = risk_amount / sl_dist
                    notional = lot * sig["entry"]
                    if lot > 0 and notional >= MIN_NOTIONAL:
                        cost = sig["entry"] * lot * COMMISSION_PCT
                        balance -= cost
                        open_trades.append({
                            "side": sig["side"], "entry": sig["entry"],
                            "sl": sig["sl"], "tp": sig["tp"], "lot": lot,
                            "ind_s": ind_s, "ind_t": ind_t,
                        })

    logger.info(f"Simulation loop for {symbol}: {time.time() - t_sim:.1f}s ({len(merged):,} candles)")

    # Close any still-open trades at the last available price
    if open_trades:
        last_price = merged.iloc[-1].close
        for trade in open_trades:
            pnl = ((last_price - trade["entry"]) if trade["side"] == "LONG" else (trade["entry"] - last_price)) * trade["lot"]
            pnl -= last_price * trade["lot"] * COMMISSION_PCT
            balance += pnl
            trades.append({"pnl": pnl, "result": "WIN" if pnl > 0 else "LOSS", "reason": "FORCED_CLOSE", "side": trade["side"]})

    if save_labels and labels:
        logger.info(f"Saving {len(labels)} training labels...")
        saved = 0
        for lbl in labels:
            ok = await save_training_label(
                trade_id=lbl["trade_id"], features=lbl["features"],
                label=lbl["label"], pnl_pct=lbl["pnl_pct"],
            )
            if ok:
                saved += 1
        logger.info(f"Saved {saved}/{len(labels)} labels")

    if not trades:
        logger.info(f"{symbol}: no trades triggered in {YEARS} years of data ({time.time() - t_start:.1f}s total)")
        return {}

    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    pnls    = [t["pnl"] for t in trades]
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pnl_arr = np.array(pnls)
    sharpe  = (pnl_arr.mean() / pnl_arr.std() * (252 ** 0.5)) if pnl_arr.std() > 0 else 0
    days    = YEARS * 365

    result = {
        "symbol":          symbol,
        "sizing_mode":     SIZING_MODE,
        "signal_tf":       signal_tf,
        "trend_tf":        trend_tf,
        "total_trades":    total,
        "longs":           sum(1 for t in trades if t["side"] == "LONG"),
        "shorts":          sum(1 for t in trades if t["side"] == "SHORT"),
        "wins":            wins,
        "losses":          total - wins,
        "win_rate":        round(wins / total, 4),
        "total_pnl":       round(sum(pnls), 4),
        "profit_factor":   round(gross_p / gross_l, 4) if gross_l else 0,
        "sharpe":          round(sharpe, 4),
        "final_balance":   round(balance, 2),
        "return_pct":      round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
        "best_trade":      round(max(pnls), 4),
        "worst_trade":     round(min(pnls), 4),
        "trades_per_day":  round(total / days, 2),
        "labels_saved":    len(labels) if save_labels else 0,
        "runtime_sec":     round(time.time() - t_start, 1),
    }
    return result


def _print_result(r: dict, label: str) -> None:
    logger.info(
        f"{label} | {r['symbol']} | "
        f"trades={r['total_trades']} (L{r['longs']}/S{r['shorts']}, {r['trades_per_day']}/day) "
        f"win={r['win_rate']:.0%} pf={r['profit_factor']:.2f} "
        f"pnl={r['total_pnl']:+.2f} ret={r['return_pct']:+.1f}% "
        f"sharpe={r['sharpe']:.2f} | {r['runtime_sec']}s"
    )


async def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"OPTIMIZED BACKTEST — {YEARS} years, max {MAX_OPEN_TRADES} concurrent trades")
    logger.info("=" * 60)

    all_results = []

    for combo in TIMEFRAME_COMBOS:
        label, signal_tf, trend_tf = combo["label"], combo["signal"], combo["trend"]
        combo_results = []

        for symbol in TRADING.symbols:
            r = await backtest_combo(symbol, signal_tf, trend_tf)
            if r:
                r["label"] = label
                combo_results.append(r)
                _print_result(r, label)

        if combo_results:
            score = sum(
                r.get("profit_factor", 0) * 0.4 +
                r.get("win_rate", 0) * 0.3 +
                r.get("sharpe", 0) * 0.2 +
                r.get("return_pct", -100) / 100 * 0.1
                for r in combo_results
            ) / len(combo_results)
            all_results.append({"combo": combo, "results": combo_results, "score": score})

    if not all_results:
        logger.error("No results — run scripts/seed_history.py first (and make sure it covers enough history)")
        return

    winner     = all_results[0]
    best_combo = winner["combo"]

    logger.info("\n" + "=" * 60)
    logger.info(f"WINNER: {best_combo['label']}")
    logger.info("=" * 60)

    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        set_key(env_path, "SIGNAL_TIMEFRAME", best_combo["signal"])
        set_key(env_path, "TREND_TIMEFRAME", best_combo["trend"])
        logger.info(".env updated")

    logger.info("\nSaving training labels from best combo...")
    for symbol in TRADING.symbols:
        await backtest_combo(symbol, best_combo["signal"], best_combo["trend"], save_labels=True)

    logger.info("\n" + "=" * 60)
    logger.info(f"Backtest complete — training labels saved | total runtime: {time.time() - t0:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())