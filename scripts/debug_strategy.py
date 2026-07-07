"""
Temporary diagnostic script — run this to see exactly which
condition is blocking every trade.

Usage:
    python scripts/debug_strategy.py

This now uses app.strategy.signal_logic (the same thresholds/logic the live
bot and backtest actually use) instead of its own separately-hardcoded copy
of the entry rules, and reads the configured signal/trend timeframes from
app.config.TIMEFRAMES instead of hardcoded "1h"/"4h" — previously this script
would silently check different, looser thresholds (RSI 40-60, pullback
±2.5%, body>=0.30) than what actually gates a trade, and would break/mislead
if you ever changed TIMEFRAMES away from 1h/4h (e.g. to 15m/1h, which
scripts/backtest.py can auto-set in your .env).
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import TRADING, TIMEFRAMES
from app.database.market import get_candles
from app.market.candles import get_indicators, seed
from app.strategy import signal_logic

import pandas as pd


def _get_trend_window(df_trend: pd.DataFrame, up_to_time) -> pd.DataFrame:
    return df_trend[df_trend["open_time"] <= up_to_time].tail(11000)


async def main():
    symbol    = TRADING.symbols[0]
    signal_tf = TIMEFRAMES.signal
    trend_tf  = TIMEFRAMES.trend

    df_signal = await get_candles(symbol, signal_tf, limit=43800)
    df_trend  = await get_candles(symbol, trend_tf, limit=11000)

    print(f"{signal_tf} candles: {len(df_signal)}")
    print(f"{trend_tf} candles: {len(df_trend)}")

    seed(symbol, signal_tf, df_signal)
    seed(symbol, trend_tf, df_trend)

    counters = {
        "total_checked":  0,
        "missing_data":   0,
        "rejected":       0,
        "long_signals":   0,
        "short_signals":  0,
    }
    reject_reasons = {
        "trend_mismatch": 0,  # covers 4h/1h-trend-direction and candle-direction gates together
        "rsi_out_of_band": 0,
        "pullback_out_of_band": 0,
        "body_too_small": 0,
        "missing_fields": 0,
    }

    min_candles = 200
    for i in range(min_candles, len(df_signal), 5):  # sample every 5th to go faster
        window_signal = df_signal.iloc[:i + 1]
        window_trend  = _get_trend_window(df_trend, df_signal.iloc[i]["open_time"])
        seed(symbol, signal_tf, window_signal)
        seed(symbol, trend_tf, window_trend)

        ind_signal = get_indicators(symbol, signal_tf)
        ind_trend  = get_indicators(symbol, trend_tf)

        counters["total_checked"] += 1

        if ind_signal is None or ind_trend is None:
            counters["missing_data"] += 1
            continue

        sig = signal_logic.evaluate_entry(ind_signal, ind_trend)

        # For diagnostics, re-check the individual gates so you can see WHY
        # a candle was rejected, even though the pass/fail decision itself
        # comes from the single shared evaluate_entry() function above.
        rsi_s = ind_signal.get("rsi")
        price_vs_ema50 = ind_signal.get("price_vs_ema50", 0.0)
        body_pct = ind_signal.get("candle_body_pct", 0.0)

        if sig is None:
            counters["rejected"] += 1
            if rsi_s is None or price_vs_ema50 is None:
                reject_reasons["missing_fields"] += 1
            elif not (signal_logic.RSI_LOW <= rsi_s <= signal_logic.RSI_HIGH):
                reject_reasons["rsi_out_of_band"] += 1
            elif not (-signal_logic.PULLBACK_BAND_PCT <= price_vs_ema50 <= signal_logic.PULLBACK_BAND_PCT):
                reject_reasons["pullback_out_of_band"] += 1
            elif body_pct < signal_logic.MIN_BODY_PCT:
                reject_reasons["body_too_small"] += 1
            else:
                reject_reasons["trend_mismatch"] += 1
        elif sig["side"] == "LONG":
            counters["long_signals"] += 1
        else:
            counters["short_signals"] += 1

    print("\n--- DIAGNOSTIC RESULTS ---")
    for k, v in counters.items():
        print(f"{k:20s}: {v}")
    print("\n--- REJECTION BREAKDOWN (sampled candles that failed) ---")
    for k, v in reject_reasons.items():
        print(f"{k:24s}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
