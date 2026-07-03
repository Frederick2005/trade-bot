"""
Temporary diagnostic script — run this to see exactly which
condition is blocking every trade.

Usage:
    python scripts/debug_strategy.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import TRADING
from app.database.market import get_candles
from app.market.candles import get_indicators, seed

import pandas as pd


def _get_trend_window(df_trend: pd.DataFrame, up_to_time) -> pd.DataFrame:
    return df_trend[df_trend["open_time"] <= up_to_time].tail(11000)


async def main():
    symbol = TRADING.symbols[0]
    df_signal = await get_candles(symbol, "1h", limit=43800)
    df_trend  = await get_candles(symbol, "4h", limit=11000)

    print(f"1h candles: {len(df_signal)}")
    print(f"4h candles: {len(df_trend)}")

    seed(symbol, "1h", df_signal)
    seed(symbol, "4h", df_trend)

    counters = {
        "total_checked":     0,
        "missing_data":      0,
        "trend_4h_fail":     0,
        "trend_1h_fail":     0,
        "pullback_fail":     0,
        "rsi_fail":          0,
        "candle_fail":       0,
        "all_passed":        0,
    }

    min_candles = 200
    for i in range(min_candles, len(df_signal), 5):  # sample every 5th to go faster
        window_signal = df_signal.iloc[:i+1]
        window_trend  = _get_trend_window(df_trend, df_signal.iloc[i]["open_time"])
        seed(symbol, "1h", window_signal)
        seed(symbol, "4h", window_trend)

        ind_1h = get_indicators(symbol, "1h")
        ind_4h = get_indicators(symbol, "4h")

        counters["total_checked"] += 1

        if ind_1h is None or ind_4h is None:
            counters["missing_data"] += 1
            continue

        ema50_1h  = ind_1h.get("ema50")
        ema200_1h = ind_1h.get("ema200")
        ema50_4h  = ind_4h.get("ema50")
        ema200_4h = ind_4h.get("ema200")
        rsi_1h    = ind_1h.get("rsi")
        price_vs_ema50 = ind_1h.get("price_vs_ema50", 0.0)
        bullish   = ind_1h.get("is_bullish_candle", False)
        body_pct  = ind_1h.get("candle_body_pct", 0.0)

        if not (ema50_4h and ema200_4h and ema50_4h > ema200_4h):
            counters["trend_4h_fail"] += 1
            continue

        if not (ema50_1h and ema200_1h and ema50_1h > ema200_1h):
            counters["trend_1h_fail"] += 1
            continue

        if not (-2.5 <= price_vs_ema50 <= 2.5):
            counters["pullback_fail"] += 1
            continue

        if not (40.0 <= rsi_1h <= 60.0):
            counters["rsi_fail"] += 1
            continue

        if not bullish or body_pct < 0.30:
            counters["candle_fail"] += 1
            continue

        counters["all_passed"] += 1

    print("\n--- DIAGNOSTIC RESULTS ---")
    for k, v in counters.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    asyncio.run(main())