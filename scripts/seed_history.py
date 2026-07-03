"""
Download 2 years of 15m and 1h candle data from Binance.
Usage: python scripts/seed_history.py
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from loguru import logger
from dotenv import load_dotenv
from app.config import TRADING
from app.database.market import save_candles, count_candles

load_dotenv()

BINANCE_BASE  = "https://api.binance.com"
BATCH_SIZE    = 1000
TIMEFRAMES    = ["15m", "1h"]
HISTORY_DAYS  = 730  # 2 years


def binance_tf_to_ms(timeframe: str) -> int:
    return {
        "1m":  60_000,
        "5m":  300_000,
        "15m": 900_000,
        "1h":  3_600_000,
        "4h":  14_400_000,
        "1d":  86_400_000,
    }[timeframe]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    resp = requests.get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval,
                "startTime": start_ms, "endTime": end_ms, "limit": BATCH_SIZE},
        timeout=10,
    )
    resp.raise_for_status()
    return [{"open_time": r[0], "open": r[1], "high": r[2],
             "low": r[3], "close": r[4], "volume": r[5]} for r in resp.json()]


def ms_to_iso(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


async def download_symbol_timeframe(symbol: str, timeframe: str) -> int:
    tf_ms    = binance_tf_to_ms(timeframe)
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - (HISTORY_DAYS * 24 * 60 * 60 * 1000)
    cursor   = start_ms
    total    = 0
    batch    = 0

    logger.info(f"Downloading {symbol} {timeframe} — {HISTORY_DAYS} days...")

    while cursor < now_ms:
        batch_end = min(cursor + BATCH_SIZE * tf_ms, now_ms)
        try:
            candles = fetch_klines(symbol, timeframe, cursor, batch_end)
        except Exception as e:
            logger.error(f"Error: {e}")
            break

        if not candles:
            break

        rows = [{"open_time": ms_to_iso(int(c["open_time"])),
                 "open": c["open"], "high": c["high"],
                 "low": c["low"], "close": c["close"], "volume": c["volume"]}
                for c in candles]

        if await save_candles(symbol, timeframe, rows):
            total += len(rows)
            batch += 1
            if batch % 10 == 0:
                logger.info(f"  {symbol} {timeframe}: {total} candles saved...")

        cursor = int(candles[-1]["open_time"]) + tf_ms
        time.sleep(0.1)

    logger.info(f"  Done {symbol} {timeframe}: {total} candles")
    return total


async def main() -> None:
    logger.info("=" * 60)
    logger.info(f"Downloading {HISTORY_DAYS} days of 15m + 1h data")
    logger.info("=" * 60)

    for symbol in TRADING.symbols:
        for tf in TIMEFRAMES:
            existing = await count_candles(symbol, tf)
            logger.info(f"{symbol} {tf}: {existing} candles already in DB")
            saved = await download_symbol_timeframe(symbol, tf)
            final = await count_candles(symbol, tf)
            logger.info(f"✓ {symbol} {tf}: {final} total (+{saved} new)")

    logger.info("Download complete — run python scripts/backtest.py")


if __name__ == "__main__":
    asyncio.run(main())