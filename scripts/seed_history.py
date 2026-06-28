"""
Fetches historical candle data from Binance and stores in Supabase.
Seeds ALL timeframes needed for backtesting and live trading.

Usage:
    python scripts/seed_history.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from binance import AsyncClient
from loguru import logger
from app.config import BINANCE, TRADING
from app.database.market import save_candles, count_candles

# Go back 18 months — gives enough candles for ALL timeframes including 4H and 1D
MONTHS_BACK = 18
BATCH_SIZE  = 1000

# Only 1H and 4H — testnet does not have enough history for 2H/1D
TIMEFRAMES_TO_SEED = ["1h", "4h"]


async def seed_symbol(client: AsyncClient, symbol: str, timeframe: str) -> int:
    since    = datetime.now(timezone.utc) - timedelta(days=MONTHS_BACK * 30)
    since_ms = int(since.timestamp() * 1000)

    existing = await count_candles(symbol, timeframe)
    logger.info(
        f"Seeding {symbol} {timeframe} | "
        f"existing={existing} | since={since.date()}"
    )

    total_saved = 0
    current_ms  = since_ms

    while True:
        try:
            raw = await client.get_klines(
                symbol=symbol,
                interval=timeframe,
                startTime=current_ms,
                limit=BATCH_SIZE,
            )
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            break

        if not raw:
            break

        candles = [
            {
                "open_time": datetime.fromtimestamp(
                    k[0] / 1000, tz=timezone.utc
                ).isoformat(),
                "open":   k[1],
                "high":   k[2],
                "low":    k[3],
                "close":  k[4],
                "volume": k[5],
            }
            for k in raw
        ]

        await save_candles(symbol, timeframe, candles)
        total_saved += len(candles)

        last_time_ms = raw[-1][0]
        logger.info(
            f"  {symbol} {timeframe}: {total_saved} candles saved | "
            f"up to {datetime.fromtimestamp(last_time_ms/1000, tz=timezone.utc).date()}"
        )

        if len(raw) < BATCH_SIZE:
            break

        current_ms = last_time_ms + 1
        await asyncio.sleep(0.3)

    logger.info(f"Done: {symbol} {timeframe} — {total_saved} total candles")
    return total_saved


async def main() -> None:
    logger.info("=" * 55)
    logger.info("Seeding historical candle data — 18 months")
    logger.info("=" * 55)

    client = await AsyncClient.create(
        api_key=BINANCE.api_key,
        api_secret=BINANCE.api_secret,
        testnet=BINANCE.testnet,
    )

    total = 0
    for symbol in TRADING.symbols:
        for tf in TIMEFRAMES_TO_SEED:
            count = await seed_symbol(client, symbol, tf)
            total += count
            await asyncio.sleep(0.5)   # be kind to the API

    await client.close_connection()

    logger.info("=" * 55)
    logger.info(f"Seeding complete — {total} candles saved total")
    logger.info("Candles per timeframe expected:")
    logger.info("  1h  → ~13,000  (18 months of hourly candles)")
    logger.info("  4h  → ~3,240   (18 months of 4-hourly candles)")
    logger.info("You can now run: python scripts/backtest.py")
    logger.info("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())