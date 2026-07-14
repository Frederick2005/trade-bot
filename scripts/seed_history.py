"""
scripts/seed_history.py
Fetches 5 years of historical candle data from Binance.

WHY 5 YEARS MATTERS FOR THE AI:
─────────────────────────────────
5 years of data captures multiple market cycles:
  - 2020 COVID crash + recovery (extreme volatility)
  - 2021 bull run (strong trends)
  - 2022 bear market (sustained downtrends)
  - 2023 recovery (ranging + breakout)
  - 2024-25 current cycle

The AI needs to see all regimes to learn which
conditions are dangerous vs profitable. Training
on only 6 months of data = biased to one regime.

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

YEARS_BACK  = 5
BATCH_SIZE  = 1000
# Only seed timeframes the bot actually uses
TIMEFRAMES  = ["1h", "4h"]


async def seed_symbol(
    client: AsyncClient,
    symbol: str,
    timeframe: str,
) -> int:
    since    = datetime.now(timezone.utc) - timedelta(days=YEARS_BACK * 365)
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
            logger.error(f"API error {symbol} {timeframe}: {e}")
            await asyncio.sleep(5)
            continue

        if not raw:
            break

        candles = []
        for k in raw:
            candles.append({
                "open_time":    datetime.fromtimestamp(
                    k[0] / 1000, tz=timezone.utc
                ).isoformat(),
                "open":         k[1],
                "high":         k[2],
                "low":          k[3],
                "close":        k[4],
                "volume":       k[5],
                "quote_volume": k[7],   # Quote asset volume
                "num_trades":   k[8],   # Number of trades
                "buy_volume":   k[9],   # Taker buy base volume
            })

        await save_candles(symbol, timeframe, candles)
        total_saved += len(candles)

        last_ms = raw[-1][0]
        last_dt = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)
        logger.info(
            f"  {symbol} {timeframe}: {total_saved} saved | "
            f"up to {last_dt.date()}"
        )

        if len(raw) < BATCH_SIZE:
            break

        current_ms = last_ms + 1
        await asyncio.sleep(0.25)   # rate limit respect

    logger.info(f"Completed {symbol} {timeframe}: {total_saved} candles")
    return total_saved


async def main() -> None:
    logger.info("=" * 60)
    logger.info(f"Seeding {YEARS_BACK} years of historical data")
    logger.info(f"Symbols: {TRADING.symbols}")
    logger.info(f"Timeframes: {TIMEFRAMES}")
    logger.info("=" * 60)

    client = await AsyncClient.create(
        api_key=BINANCE.api_key,
        api_secret=BINANCE.api_secret,
        testnet=BINANCE.testnet,
    )

    total_all = 0
    for symbol in TRADING.symbols:
        for tf in TIMEFRAMES:
            count = await seed_symbol(client, symbol, tf)
            total_all += count
            await asyncio.sleep(1)   # pause between pairs

    await client.close_connection()

    logger.info("=" * 60)
    logger.info(f"Seeding complete — {total_all:,} total candles saved")
    logger.info("")
    logger.info("Expected candle counts (5 years):")
    logger.info("  BTCUSDT 1h  → ~43,800")
    logger.info("  BTCUSDT 4h  → ~10,950")
    logger.info("  ETHUSDT 1h  → ~43,800")
    logger.info("  ETHUSDT 4h  → ~10,950")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  python scripts/backtest.py")
    logger.info("  python main.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())