"""
app/market/stream.py
Binance WebSocket candle stream.

WHY WE COLLECT EXTRA KLINE FIELDS:
  buy_volume  → tells us who is driving the move. When buy volume
                dominates a candle, the move has real buyers behind it
                not just short covering. AI learns this distinction.
  num_trades  → high trade count = high participation = stronger signal
  quote_volume → USDT volume, more stable measure than coin volume
"""
import asyncio
from datetime import datetime, timezone
from typing import Callable, Awaitable
from binance import AsyncClient, BinanceSocketManager
from loguru import logger
from app.config import BINANCE, TRADING, TIMEFRAMES
from app.market import candles as candle_store
from app.database.market import save_candles
from app.state import state

OnCandleClose = Callable[[str, str, dict], Awaitable[None]]

_reconnect_delay = 5


def _parse_kline(msg: dict) -> dict | None:
    if not isinstance(msg, dict):
        return None
    if msg.get("e") != "kline" and "k" not in msg:
        return None
    try:
        k = msg["k"]
        if not k.get("x", False):
            return None
        return {
            "open_time":   datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc).isoformat(),
            "open":        k["o"],
            "high":        k["h"],
            "low":         k["l"],
            "close":       k["c"],
            "volume":      k["v"],
            # Extended fields — all available in the kline WebSocket
            "quote_volume": k.get("q", 0),   # Quote asset volume (USDT)
            "num_trades":   k.get("n", 0),   # Number of trades in candle
            "buy_volume":   k.get("V", 0),   # Taker buy base asset volume
        }
    except (KeyError, TypeError, ValueError):
        return None


async def _create_client() -> AsyncClient:
    client = AsyncClient(
        api_key=BINANCE.api_key,
        api_secret=BINANCE.api_secret,
        testnet=BINANCE.testnet,
    )
    await client.close_connection()
    client.session = client._init_session()
    logger.info(f"Binance client ready ({'TESTNET' if BINANCE.testnet else 'LIVE'})")
    return client


async def _stream_symbol(
    client: AsyncClient,
    symbol: str,
    timeframe: str,
    on_close: OnCandleClose,
) -> None:
    bm     = BinanceSocketManager(client)
    stream = bm.kline_socket(symbol, interval=timeframe)

    async with stream as s:
        logger.info(f"Stream open: {symbol} {timeframe}")
        state.binance_connected = True
        while True:
            msg    = await s.recv()
            candle = _parse_kline(msg)
            if candle is None:
                continue
            candle_store.update(symbol, timeframe, candle)
            asyncio.create_task(save_candles(symbol, timeframe, [candle]))
            await on_close(symbol, timeframe, candle)


async def _stream_with_reconnect(
    client: AsyncClient,
    symbol: str,
    timeframe: str,
    on_close: OnCandleClose,
) -> None:
    delay = _reconnect_delay
    while True:
        try:
            await _stream_symbol(client, symbol, timeframe, on_close)
        except Exception as e:
            state.binance_connected = False
            logger.warning(f"Stream {symbol} {timeframe} disconnected: {e} — reconnecting in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        else:
            delay = _reconnect_delay


async def start_streams(on_candle_close: OnCandleClose) -> None:
    logger.info("Starting Binance WebSocket streams...")
    client = await _create_client()

    tasks = []
    for symbol in TRADING.symbols:
        for tf in [TIMEFRAMES.signal, TIMEFRAMES.trend]:
            tasks.append(asyncio.create_task(
                _stream_with_reconnect(client, symbol, tf, on_candle_close)
            ))

    logger.info(f"Streaming {len(tasks)} feeds: {TRADING.symbols} × [{TIMEFRAMES.signal}, {TIMEFRAMES.trend}]")
    await asyncio.gather(*tasks)