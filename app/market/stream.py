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
            "open_time": datetime.fromtimestamp(
                k["t"] / 1000, tz=timezone.utc
            ).isoformat(),
            "open":   k["o"],
            "high":   k["h"],
            "low":    k["l"],
            "close":  k["c"],
            "volume": k["v"],
        }
    except (KeyError, TypeError, ValueError):
        return None


async def _create_client() -> AsyncClient:
    """Create Binance client with explicit testnet URLs to avoid geo-blocking."""
    if BINANCE.testnet:
        client = await AsyncClient.create(
            api_key=BINANCE.api_key,
            api_secret=BINANCE.api_secret,
            testnet=True,
            tld="com",
        )
        # Override to testnet endpoints explicitly
        client.API_URL = "https://testnet.binancefuture.com/fapi/v1"
        client.STREAM_URL = "wss://stream.binancefuture.com/ws"
    else:
        client = await AsyncClient.create(
            api_key=BINANCE.api_key,
            api_secret=BINANCE.api_secret,
        )
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
            logger.warning(
                f"Stream {symbol} {timeframe} disconnected: {e} — "
                f"reconnecting in {delay}s"
            )
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
            tasks.append(
                asyncio.create_task(
                    _stream_with_reconnect(client, symbol, tf, on_candle_close)
                )
            )

    logger.info(
        f"Streaming {len(tasks)} feeds: "
        f"{TRADING.symbols} × [{TIMEFRAMES.signal}, {TIMEFRAMES.trend}]"
    )
    await asyncio.gather(*tasks)