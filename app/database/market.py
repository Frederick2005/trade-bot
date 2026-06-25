from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from loguru import logger
from app.database.client import get_client


async def save_candles(symbol: str, timeframe: str, candles: list[dict]) -> bool:
    if not candles:
        return True
    try:
        client = get_client()
        rows = [
            {
                "symbol":    symbol,
                "timeframe": timeframe,
                "open_time": c["open_time"],
                "open":      float(c["open"]),
                "high":      float(c["high"]),
                "low":       float(c["low"]),
                "close":     float(c["close"]),
                "volume":    float(c["volume"]),
            }
            for c in candles
        ]
        # upsert — safe to re-run, won't duplicate
        client.table("market_data").upsert(
            rows, on_conflict="symbol,timeframe,open_time"
        ).execute()
        logger.debug(f"Saved {len(rows)} candles | {symbol} {timeframe}")
        return True
    except Exception as e:
        logger.error(f"Failed to save candles for {symbol} {timeframe}: {e}")
        return False


async def get_candles(
    symbol: str,
    timeframe: str,
    limit: int = 300,
    since: Optional[datetime] = None,
) -> pd.DataFrame:
    try:
        client = get_client()
        query = (
            client.table("market_data")
            .select("open_time,open,high,low,close,volume")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .order("open_time", desc=True)
            .limit(limit)
        )
        if since:
            query = query.gte("open_time", since.isoformat())

        result = query.execute()
        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch candles for {symbol} {timeframe}: {e}")
        return pd.DataFrame()


async def get_latest_candle_time(symbol: str, timeframe: str) -> Optional[datetime]:
    try:
        client = get_client()
        result = (
            client.table("market_data")
            .select("open_time")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .order("open_time", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            return datetime.fromisoformat(result.data[0]["open_time"])
        return None
    except Exception as e:
        logger.error(f"Failed to fetch latest candle time: {e}")
        return None


async def count_candles(symbol: str, timeframe: str) -> int:
    try:
        client = get_client()
        result = (
            client.table("market_data")
            .select("id", count="exact")
            .eq("symbol", symbol)
            .eq("timeframe", timeframe)
            .execute()
        )
        return result.count or 0
    except Exception as e:
        logger.error(f"Failed to count candles: {e}")
        return 0