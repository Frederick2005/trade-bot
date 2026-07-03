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
    """
    Fetch candles from Supabase, paginating past the default
    PostgREST 1000-row cap so `limit` is honoured exactly.
    """
    try:
        client = get_client()
        page_size = 1000
        all_rows: list[dict] = []
        offset = 0

        while len(all_rows) < limit:
            remaining = limit - len(all_rows)
            fetch_size = min(page_size, remaining)

            query = (
                client.table("market_data")
                .select("open_time,open,high,low,close,volume")
                .eq("symbol", symbol)
                .eq("timeframe", timeframe)
                .order("open_time", desc=True)
                .range(offset, offset + fetch_size - 1)
            )
            if since:
                query = query.gte("open_time", since.isoformat())

            result = query.execute()
            batch = result.data or []

            if not batch:
                break

            all_rows.extend(batch)
            offset += fetch_size

            if len(batch) < fetch_size:
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
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