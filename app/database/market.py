"""
app/database/market.py
Saves and fetches candle data.

WHY EXTENDED CANDLE DATA MATTERS FOR AI:
  buy_volume/sell_volume → AI learns that candles where buyers dominate
    (buy_volume > 60% of total) have higher follow-through.
  num_trades → more trades = more market participants = stronger signal.
    AI learns to be cautious on low-trade-count candles.
  quote_volume → USDT amount traded, more stable than coin count.
    Funding rate/OI → futures-specific, tells AI about leverage buildup.
    High OI + high funding = crowded trade = danger of reversal.
"""
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
        rows = []
        for c in candles:
            row = {
                "symbol":    symbol,
                "timeframe": timeframe,
                "open_time": c["open_time"],
                "open":      float(c["open"]),
                "high":      float(c["high"]),
                "low":       float(c["low"]),
                "close":     float(c["close"]),
                "volume":    float(c["volume"]),
            }
            # Extended fields — only add if present
            if "quote_volume" in c:
                row["quote_volume"] = float(c["quote_volume"])
            if "num_trades" in c:
                row["num_trades"] = int(c["num_trades"])
            if "buy_volume" in c:
                row["buy_volume"]  = float(c["buy_volume"])
                row["sell_volume"] = float(c["volume"]) - float(c["buy_volume"])
            if "funding_rate" in c:
                row["funding_rate"] = float(c["funding_rate"])
            if "open_interest" in c:
                row["open_interest"] = float(c["open_interest"])
            if "mark_price" in c:
                row["mark_price"] = float(c["mark_price"])
            rows.append(row)

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
        all_data = []
        batch    = 1000
        offset   = 0

        while True:
            query = (
                client.table("market_data")
                .select("open_time,open,high,low,close,volume,buy_volume,sell_volume,num_trades,quote_volume")
                .eq("symbol", symbol)
                .eq("timeframe", timeframe)
                .order("open_time", desc=True)
            )
            if since:
                query = query.gte("open_time", since.isoformat())

            result = query.range(offset, offset + batch - 1).execute()
            batch_data = result.data or []
            all_data.extend(batch_data)

            if len(batch_data) < batch or len(all_data) >= limit:
                break
            offset += batch

        if not all_data:
            return pd.DataFrame()

        # Take most recent `limit` rows
        all_data = all_data[:limit]

        df = pd.DataFrame(all_data)
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col in ["buy_volume", "sell_volume", "num_trades", "quote_volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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