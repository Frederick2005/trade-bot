from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from app.database.client import get_client


async def save_trade(
    symbol: str,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    lot_size: float,
    strategy_version: str,
    account_balance: float,
    order_id: Optional[str] = None,
) -> Optional[str]:
    try:
        client = get_client()
        result = client.table("trades").insert({
            "symbol":           symbol,
            "side":             side,
            "entry_price":      entry_price,
            "stop_loss":        stop_loss,
            "take_profit":      take_profit,
            "lot_size":         lot_size,
            "strategy_version": strategy_version,
            "account_balance":  account_balance,
            "order_id":         order_id,
            "status":           "OPEN",
            "opened_at":        datetime.now(timezone.utc).isoformat(),
        }).execute()
        trade_id = result.data[0]["id"]
        logger.info(f"Trade saved: {symbol} {side} @ {entry_price} | id={trade_id}")
        return trade_id
    except Exception as e:
        logger.error(f"Failed to save trade: {e}")
        return None


async def close_trade(
    trade_id: str,
    exit_price: float,
    profit_loss: float,
    profit_pct: float,
    exit_reason: str,
) -> bool:
    try:
        client = get_client()
        client.table("trades").update({
            "exit_price":  exit_price,
            "profit_loss": profit_loss,
            "profit_pct":  profit_pct,
            "exit_reason": exit_reason,
            "status":      "CLOSED",
            "closed_at":   datetime.now(timezone.utc).isoformat(),
        }).eq("id", trade_id).execute()
        logger.info(f"Trade closed: id={trade_id} | pnl={profit_loss:+.4f} | reason={exit_reason}")
        return True
    except Exception as e:
        logger.error(f"Failed to close trade {trade_id}: {e}")
        return False


async def get_open_trades() -> list[dict]:
    try:
        client = get_client()
        result = client.table("trades").select("*").eq("status", "OPEN").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch open trades: {e}")
        return []


async def get_closed_trades(limit: int = 100) -> list[dict]:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("*")
            .eq("status", "CLOSED")
            .order("closed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch closed trades: {e}")
        return []


async def get_trades_since(since: datetime) -> list[dict]:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("*")
            .eq("status", "CLOSED")
            .gte("closed_at", since.isoformat())
            .order("closed_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch trades since {since}: {e}")
        return []


async def count_closed_trades() -> int:
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("id", count="exact")
            .eq("status", "CLOSED")
            .execute()
        )
        return result.count or 0
    except Exception as e:
        logger.error(f"Failed to count closed trades: {e}")
        return 0