from loguru import logger
from app.database.client import get_client


async def win_rate_by_rsi_bucket() -> list[dict]:
    """
    Groups trades into RSI buckets of 5 and calculates win rate per bucket.
    Reveals which RSI zones actually produce winning trades.
    """
    try:
        client = get_client()
        result = client.rpc("win_rate_by_rsi_bucket").execute()
        return result.data or []
    except Exception:
        # Fallback: manual calculation
        return await _win_rate_by_rsi_manual()


async def _win_rate_by_rsi_manual() -> list[dict]:
    try:
        client  = get_client()
        ctx     = client.table("trade_context").select("trade_id,rsi_1h").execute()
        trades  = client.table("trades").select("id,profit_loss").eq("status","CLOSED").execute()

        trade_map = {t["id"]: t["profit_loss"] for t in (trades.data or [])}
        buckets: dict[int, list[float]] = {}

        for row in (ctx.data or []):
            rsi = row.get("rsi_1h")
            pnl = trade_map.get(row["trade_id"])
            if rsi is None or pnl is None:
                continue
            bucket = int(rsi // 5) * 5
            buckets.setdefault(bucket, []).append(float(pnl))

        results = []
        for bucket, pnls in sorted(buckets.items()):
            wins = sum(1 for p in pnls if p > 0)
            results.append({
                "rsi_bucket": bucket,
                "total":      len(pnls),
                "wins":       wins,
                "win_rate":   round(wins / len(pnls), 4),
                "avg_pnl":    round(sum(pnls) / len(pnls), 4),
            })
        return results
    except Exception as e:
        logger.error(f"RSI bucket query failed: {e}")
        return []


async def win_rate_by_hour() -> list[dict]:
    """Which hours of day (UTC) are most profitable?"""
    try:
        client  = get_client()
        ctx     = client.table("trade_context").select("trade_id,hour_of_day").execute()
        trades  = client.table("trades").select("id,profit_loss").eq("status","CLOSED").execute()

        trade_map = {t["id"]: t["profit_loss"] for t in (trades.data or [])}
        hours: dict[int, list[float]] = {}

        for row in (ctx.data or []):
            hour = row.get("hour_of_day")
            pnl  = trade_map.get(row["trade_id"])
            if hour is None or pnl is None:
                continue
            hours.setdefault(int(hour), []).append(float(pnl))

        results = []
        for hour, pnls in sorted(hours.items()):
            wins = sum(1 for p in pnls if p > 0)
            results.append({
                "hour":     hour,
                "total":    len(pnls),
                "wins":     wins,
                "win_rate": round(wins / len(pnls), 4),
                "total_pnl": round(sum(pnls), 4),
            })
        return sorted(results, key=lambda x: x["total_pnl"], reverse=True)
    except Exception as e:
        logger.error(f"Hour win rate query failed: {e}")
        return []


async def win_rate_by_volatility() -> list[dict]:
    """Does high volatility help or hurt us?"""
    try:
        client  = get_client()
        ctx     = client.table("trade_context").select("trade_id,volatility_pct").execute()
        trades  = client.table("trades").select("id,profit_loss").eq("status","CLOSED").execute()

        trade_map = {t["id"]: t["profit_loss"] for t in (trades.data or [])}
        buckets: dict[str, list[float]] = {"low": [], "medium": [], "high": []}

        for row in (ctx.data or []):
            vol = row.get("volatility_pct")
            pnl = trade_map.get(row["trade_id"])
            if vol is None or pnl is None:
                continue
            vol = float(vol)
            if vol < 1.5:
                buckets["low"].append(float(pnl))
            elif vol < 3.0:
                buckets["medium"].append(float(pnl))
            else:
                buckets["high"].append(float(pnl))

        results = []
        for regime, pnls in buckets.items():
            if not pnls:
                continue
            wins = sum(1 for p in pnls if p > 0)
            results.append({
                "regime":   regime,
                "total":    len(pnls),
                "wins":     wins,
                "win_rate": round(wins / len(pnls), 4),
                "avg_pnl":  round(sum(pnls) / len(pnls), 4),
            })
        return results
    except Exception as e:
        logger.error(f"Volatility query failed: {e}")
        return []


async def overall_performance(days: int = 30) -> dict:
    """Returns overall performance stats for the last N days."""
    try:
        from datetime import datetime, timezone, timedelta
        since   = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        client  = get_client()
        result  = (
            client.table("trades")
            .select("profit_loss,profit_pct,exit_reason")
            .eq("status", "CLOSED")
            .gte("closed_at", since)
            .execute()
        )
        data = result.data or []
        if not data:
            return {}

        pnls    = [float(t["profit_loss"] or 0) for t in data]
        total   = len(pnls)
        wins    = sum(1 for p in pnls if p > 0)
        losses  = total - wins
        gross_p = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))

        return {
            "period_days":    days,
            "total_trades":   total,
            "wins":           wins,
            "losses":         losses,
            "win_rate":       round(wins / total, 4) if total else 0,
            "total_pnl":      round(sum(pnls), 4),
            "profit_factor":  round(gross_p / gross_l, 4) if gross_l else 0,
            "avg_win":        round(gross_p / wins, 4) if wins else 0,
            "avg_loss":       round(gross_l / losses, 4) if losses else 0,
            "best_trade":     round(max(pnls), 4) if pnls else 0,
            "worst_trade":    round(min(pnls), 4) if pnls else 0,
        }
    except Exception as e:
        logger.error(f"Overall performance query failed: {e}")
        return {}