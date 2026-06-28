from loguru import logger
from app.database.client import get_client


async def analyse_closed_trade(trade: dict, context: dict | None) -> dict:
    """
    Analyses a single closed trade and returns a structured
    analysis dict that feeds into the tuner and weekly report.
    """
    pnl      = trade.get("profit_loss", 0.0) or 0.0
    pnl_pct  = trade.get("profit_pct", 0.0) or 0.0
    reason   = trade.get("exit_reason", "UNKNOWN")
    side     = trade.get("side", "")
    symbol   = trade.get("symbol", "")

    won      = pnl > 0
    result   = "WIN" if won else "LOSS"

    analysis = {
        "trade_id":   trade["id"],
        "symbol":     symbol,
        "side":       side,
        "result":     result,
        "pnl":        pnl,
        "pnl_pct":    pnl_pct,
        "exit_reason": reason,
        "flags":      [],
    }

    if context is None:
        logger.warning(f"No context for trade {trade['id']} — limited analysis")
        return analysis

    rsi         = context.get("rsi_1h", 50.0)
    vol_ratio   = context.get("volume_ratio", 1.0)
    volatility  = context.get("volatility_pct", 1.5)
    trend_str   = context.get("trend_strength", 0.0)
    hour        = context.get("hour_of_day", 12)

    # ── Flag potential issues ──────────────────────────────────────
    if not won and rsi > 58:
        analysis["flags"].append("high_rsi_loss")
    if not won and vol_ratio < 0.8:
        analysis["flags"].append("low_volume_loss")
    if not won and volatility > 3.0:
        analysis["flags"].append("high_volatility_loss")
    if not won and abs(trend_str) < 0.1:
        analysis["flags"].append("weak_trend_loss")
    if not won and reason == "SL_HIT" and volatility > 2.5:
        analysis["flags"].append("sl_hit_high_vol")
    if won and rsi < 50:
        analysis["flags"].append("low_rsi_win")
    if hour in [22, 23, 0, 1, 2]:
        analysis["flags"].append("off_peak_hours")

    analysis.update({
        "rsi":        rsi,
        "vol_ratio":  vol_ratio,
        "volatility": volatility,
        "trend_str":  trend_str,
        "hour":       hour,
    })

    logger.debug(
        f"Trade analysed: {symbol} {side} {result} "
        f"rsi={rsi:.1f} flags={analysis['flags']}"
    )
    return analysis


async def get_recent_win_rate(last_n: int = 10) -> float:
    """Returns win rate over the last N closed trades."""
    try:
        client = get_client()
        result = (
            client.table("trades")
            .select("profit_loss")
            .eq("status", "CLOSED")
            .order("closed_at", desc=True)
            .limit(last_n)
            .execute()
        )
        data = result.data or []
        if not data:
            return 0.5
        wins = sum(1 for t in data if (t.get("profit_loss") or 0) > 0)
        return wins / len(data)
    except Exception as e:
        logger.error(f"Failed to calculate recent win rate: {e}")
        return 0.5


async def get_trade_context_by_id(trade_id: str) -> dict | None:
    try:
        client = get_client()
        result = (
            client.table("trade_context")
            .select("*")
            .eq("trade_id", trade_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Failed to fetch trade context for {trade_id}: {e}")
        return None


async def flag_summary(last_n: int = 50) -> dict:
    """
    Counts how often each flag appears across recent trades.
    Used by the tuner to decide what to adjust.
    """
    try:
        client = get_client()
        trades = (
            client.table("trades")
            .select("id,profit_loss,side,symbol,exit_reason")
            .eq("status", "CLOSED")
            .order("closed_at", desc=True)
            .limit(last_n)
            .execute()
        ).data or []

        counts: dict[str, int] = {}
        for trade in trades:
            ctx  = await get_trade_context_by_id(trade["id"])
            anlz = await analyse_closed_trade(trade, ctx)
            for flag in anlz["flags"]:
                counts[flag] = counts.get(flag, 0) + 1

        logger.info(f"Flag summary over last {last_n} trades: {counts}")
        return counts
    except Exception as e:
        logger.error(f"Failed to generate flag summary: {e}")
        return {}