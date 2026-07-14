"""
app/learning/analyser.py
Analyses every closed trade and auto-classifies errors.

WHY ERROR CLASSIFICATION MAKES THE BOT SMARTER:
─────────────────────────────────────────────────
Instead of just knowing "this trade lost", the bot learns WHY.
Pattern of errors drives specific fixes:

  HIGH_VOLATILITY_LOSS  → Raise ATR multiplier (wider stop)
  LOW_VOLUME_LOSS       → Raise MIN_VOLUME_RATIO threshold
  STOP_TOO_TIGHT        → Increase ATR multiplier
  FALSE_BREAKOUT        → Add volume confirmation requirement
  AGAINST_SESSION       → Session filter already blocks this
  RANGING_MARKET        → Add ADX filter (already added)
  RSI_EXTREME_LOSS      → Tighten RSI zone
  WEAK_TREND            → Raise minimum EMA gap requirement

Over 100+ trades this builds a clear picture of what
to fix and in what priority order.
"""
from loguru import logger
from app.database.client import get_client
from app.database.context import save_error_analysis


def _classify_error(trade: dict, context: dict) -> tuple[str, float, str]:
    """
    Classify why a trade lost. Returns (error_type, confidence, description).
    Called only on losing trades.
    """
    rsi        = float(context.get("rsi_1h", 50) or 50)
    vol        = float(context.get("volume_ratio", 1) or 1)
    vol_pct    = float(context.get("volatility_pct", 1.5) or 1.5)
    trend_str  = abs(float(context.get("trend_strength", 0) or 0))
    adx        = float(context.get("adx", 20) or 20)
    session    = context.get("session_name", "UNKNOWN")
    regime     = context.get("market_regime", "NEUTRAL")
    exit_r     = trade.get("exit_reason", "")
    holding    = float(trade.get("holding_time_minutes", 0) or 0)
    side       = trade.get("side", "LONG")

    # ── Rule-based error classification ──────────────────────────
    if vol_pct > 3.5 and exit_r == "SL_HIT":
        return ("HIGH_VOLATILITY_LOSS", 0.85,
                f"Volatility {vol_pct:.2f}% was extreme — stop blown by spike")

    if vol < 0.7:
        return ("LOW_VOLUME_LOSS", 0.80,
                f"Volume ratio {vol:.2f} too low — no real momentum behind move")

    if trend_str < 0.05 or adx < 15:
        return ("RANGING_MARKET", 0.78,
                f"ADX={adx:.1f} trend_strength={trend_str:.3f} — market was ranging")

    if exit_r == "SL_HIT" and holding < 30:
        return ("STOP_TOO_TIGHT", 0.75,
                f"Trade stopped out in {holding:.0f}min — ATR multiplier too small")

    if session in ["ASIAN", "ASIAN_OPEN"] and exit_r == "SL_HIT":
        return ("AGAINST_SESSION", 0.82,
                f"Loss in Asian session ({session}) — low liquidity stop hunt")

    if regime == "RANGING":
        return ("RANGING_MARKET", 0.80,
                "Market regime was RANGING — EMA strategy underperforms in ranges")

    if rsi > 60 and side == "LONG":
        return ("RSI_OVERBOUGHT_ENTRY", 0.72,
                f"RSI={rsi:.1f} was high for long entry — momentum was exhausted")

    if rsi < 40 and side == "SHORT":
        return ("RSI_OVERSOLD_ENTRY", 0.72,
                f"RSI={rsi:.1f} was low for short entry — too extended")

    if exit_r == "SL_HIT" and holding > 180:
        return ("TREND_REVERSAL", 0.68,
                f"Held {holding:.0f}min then reversed — trend changed direction")

    if vol < 0.9:
        return ("LOW_VOLUME_LOSS", 0.65,
                f"Below average volume {vol:.2f} — weak participation")

    return ("UNKNOWN", 0.40, "Could not classify — needs manual review")


async def analyse_closed_trade(trade: dict, context: dict | None) -> dict:
    pnl    = trade.get("profit_loss", 0.0) or 0.0
    pnl_pct = trade.get("profit_pct", 0.0) or 0.0
    reason = trade.get("exit_reason", "UNKNOWN")
    side   = trade.get("side", "")
    symbol = trade.get("symbol", "")
    won    = pnl > 0

    analysis = {
        "trade_id":   trade["id"],
        "symbol":     symbol,
        "side":       side,
        "result":     "WIN" if won else "LOSS",
        "pnl":        pnl,
        "pnl_pct":    pnl_pct,
        "exit_reason": reason,
        "flags":      [],
        "error_type": None,
    }

    if context is None:
        logger.warning(f"No context for trade {trade['id']} — limited analysis")
        return analysis

    rsi        = float(context.get("rsi_1h", 50) or 50)
    vol_ratio  = float(context.get("volume_ratio", 1) or 1)
    volatility = float(context.get("volatility_pct", 1.5) or 1.5)
    hour       = int(context.get("hour_of_day", 12) or 12)
    session    = context.get("session_name", "UNKNOWN")

    # ── Flag patterns ─────────────────────────────────────────────
    if not won and rsi > 58:   analysis["flags"].append("high_rsi_loss")
    if not won and vol_ratio < 0.8: analysis["flags"].append("low_volume_loss")
    if not won and volatility > 3.0: analysis["flags"].append("high_volatility_loss")
    if not won and session in ["ASIAN", "ASIAN_OPEN"]: analysis["flags"].append("off_session")
    if hour in range(0, 7):    analysis["flags"].append("asian_hours")
    if won and rsi < 48:       analysis["flags"].append("low_rsi_win")

    analysis.update({
        "rsi":       rsi,
        "vol_ratio": vol_ratio,
        "volatility":volatility,
        "hour":      hour,
        "session":   session,
    })

    # ── Auto-classify errors on losing trades ─────────────────────
    if not won:
        error_type, confidence, description = _classify_error(trade, context)
        analysis["error_type"] = error_type

        await save_error_analysis(
            trade_id=trade["id"],
            symbol=symbol,
            error_type=error_type,
            confidence=confidence,
            description=description,
            context=context,
        )
        logger.debug(f"Error classified: {error_type} ({confidence:.0%}) — {description}")

    logger.debug(
        f"Trade analysed: {symbol} {side} {'WIN' if won else 'LOSS'} "
        f"RSI={rsi:.1f} session={session} flags={analysis['flags']}"
    )
    return analysis


async def get_recent_win_rate(last_n: int = 10) -> float:
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


async def get_error_summary(last_n: int = 100) -> dict:
    """
    Returns count of each error type over last N losses.
    Used by tuner to decide what parameters to adjust.
    """
    try:
        client = get_client()
        result = (
            client.table("error_analysis")
            .select("error_type,confidence")
            .order("created_at", desc=True)
            .limit(last_n)
            .execute()
        )
        data   = result.data or []
        counts: dict[str, int] = {}
        for row in data:
            et = row["error_type"]
            counts[et] = counts.get(et, 0) + 1

        logger.info(f"Error summary (last {last_n}): {counts}")
        return counts
    except Exception as e:
        logger.error(f"Failed to get error summary: {e}")
        return {}