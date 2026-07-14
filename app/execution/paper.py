import uuid
from datetime import datetime, timezone
from loguru import logger
from app.state import state, OpenTrade


async def open_order(
    symbol: str,
    side: str,
    lot_size: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    strategy_version: str,
) -> dict | None:
    order_id  = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
    opened_at = datetime.now(timezone.utc).isoformat()

    trade = OpenTrade(
        trade_id=order_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        lot_size=lot_size,
        opened_at=opened_at,
        strategy_version=strategy_version,
        order_id=order_id,
    )
    state.open_trades[symbol] = trade

    logger.info(
        f"[PAPER] Order opened: {symbol} {side} "
        f"entry={entry_price:.2f} SL={stop_loss:.2f} "
        f"TP={take_profit:.2f} lot={lot_size:.6f}"
    )
    return {
        "order_id":    order_id,
        "symbol":      symbol,
        "side":        side,
        "entry_price": entry_price,
        "lot_size":    lot_size,
        "opened_at":   opened_at,
        "status":      "FILLED",
    }


async def close_order(
    symbol: str,
    exit_price: float,
    reason: str,
) -> dict | None:
    from app.database.trades import close_trade

    trade = state.open_trades.get(symbol)
    if not trade:
        logger.warning(f"[PAPER] No open trade found for {symbol}")
        return None

    if trade.side == "LONG":
        pnl = (exit_price - trade.entry_price) * trade.lot_size
    else:
        pnl = (trade.entry_price - exit_price) * trade.lot_size

    pnl_pct = pnl / state.balance if state.balance else 0.0

    # Was previously never called anywhere in the codebase — trades never
    # actually got marked CLOSED in the database, they stayed OPEN forever
    # (this is why the `trades` table only ever had rows stuck OPEN or
    # manually edited to CANCELLED). Fixed here since this is the one
    # place a paper trade's closure is actually determined.
    await close_trade(
        trade_id=trade.trade_id,
        exit_price=exit_price,
        profit_loss=pnl,
        profit_pct=pnl_pct,
        exit_reason=reason,
    )

    state.record_closed_trade(pnl)
    del state.open_trades[symbol]

    logger.info(
        f"[PAPER] Order closed: {symbol} "
        f"exit={exit_price:.2f} pnl={pnl:+.4f} ({pnl_pct:+.2%}) "
        f"reason={reason}"
    )
    return {
        "symbol":     symbol,
        "side":       trade.side,   # captured before state is cleared
        "trade_id":   trade.trade_id,
        "exit_price": exit_price,
        "pnl":        pnl,
        "pnl_pct":    pnl_pct,
        "reason":     reason,
    }


async def check_exits(current_prices: dict[str, float]) -> list[dict]:
    """
    Checks all open paper trades against current prices.
    Closes any that have hit TP or SL, or that have been open longer than
    DECISION_ENGINE.max_hold_hours (default 7 days) — prevents capital
    sitting tied up indefinitely in a setup whose original thesis has
    gone stale, mirroring the same check in scripts/backtest.py.
    """
    from datetime import datetime, timezone
    from app.config import DECISION_ENGINE

    closed = []
    for symbol, trade in list(state.open_trades.items()):
        price = current_prices.get(symbol)
        if price is None:
            continue

        if trade.side == "LONG":
            if price >= trade.take_profit:
                result = await close_order(symbol, trade.take_profit, "TP_HIT")
                if result:
                    closed.append(result)
                continue
            elif price <= trade.stop_loss:
                result = await close_order(symbol, trade.stop_loss, "SL_HIT")
                if result:
                    closed.append(result)
                continue
        else:  # SHORT
            if price <= trade.take_profit:
                result = await close_order(symbol, trade.take_profit, "TP_HIT")
                if result:
                    closed.append(result)
                continue
            elif price >= trade.stop_loss:
                result = await close_order(symbol, trade.stop_loss, "SL_HIT")
                if result:
                    closed.append(result)
                continue

        try:
            opened_at = datetime.fromisoformat(trade.opened_at)
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            hours_open = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
            if hours_open >= DECISION_ENGINE.max_hold_hours:
                result = await close_order(symbol, price, "MAX_HOLD_TIME")
                if result:
                    closed.append(result)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse opened_at for {symbol}: {e}")

    return closed