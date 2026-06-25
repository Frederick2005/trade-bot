from binance import AsyncClient
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
)
from loguru import logger
from app.config import BINANCE, TRADING
from app.state import state, OpenTrade
from app.risk.limits import enforce_leverage
from datetime import datetime, timezone


_client: AsyncClient | None = None


async def get_client() -> AsyncClient:
    global _client
    if _client is None:
        _client = await AsyncClient.create(
            api_key=BINANCE.api_key,
            api_secret=BINANCE.api_secret,
            testnet=BINANCE.testnet,
        )
        logger.info(
            f"Binance client created "
            f"({'TESTNET' if BINANCE.testnet else 'LIVE'})"
        )
    return _client


async def get_account_balance() -> float:
    try:
        client = await get_client()
        balances = await client.futures_account_balance()
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["balance"])
    except Exception as e:
        logger.error(f"Failed to fetch account balance: {e}")
    return 0.0


async def set_leverage(symbol: str, leverage: int) -> bool:
    try:
        safe = enforce_leverage(leverage)
        client = await get_client()
        await client.futures_change_leverage(symbol=symbol, leverage=safe)
        logger.debug(f"Leverage set: {symbol} {safe}x")
        return True
    except Exception as e:
        logger.error(f"Failed to set leverage for {symbol}: {e}")
        return False


async def open_order(
    symbol: str,
    side: str,
    lot_size: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    strategy_version: str,
) -> dict | None:
    try:
        client    = await get_client()
        bs        = SIDE_BUY if side == "LONG" else SIDE_SELL
        close_bs  = SIDE_SELL if side == "LONG" else SIDE_BUY

        await set_leverage(symbol, TRADING.max_leverage)

        # Market entry
        order = await client.futures_create_order(
            symbol=symbol,
            side=bs,
            type=ORDER_TYPE_MARKET,
            quantity=lot_size,
        )
        order_id   = str(order["orderId"])
        filled_price = float(order.get("avgPrice", entry_price) or entry_price)
        opened_at  = datetime.now(timezone.utc).isoformat()

        # Stop loss
        await client.futures_create_order(
            symbol=symbol,
            side=close_bs,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=round(stop_loss, 2),
            closePosition=True,
        )

        # Take profit
        await client.futures_create_order(
            symbol=symbol,
            side=close_bs,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=round(take_profit, 2),
            closePosition=True,
        )

        trade = OpenTrade(
            trade_id=order_id,
            symbol=symbol,
            side=side,
            entry_price=filled_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            opened_at=opened_at,
            strategy_version=strategy_version,
            order_id=order_id,
        )
        state.open_trades[symbol] = trade

        logger.info(
            f"[LIVE] Order opened: {symbol} {side} "
            f"entry={filled_price:.2f} SL={stop_loss:.2f} "
            f"TP={take_profit:.2f} lot={lot_size:.6f} "
            f"order_id={order_id}"
        )
        return {
            "order_id":    order_id,
            "symbol":      symbol,
            "side":        side,
            "entry_price": filled_price,
            "lot_size":    lot_size,
            "opened_at":   opened_at,
            "status":      "FILLED",
        }

    except Exception as e:
        logger.error(f"[LIVE] Failed to open order {symbol} {side}: {e}")
        return None


async def close_order(
    symbol: str,
    exit_price: float,
    reason: str,
) -> dict | None:
    try:
        trade  = state.open_trades.get(symbol)
        if not trade:
            logger.warning(f"[LIVE] No open trade found for {symbol}")
            return None

        client   = await get_client()
        close_bs = SIDE_SELL if trade.side == "LONG" else SIDE_BUY

        order = await client.futures_create_order(
            symbol=symbol,
            side=close_bs,
            type=ORDER_TYPE_MARKET,
            quantity=trade.lot_size,
            reduceOnly=True,
        )
        filled_price = float(order.get("avgPrice", exit_price) or exit_price)

        if trade.side == "LONG":
            pnl = (filled_price - trade.entry_price) * trade.lot_size
        else:
            pnl = (trade.entry_price - filled_price) * trade.lot_size

        pnl_pct = pnl / state.balance if state.balance else 0.0
        state.record_closed_trade(pnl)
        del state.open_trades[symbol]

        logger.info(
            f"[LIVE] Order closed: {symbol} "
            f"exit={filled_price:.2f} pnl={pnl:+.4f} ({pnl_pct:+.2%}) "
            f"reason={reason}"
        )
        return {
            "symbol":     symbol,
            "exit_price": filled_price,
            "pnl":        pnl,
            "pnl_pct":    pnl_pct,
            "reason":     reason,
        }

    except Exception as e:
        logger.error(f"[LIVE] Failed to close order {symbol}: {e}")
        return None


async def close_all_positions(reason: str = "EMERGENCY") -> None:
    for symbol in list(state.open_trades.keys()):
        trade = state.open_trades[symbol]
        await close_order(symbol, trade.entry_price, reason)
    logger.warning(f"All positions closed: reason={reason}")