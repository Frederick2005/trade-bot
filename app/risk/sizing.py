from loguru import logger
from app.config import TRADING


def calculate_lot_size(
    balance: float,
    entry_price: float,
    stop_loss: float,
    leverage: int | None = None,
) -> tuple[float, float]:
    """
    Returns (lot_size, notional_value) using 1% account risk per trade.

    lot_size     = units of the asset to buy/sell
    notional     = lot_size × entry_price (total position value)
    """
    if leverage is None:
        leverage = TRADING.max_leverage

    risk_amount   = balance * TRADING.risk_per_trade
    stop_distance = abs(entry_price - stop_loss)

    if stop_distance == 0:
        logger.error("Stop loss distance is zero — cannot size position")
        return 0.0, 0.0

    # How many units to risk exactly risk_amount on the stop
    lot_size = risk_amount / stop_distance

    # Cap by max leverage
    max_notional = balance * leverage
    notional     = lot_size * entry_price

    if notional > max_notional:
        lot_size = max_notional / entry_price
        notional = max_notional
        logger.debug(
            f"Position capped by {leverage}x leverage: "
            f"lot={lot_size:.6f} notional=${notional:.2f}"
        )

    # Binance minimum notional check (~$5)
    if notional < 5.0:
        logger.warning(
            f"Notional ${notional:.2f} below Binance minimum $5 — skipping"
        )
        return 0.0, 0.0

    logger.debug(
        f"Position sized: risk=${risk_amount:.2f} "
        f"lot={lot_size:.6f} notional=${notional:.2f}"
    )
    return round(lot_size, 6), round(notional, 2)