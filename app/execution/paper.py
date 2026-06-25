from loguru import logger
from app.config import TRADING
from app.state import state


def enforce_leverage(requested: int) -> int:
    if requested > TRADING.max_leverage:
        logger.warning(
            f"Leverage {requested}x exceeds ceiling "
            f"{TRADING.max_leverage}x — capped"
        )
        return TRADING.max_leverage
    return requested


def auto_reduce_leverage(balance: float, original_leverage: int) -> int:
    """
    Reduce leverage automatically when balance drops below 80% of start.
    Protects a small account from being wiped by a string of losses.
    """
    if state.starting_balance == 0:
        return original_leverage

    pct_remaining = balance / state.starting_balance
    if pct_remaining < 0.80:
        reduced = max(1, original_leverage - 2)
        logger.warning(
            f"Balance at {pct_remaining:.0%} of start — "
            f"leverage auto-reduced to {reduced}x"
        )
        return reduced
    return original_leverage


def check_emergency_stop() -> tuple[bool, str]:
    """
    Hard stop conditions that trigger regardless of other guards.
    Returns (should_stop, reason).
    """
    if state.drawdown_pct() >= TRADING.max_drawdown:
        return True, (
            f"EMERGENCY STOP: drawdown {state.drawdown_pct() * 100:.2f}% "
            f">= {TRADING.max_drawdown * 100:.1f}%"
        )
    return False, ""


def check_min_balance(balance: float, min_balance: float = 50.0) -> tuple[bool, str]:
    if balance < min_balance:
        return False, f"Balance ${balance:.2f} below minimum ${min_balance:.2f}"
    return True, "OK"