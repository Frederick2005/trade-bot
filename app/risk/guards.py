from loguru import logger
from app.config import TRADING
from app.state import state, OpenTrade


CORRELATED_PAIRS = [
    {"BTCUSDT", "ETHUSDT"},   # BTC and ETH are highly correlated
]


def check_all(symbol: str, side: str) -> tuple[bool, str]:
    """
    Run all risk guards. Returns (allowed, reason).
    Order matters — cheapest checks first.
    """
    checks = [
        _check_paused,
        _check_max_open_trades,
        _check_duplicate_symbol,
        _check_daily_loss_limit,
        _check_max_drawdown,
        _check_correlation,
    ]
    for check in checks:
        allowed, reason = check(symbol, side)
        if not allowed:
            logger.warning(f"Trade blocked [{symbol} {side}]: {reason}")
            return False, reason
    return True, "OK"


def _check_paused(symbol: str, side: str) -> tuple[bool, str]:
    if state.is_paused:
        return False, "Bot is paused"
    return True, "OK"


def _check_max_open_trades(symbol: str, side: str) -> tuple[bool, str]:
    if state.open_trade_count() >= TRADING.max_open_trades:
        return False, f"Max open trades reached ({TRADING.max_open_trades})"
    return True, "OK"


def _check_duplicate_symbol(symbol: str, side: str) -> tuple[bool, str]:
    if state.has_open_trade(symbol):
        return False, f"Already have open trade on {symbol}"
    return True, "OK"


def _check_daily_loss_limit(symbol: str, side: str) -> tuple[bool, str]:
    if state.daily_loss_pct() >= TRADING.daily_loss_limit:
        return False, (
            f"Daily loss limit reached: "
            f"{state.daily_loss_pct() * 100:.2f}% >= "
            f"{TRADING.daily_loss_limit * 100:.1f}%"
        )
    return True, "OK"


def _check_max_drawdown(symbol: str, side: str) -> tuple[bool, str]:
    if state.drawdown_pct() >= TRADING.max_drawdown:
        return False, (
            f"Max drawdown reached: "
            f"{state.drawdown_pct() * 100:.2f}% >= "
            f"{TRADING.max_drawdown * 100:.1f}%"
        )
    return True, "OK"


def _check_correlation(symbol: str, side: str) -> tuple[bool, str]:
    for group in CORRELATED_PAIRS:
        if symbol not in group:
            continue
        for open_symbol, trade in state.open_trades.items():
            if open_symbol in group and trade.side == side:
                return False, (
                    f"Correlated pair already open: "
                    f"{open_symbol} {trade.side}"
                )
    return True, "OK"