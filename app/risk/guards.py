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


def has_correlated_exposure(symbol: str, side: str) -> bool:
    """
    True if a same-side trade is already open on a symbol correlated with
    `symbol` (e.g. BTCUSDT LONG open while checking ETHUSDT LONG). Exposed
    separately from _check_correlation so app/engine.py can use it to size
    DOWN a correlated trade instead of blocking it outright, when
    TRADING.correlation_mode == "reduce_size".
    """
    for group in CORRELATED_PAIRS:
        if symbol not in group:
            continue
        for open_symbol, trade in state.open_trades.items():
            if open_symbol in group and open_symbol != symbol and trade.side == side:
                return True
    return False


def _check_correlation(symbol: str, side: str) -> tuple[bool, str]:
    """
    CORRELATION_MODE controls what happens when a same-side trade is
    already open on a correlated symbol (BTC/ETH move together ~0.8-0.9
    correlation historically — being LONG both at once is one concentrated
    bet on "crypto goes up" split into two positions, not two independent
    bets):

      "block"       — the original behavior: reject the second trade outright.
      "allow"       — let it through at full size. You get more trade
                      frequency, but real risk during a correlated move is
                      closer to 2x a single trade's risk, not two
                      diversified 1% risks.
      "reduce_size" — (default) let it through, but app/engine.py halves
                      the position size for the correlated trade, so total
                      correlated exposure stays capped near the original
                      single-trade risk instead of stacking to 2x.
    """
    if TRADING.correlation_mode == "block":
        for group in CORRELATED_PAIRS:
            if symbol not in group:
                continue
            for open_symbol, trade in state.open_trades.items():
                if open_symbol in group and open_symbol != symbol and trade.side == side:
                    return False, f"Correlated pair already open: {open_symbol} {trade.side}"
        return True, "OK"

    # "allow" and "reduce_size" both let the trade through here — sizing
    # adjustment for "reduce_size" happens in app/engine.py at position-size
    # calculation time, not here (this function only gates yes/no).
    return True, "OK"