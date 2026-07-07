from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from app.config import DECISION_ENGINE


@dataclass
class OpenTrade:
    trade_id:         str
    symbol:           str
    side:             str        # 'LONG' or 'SHORT'
    entry_price:      float
    stop_loss:        float
    take_profit:      float
    lot_size:         float
    opened_at:        str
    strategy_version: str
    order_id:         Optional[str] = None


@dataclass
class BotState:
    # Account
    balance:           float = 0.0
    equity:            float = 0.0
    starting_balance:  float = 0.0

    # Open positions — keyed by symbol
    open_trades: dict[str, OpenTrade] = field(default_factory=dict)

    # Daily loss tracking — resets at midnight UTC
    daily_pnl:         float         = 0.0
    daily_trade_count: int           = 0
    daily_reset_date:  Optional[date] = None

    # Losing-streak circuit breaker (AtlasQuant v2 spec: "Cooldown after
    # consecutive losses" / originally the LTA Concepts "Two-Strike Rule").
    # Threshold is configurable via MAX_LOSING_STREAK (DECISION_ENGINE.
    # max_losing_streak, default 2) rather than hardcoded — this triggers
    # on LOSS STREAK regardless of how small each loss was, which is a
    # distinct control from the daily loss % limit below.
    consecutive_losses:      int            = 0
    losing_streak_until_date: Optional[date] = None

    # Control flags
    is_paused:   bool = False
    is_running:  bool = False

    # Connection health
    binance_connected:  bool          = False
    supabase_connected: bool          = False
    last_heartbeat:     Optional[str] = None

    # AI model
    active_model_version: Optional[str] = None
    model_loaded:         bool          = False

    def open_trade_count(self) -> int:
        return len(self.open_trades)

    def has_open_trade(self, symbol: str) -> bool:
        return symbol in self.open_trades

    def can_trade(self, max_open: int) -> tuple[bool, str]:
        if self.is_paused:
            return False, "Bot is paused"
        if self.open_trade_count() >= max_open:
            return False, f"Max open trades reached ({max_open})"
        blocked, reason = self.losing_streak_blocked()
        if blocked:
            return False, reason
        return True, "OK"

    def losing_streak_blocked(self) -> tuple[bool, str]:
        """Blocked for the rest of the UTC day after
        DECISION_ENGINE.max_losing_streak consecutive losses. Resets
        automatically once that day has passed."""
        today = datetime.now(timezone.utc).date()
        if self.losing_streak_until_date is not None and today <= self.losing_streak_until_date:
            return True, (
                f"Losing-streak breaker: {self.consecutive_losses} consecutive losses — "
                f"paused until {self.losing_streak_until_date.isoformat()} (UTC)"
            )
        return False, "OK"

    def record_closed_trade(self, pnl: float) -> None:
        self.daily_pnl        += pnl
        self.daily_trade_count += 1
        self.balance           += pnl
        self.equity             = self.balance
        # Persist updated balance so next startup restores it
        _save_balance(self.balance)

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= DECISION_ENGINE.max_losing_streak:
                today = datetime.now(timezone.utc).date()
                self.losing_streak_until_date = today
        else:
            self.consecutive_losses = 0
            self.losing_streak_until_date = None

    def drawdown_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return max(0.0, (self.starting_balance - self.balance) / self.starting_balance)

    def daily_loss_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return abs(min(self.daily_pnl, 0)) / self.balance


def _save_balance(balance: float) -> None:
    """Persist paper balance to Supabase without blocking."""
    try:
        from app.database.client import set_state_value
        set_state_value("paper_balance", str(round(balance, 8)))
    except Exception:
        pass   # never crash the bot over a persistence failure


# Single global instance shared across all modules
state = BotState()