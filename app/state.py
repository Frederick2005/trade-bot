from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class OpenTrade:
    trade_id: str
    symbol: str
    side: str                 # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    opened_at: str
    strategy_version: str
    order_id: Optional[str] = None


@dataclass
class BotState:
    # Account
    balance: float = 0.0
    equity: float = 0.0
    starting_balance: float = 0.0

    # Open positions — keyed by symbol
    open_trades: dict[str, OpenTrade] = field(default_factory=dict)

    # Daily loss tracking — resets at midnight UTC
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    daily_reset_date: Optional[date] = None

    # Control flags
    is_paused: bool = False        # set by /pause Telegram command
    is_running: bool = False       # set True once engine starts

    # Connection health
    binance_connected: bool = False
    supabase_connected: bool = False
    last_heartbeat: Optional[str] = None

    # AI model
    active_model_version: Optional[str] = None
    model_loaded: bool = False

    def open_trade_count(self) -> int:
        return len(self.open_trades)

    def has_open_trade(self, symbol: str) -> bool:
        return symbol in self.open_trades

    def can_trade(self, max_open: int) -> tuple[bool, str]:
        if self.is_paused:
            return False, "Bot is paused"
        if self.open_trade_count() >= max_open:
            return False, f"Max open trades reached ({max_open})"
        return True, "OK"

    def record_closed_trade(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self.daily_trade_count += 1
        self.balance += pnl
        self.equity = self.balance

    def drawdown_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.starting_balance - self.balance) / self.starting_balance

    def daily_loss_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return abs(min(self.daily_pnl, 0)) / self.balance


# Single global instance shared across all modules
state = BotState()