import os
from dotenv import load_dotenv
from pydantic import BaseModel
from loguru import logger

# Load .env file into environment
load_dotenv()


def _require(key: str) -> str:
    """Fetch a required env variable — crash early with a clear message if missing."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _optional(key: str, default: str = "") -> str:
    """Fetch an optional env variable with a fallback default."""
    return os.getenv(key, default)


# ── Binance ───────────────────────────────────────────────────────────────────

class BinanceConfig(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool

BINANCE = BinanceConfig(
    api_key=_require("BINANCE_API_KEY"),
    api_secret=_require("BINANCE_API_SECRET"),
    testnet=_optional("BINANCE_TESTNET", "true").lower() == "true",
)


# ── Supabase ──────────────────────────────────────────────────────────────────

class SupabaseConfig(BaseModel):
    url: str
    key: str

SUPABASE = SupabaseConfig(
    url=_require("SUPABASE_URL"),
    key=_require("SUPABASE_KEY"),
)


# ── Telegram ──────────────────────────────────────────────────────────────────

class TelegramConfig(BaseModel):
    bot_token: str
    chat_id: str
    enabled: bool

TELEGRAM = TelegramConfig(
    bot_token=_optional("TELEGRAM_BOT_TOKEN", ""),
    chat_id=_optional("TELEGRAM_CHAT_ID", ""),
    enabled=bool(
        _optional("TELEGRAM_BOT_TOKEN") and _optional("TELEGRAM_CHAT_ID")
    ),
)

if not TELEGRAM.enabled:
    logger.warning("Telegram credentials not set — notifications disabled until configured.")


# ── Trading settings ──────────────────────────────────────────────────────────

class TradingConfig(BaseModel):
    mode: str                  # 'paper' or 'live'
    symbols: list[str]         # ['BTCUSDT', 'ETHUSDT']
    risk_per_trade: float      # 0.01 = 1%
    max_leverage: int          # hard ceiling
    daily_loss_limit: float    # 0.03 = 3%
    max_drawdown: float        # 0.10 = 10%
    max_open_trades: int       # 2

TRADING = TradingConfig(
    mode=_optional("TRADING_MODE", "paper"),
    symbols=[s.strip() for s in _optional("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")],
    risk_per_trade=float(_optional("RISK_PER_TRADE", "0.01")),
    max_leverage=int(_optional("MAX_LEVERAGE", "5")),
    daily_loss_limit=float(_optional("DAILY_LOSS_LIMIT", "0.03")),
    max_drawdown=float(_optional("MAX_DRAWDOWN", "0.10")),
    max_open_trades=int(_optional("MAX_OPEN_TRADES", "2")),
)


# ── Timeframes ────────────────────────────────────────────────────────────────

class TimeframeConfig(BaseModel):
    signal: str     # '1h' — strategy signals generated here
    trend: str      # '4h' — higher timeframe trend filter

TIMEFRAMES = TimeframeConfig(
    signal=_optional("SIGNAL_TIMEFRAME", "1h"),
    trend=_optional("TREND_TIMEFRAME", "4h"),
)


# ── Strategy parameters ───────────────────────────────────────────────────────

class StrategyConfig(BaseModel):
    rsi_lower: float        # min RSI to enter
    rsi_upper: float        # max RSI to enter
    atr_multiplier: float   # stop loss = ATR * this
    min_risk_reward: float  # skip trade if RR below this

STRATEGY = StrategyConfig(
    rsi_lower=float(_optional("RSI_LOWER", "45")),
    rsi_upper=float(_optional("RSI_UPPER", "60")),
    atr_multiplier=float(_optional("ATR_MULTIPLIER", "1.5")),
    min_risk_reward=float(_optional("MIN_RISK_REWARD", "2.0")),
)


# ── Notifications schedule ────────────────────────────────────────────────────

class NotificationConfig(BaseModel):
    weekly_report_day: str   # 'sunday'
    weekly_report_hour: int  # hour in UTC
    heartbeat_hours: int     # send heartbeat every X hours

NOTIFICATIONS = NotificationConfig(
    weekly_report_day=_optional("WEEKLY_REPORT_DAY", "sunday"),
    weekly_report_hour=int(_optional("WEEKLY_REPORT_HOUR", "8")),
    heartbeat_hours=int(_optional("HEARTBEAT_HOURS", "6")),
)


# ── Startup summary ───────────────────────────────────────────────────────────

def log_config():
    """Log a safe summary of loaded config on startup — never logs secrets."""
    logger.info("=== Configuration loaded ===")
    logger.info(f"Mode:           {TRADING.mode.upper()}")
    logger.info(f"Symbols:        {', '.join(TRADING.symbols)}")
    logger.info(f"Binance:        {'testnet' if BINANCE.testnet else 'LIVE'}")
    logger.info(f"Risk per trade: {TRADING.risk_per_trade * 100:.1f}%")
    logger.info(f"Max leverage:   {TRADING.max_leverage}x")
    logger.info(f"Daily loss limit: {TRADING.daily_loss_limit * 100:.1f}%")
    logger.info(f"Max drawdown:   {TRADING.max_drawdown * 100:.1f}%")
    logger.info(f"Signal TF:      {TIMEFRAMES.signal} | Trend TF: {TIMEFRAMES.trend}")
    logger.info(f"Telegram:       {'enabled' if TELEGRAM.enabled else 'disabled'}")
    logger.info("============================")