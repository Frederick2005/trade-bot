import os
from dotenv import load_dotenv
from pydantic import BaseModel
from loguru import logger

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _optional(key: str, default: str = "") -> str:
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
    logger.warning("Telegram not configured — notifications disabled.")


# ── Trading ───────────────────────────────────────────────────────────────────

class TradingConfig(BaseModel):
    mode: str
    symbols: list[str]
    risk_per_trade: float
    max_leverage: int
    daily_loss_limit: float
    max_drawdown: float
    max_open_trades: int

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
    signal: str
    trend: str

TIMEFRAMES = TimeframeConfig(
    signal=_optional("SIGNAL_TIMEFRAME", "1h"),
    trend=_optional("TREND_TIMEFRAME", "4h"),
)


# ── Strategy ──────────────────────────────────────────────────────────────────

class StrategyConfig(BaseModel):
    rsi_lower: float
    rsi_upper: float
    atr_multiplier: float
    min_risk_reward: float

STRATEGY = StrategyConfig(
    rsi_lower=float(_optional("RSI_LOWER", "45")),
    rsi_upper=float(_optional("RSI_UPPER", "60")),
    atr_multiplier=float(_optional("ATR_MULTIPLIER", "1.5")),
    min_risk_reward=float(_optional("MIN_RISK_REWARD", "2.0")),
)


# ── AI ────────────────────────────────────────────────────────────────────────
class AIConfig(BaseModel):
    model_config = {"protected_namespaces": ()}  # ← tells Pydantic to relax the restriction
    min_confidence: float
    min_trades_to_train: int
    retrain_every_days: int
    model_dir: str

AI = AIConfig(
    min_confidence=float(_optional("AI_MIN_CONFIDENCE", "0.70")),
    min_trades_to_train=int(_optional("AI_MIN_TRADES", "200")),
    retrain_every_days=int(_optional("AI_RETRAIN_DAYS", "7")),
    model_dir=_optional("AI_MODEL_DIR", "models"),
)


# ── Notifications ─────────────────────────────────────────────────────────────

class NotificationConfig(BaseModel):
    weekly_report_day: str
    weekly_report_hour: int
    heartbeat_hours: int

NOTIFICATIONS = NotificationConfig(
    weekly_report_day=_optional("WEEKLY_REPORT_DAY", "sunday"),
    weekly_report_hour=int(_optional("WEEKLY_REPORT_HOUR", "8")),
    heartbeat_hours=int(_optional("HEARTBEAT_HOURS", "6")),
)


# ── Startup summary ───────────────────────────────────────────────────────────

def log_config() -> None:
    logger.info("=" * 40)
    logger.info("Bot configuration loaded")
    logger.info(f"  Mode        : {TRADING.mode.upper()}")
    logger.info(f"  Symbols     : {', '.join(TRADING.symbols)}")
    logger.info(f"  Binance     : {'TESTNET' if BINANCE.testnet else 'LIVE ⚠️'}")
    logger.info(f"  Risk/trade  : {TRADING.risk_per_trade * 100:.1f}%")
    logger.info(f"  Max leverage: {TRADING.max_leverage}x")
    logger.info(f"  Daily limit : {TRADING.daily_loss_limit * 100:.1f}%")
    logger.info(f"  Max drawdown: {TRADING.max_drawdown * 100:.1f}%")
    logger.info(f"  Signal TF   : {TIMEFRAMES.signal}  Trend TF: {TIMEFRAMES.trend}")
    logger.info(f"  AI min conf : {AI.min_confidence * 100:.0f}%")
    logger.info(f"  Telegram    : {'enabled' if TELEGRAM.enabled else 'disabled'}")
    logger.info("=" * 40)   