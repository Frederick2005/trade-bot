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
    correlation_mode: str   # "block" | "allow" | "reduce_size" — see app/risk/guards.py

TRADING = TradingConfig(
    mode=_optional("TRADING_MODE", "paper"),
    symbols=[s.strip() for s in _optional("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")],
    risk_per_trade=float(_optional("RISK_PER_TRADE", "0.01")),
    max_leverage=int(_optional("MAX_LEVERAGE", "5")),
    daily_loss_limit=float(_optional("DAILY_LOSS_LIMIT", "0.03")),
    max_drawdown=float(_optional("MAX_DRAWDOWN", "0.10")),
    max_open_trades=int(_optional("MAX_OPEN_TRADES", "2")),
    correlation_mode=_optional("CORRELATION_MODE", "reduce_size"),
)


# ── Timeframes ────────────────────────────────────────────────────────────────
# 3-tier per the AtlasQuant v2 spec: Primary (4H) = main trend context,
# Secondary (1H) = trend confirmation + market structure, Execution (15M) =
# entry timing. `.signal` / `.trend` are kept as properties for backward
# compatibility with app/strategy/ema_rsi.py, scripts/backtest.py, and
# scripts/debug_strategy.py, which only know about a 2-tier signal/trend
# split — they now map to execution/secondary respectively, so nothing
# else needs to change to keep working.

class TimeframeConfig(BaseModel):
    primary:   str   # 4H  — main trend context (Stage 1/2)
    secondary: str   # 1H  — trend confirmation + market structure (Stage 3)
    execution: str   # 15M — entry timing (Stage 10)

    @property
    def signal(self) -> str:
        return self.execution

    @property
    def trend(self) -> str:
        return self.secondary

TIMEFRAMES = TimeframeConfig(
    primary=_optional("PRIMARY_TIMEFRAME", "4h"),
    # fall back to the old env var names so existing .env files (including
    # ones scripts/backtest.py has already auto-written SIGNAL_TIMEFRAME /
    # TREND_TIMEFRAME into) keep working without edits
    secondary=_optional("SECONDARY_TIMEFRAME", _optional("TREND_TIMEFRAME", "1h")),
    execution=_optional("EXECUTION_TIMEFRAME", _optional("SIGNAL_TIMEFRAME", "15m")),
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


# ── Decision Engine (AtlasQuant v2 multi-stage pipeline) ───────────────────────
# Every value here is a starting hypothesis, not a validated optimum — see
# app/strategy/decision_engine.py's module docstring on why a higher
# min_reward_ratio isn't automatically better. Sweep these against real
# backtest data before trusting any specific number.

class DecisionEngineConfig(BaseModel):
    quality_threshold:   float   # 0-100, minimum score to accept a trade
    min_reward_ratio:    float   # take-profit distance = stop distance * this
    max_losing_streak:   int     # consecutive losses before pausing new entries

DECISION_ENGINE = DecisionEngineConfig(
    quality_threshold=float(_optional("QUALITY_SCORE_THRESHOLD", "80")),
    min_reward_ratio=float(_optional("MIN_REWARD_RATIO", "3.0")),
    max_losing_streak=int(_optional("MAX_LOSING_STREAK", "2")),
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
    logger.info(f"  Timeframes  : primary={TIMEFRAMES.primary} secondary={TIMEFRAMES.secondary} execution={TIMEFRAMES.execution}")
    logger.info(f"  Quality gate: {DECISION_ENGINE.quality_threshold}/100  Min R:R: {DECISION_ENGINE.min_reward_ratio}:1")
    logger.info(f"  Loss streak : pause after {DECISION_ENGINE.max_losing_streak} consecutive losses")
    logger.info(f"  AI min conf : {AI.min_confidence * 100:.0f}%")
    logger.info(f"  Telegram    : {'enabled' if TELEGRAM.enabled else 'disabled'}")
    logger.info("=" * 40)