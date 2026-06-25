import asyncio
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
from loguru import logger
from app.config import TELEGRAM, TRADING
from app.state import state


_bot: Bot | None = None
_app: Application | None = None


def is_enabled() -> bool:
    return TELEGRAM.enabled


async def send(message: str) -> None:
    if not is_enabled():
        logger.debug(f"Telegram disabled — skipped: {message[:60]}")
        return
    try:
        bot = await _get_bot()
        await bot.send_message(
            chat_id=TELEGRAM.chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


async def _get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=TELEGRAM.bot_token)
    return _bot


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from app.notifications.messages import heartbeat
    msg = heartbeat(
        balance=state.balance,
        open_trades=state.open_trade_count(),
        binance_ok=state.binance_connected,
        supabase_ok=state.supabase_connected,
        mode=TRADING.mode,
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from app.notifications.messages import bot_paused
    state.is_paused = True
    logger.warning("Bot paused via Telegram command")
    await update.message.reply_text(
        bot_paused("Telegram /pause"), parse_mode=ParseMode.MARKDOWN
    )


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from app.notifications.messages import bot_resumed
    state.is_paused = False
    logger.info("Bot resumed via Telegram command")
    await update.message.reply_text(
        bot_resumed(), parse_mode=ParseMode.MARKDOWN
    )


async def cmd_closeall(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if TRADING.mode == "live":
        from app.execution.binance import close_all_positions
        await close_all_positions(reason="MANUAL")
    else:
        from app.execution.paper import close_order
        for symbol in list(state.open_trades.keys()):
            await close_order(symbol, 0.0, "MANUAL")
    await update.message.reply_text(
        "🚨 All positions closed.", parse_mode=ParseMode.MARKDOWN
    )


async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📊 Generating report...", parse_mode=ParseMode.MARKDOWN
    )
    from scripts.weekly_report import generate_and_send
    await generate_and_send()


async def cmd_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    args = ctx.args
    if not args:
        await update.message.reply_text("Usage: /risk 0.5")
        return
    try:
        new_risk = float(args[0]) / 100
        from app import config
        config.TRADING.risk_per_trade = new_risk
        await update.message.reply_text(
            f"✅ Risk per trade updated to `{new_risk:.1%}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(f"Risk per trade updated via Telegram: {new_risk:.1%}")
    except ValueError:
        await update.message.reply_text("Invalid value. Usage: /risk 0.5")


# ── Start the command listener ────────────────────────────────────────────────

async def start_listener() -> None:
    if not is_enabled():
        logger.info("Telegram listener not started — credentials not configured")
        return

    global _app
    _app = (
        Application.builder()
        .token(TELEGRAM.bot_token)
        .build()
    )
    _app.add_handler(CommandHandler("status",   cmd_status))
    _app.add_handler(CommandHandler("pause",    cmd_pause))
    _app.add_handler(CommandHandler("resume",   cmd_resume))
    _app.add_handler(CommandHandler("closeall", cmd_closeall))
    _app.add_handler(CommandHandler("report",   cmd_report))
    _app.add_handler(CommandHandler("risk",     cmd_risk))

    logger.info("Telegram command listener started")
    await _app.initialize()
    await _app.start()
    await _app.updater.start_polling()