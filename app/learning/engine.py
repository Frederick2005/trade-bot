from loguru import logger
from app.learning.analyser import (
    analyse_closed_trade,
    get_trade_context_by_id,
    get_recent_win_rate,
)
from app.ai.labels import label_closed_trade
from app.ai.features import build_feature_vector
from app.learning.tuner import maybe_tune
from app.database.context import log_bot_event
from app.database.trades import count_closed_trades
from app.config import AI
from app.state import state
import asyncio
from datetime import datetime, timezone


async def on_trade_closed(
    trade: dict,
    indicators_1h: dict,
    indicators_4h: dict,
) -> None:
    """
    Called by the engine every time a trade is closed.
    Runs the full learning pipeline:
      1. Analyse why the trade won or lost
      2. Build the feature vector for this trade
      3. Save it as a training label
      4. Check if parameter tuning is needed
      5. Check if model retraining is needed
      6. Send Telegram learning summary
    """
    trade_id = trade.get("id")
    pnl      = trade.get("profit_loss", 0.0) or 0.0
    pnl_pct  = trade.get("profit_pct", 0.0) or 0.0

    logger.info(f"Learning loop started for trade {trade_id}")

    # ── 1. Fetch context + analyse ─────────────────────────────────
    context  = await get_trade_context_by_id(trade_id)
    analysis = await analyse_closed_trade(trade, context)

    # ── 2. Build feature vector ────────────────────────────────────
    recent_wr = await get_recent_win_rate(last_n=10)
    features  = build_feature_vector(
        indicators_1h=indicators_1h,
        indicators_4h=indicators_4h,
        candle_time=datetime.now(timezone.utc),
        recent_win_rate=recent_wr,
        current_drawdown=state.drawdown_pct(),
    )

    # ── 3. Save training label ─────────────────────────────────────
    await label_closed_trade(
        trade_id=trade_id,
        profit_loss=pnl,
        profit_pct=pnl_pct,
        features=features,
    )

    # ── 4. Parameter tuning — runs every 20 trades ─────────────────
    total = await count_closed_trades()
    if total > 0 and total % 20 == 0:
        logger.info(f"Running tuner at {total} trades...")
        tuned = await maybe_tune()
        if tuned:
            await log_bot_event(
                "INFO",
                "Strategy parameters auto-tuned",
                {"total_trades": total},
            )

    # ── 5. Model retraining — runs every 50 trades once we have enough ──
    if total >= AI.min_trades_to_train and total % 50 == 0:
        logger.info(f"Triggering model retraining at {total} trades...")
        asyncio.create_task(_retrain_async())

    # ── 6. Log analysis ────────────────────────────────────────────
    result_icon = "✅" if pnl > 0 else "❌"
    flags_str   = ", ".join(analysis["flags"]) if analysis["flags"] else "none"

    logger.info(
        f"{result_icon} Learning loop complete: "
        f"{trade['symbol']} {trade['side']} | "
        f"pnl={pnl:+.4f} | flags=[{flags_str}] | "
        f"recent_wr={recent_wr:.1%}"
    )

    # Send to Telegram
    if total % 10 == 0:
        await _send_learning_update(total, recent_wr, analysis)


async def _retrain_async() -> None:
    """Runs model retraining in the background without blocking the engine."""
    try:
        from app.ai.trainer import run_training
        from app.notifications import telegram
        from app.notifications.messages import weekly_report

        logger.info("Background model retraining started")
        metrics = await run_training()

        if metrics:
            msg = (
                f"🧠 *Model retrained*\n"
                f"Version:  `{metrics.get('version')}`\n"
                f"Accuracy: `{metrics.get('accuracy', 0):.2%}`\n"
                f"F1 score: `{metrics.get('f1_score', 0):.2%}`\n"
                f"Trained on: `{metrics.get('trained_on')}` trades"
            )
            await telegram.send(msg)
        else:
            logger.info("Retraining skipped — model did not improve")

    except Exception as e:
        logger.error(f"Background retraining failed: {e}")


async def _send_learning_update(
    total_trades: int,
    win_rate: float,
    analysis: dict,
) -> None:
    try:
        from app.notifications import telegram
        flags_str = ", ".join(analysis["flags"]) if analysis["flags"] else "none"
        msg = (
            f"🧠 *Learning update — {total_trades} trades*\n"
            f"Last trade: `{'WIN' if analysis['pnl'] > 0 else 'LOSS'}` "
            f"{analysis['pnl']:+.4f} USDT\n"
            f"Win rate (last 10): `{win_rate:.1%}`\n"
            f"Flags: _{flags_str}_"
        )
        await telegram.send(msg)
    except Exception as e:
        logger.error(f"Failed to send learning update: {e}")