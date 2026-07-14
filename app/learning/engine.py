"""
app/learning/engine.py
Orchestrates the full learning loop after every trade closes.

LEARNING PIPELINE AFTER EVERY TRADE:
─────────────────────────────────────
1. Analyse why it won/lost → error_analysis table
2. Build extended feature vector (38 features)
3. Save training label → training_labels table
4. Update AI prediction outcome (was it correct?)
5. Save risk metrics snapshot
6. Check if parameter tuning needed (every 20 trades)
7. Check if model retraining needed (every 50 trades)
8. Send learning summary to Telegram

Every step adds data that makes future predictions better.
The system is designed to improve continuously without human input.
"""
import asyncio
import time
from datetime import datetime, timezone
from loguru import logger

from app.learning.analyser import (
    analyse_closed_trade,
    get_trade_context_by_id,
    get_recent_win_rate,
)
from app.ai.labels import label_closed_trade
from app.ai.features import build_feature_vector
from app.learning.tuner import maybe_tune
from app.database.context import (
    log_bot_event,
    update_prediction_outcome,
    save_ai_prediction_log,
)
from app.database.trades import count_closed_trades
from app.config import AI
from app.state import state


async def on_trade_closed(
    trade: dict,
    indicators_1h: dict,
    indicators_4h: dict,
    ai_confidence: float = 0.0,
    ai_features: dict | None = None,
    ai_feature_importance: dict | None = None,
) -> None:
    trade_id = trade.get("id")
    pnl      = trade.get("profit_loss", 0.0) or 0.0
    pnl_pct  = trade.get("profit_pct", 0.0) or 0.0
    won      = pnl > 0

    logger.info(f"Learning loop: trade {trade_id} {'WIN' if won else 'LOSS'} {pnl:+.4f}")

    # ── 1. Fetch context and analyse ──────────────────────────────
    context  = await get_trade_context_by_id(trade_id)
    analysis = await analyse_closed_trade(trade, context)

    # ── 2. Build feature vector ───────────────────────────────────
    recent_wr = await get_recent_win_rate(last_n=10)
    features  = ai_features or build_feature_vector(
        indicators_1h=indicators_1h,
        indicators_4h=indicators_4h,
        candle_time=datetime.now(timezone.utc),
        recent_win_rate=recent_wr,
        current_drawdown=state.drawdown_pct(),
    )

    # ── 3. Save training label ────────────────────────────────────
    await label_closed_trade(
        trade_id=trade_id,
        profit_loss=pnl,
        profit_pct=pnl_pct,
        features=features,
    )

    # ── 4. Update AI prediction outcome ──────────────────────────
    if ai_confidence > 0:
        await update_prediction_outcome(
            trade_id=trade_id,
            outcome="CORRECT" if won else "INCORRECT",
            result="WIN" if won else "LOSS",
        )

    # ── 5. Save risk metrics snapshot ────────────────────────────
    await _save_risk_snapshot()

    # ── 6. Parameter tuning — every 20 trades ────────────────────
    total = await count_closed_trades()
    if total > 0 and total % 20 == 0:
        logger.info(f"Running tuner at {total} trades...")
        tuned = await maybe_tune()
        if tuned:
            await log_bot_event("INFO", f"Strategy auto-tuned at {total} trades")

    # ── 7. Model retraining — every 50 trades ────────────────────
    if total >= AI.min_trades_to_train and total % 50 == 0:
        logger.info(f"Triggering model retraining at {total} trades...")
        asyncio.create_task(_retrain_async())

    # ── 8. Learning summary every 10 trades ──────────────────────
    if total % 10 == 0:
        await _send_learning_update(total, recent_wr, analysis, won)

    logger.info(
        f"Learning loop complete: {trade['symbol']} {trade['side']} "
        f"{'WIN' if won else 'LOSS'} | flags={analysis['flags']} | "
        f"error={analysis.get('error_type')} | win_rate={recent_wr:.1%}"
    )


async def _save_risk_snapshot() -> None:
    """Save current risk metrics to Supabase."""
    try:
        from app.database.client import get_client
        client = get_client()

        # Simple drawdown and balance snapshot
        client.table("risk_metrics").insert({
            "snapshot_time":    datetime.now(timezone.utc).isoformat(),
            "period":           "TRADE",
            "balance":          state.balance,
            "equity":           state.equity,
            "current_drawdown": state.drawdown_pct(),
            "peak_balance":     state.starting_balance,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save risk snapshot: {e}")


async def _retrain_async() -> None:
    try:
        from app.ai.trainer import run_training
        from app.notifications import telegram
        logger.info("Background model retraining started")
        metrics = await run_training()
        if metrics:
            await telegram.send(
                f"🧠 *Model retrained*\n"
                f"Version:  `{metrics.get('version')}`\n"
                f"Accuracy: `{metrics.get('accuracy', 0):.2%}`\n"
                f"F1 score: `{metrics.get('f1_score', 0):.2%}`\n"
                f"Trained on: `{metrics.get('trained_on')}` trades"
            )
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")


async def _send_learning_update(
    total: int,
    win_rate: float,
    analysis: dict,
    won: bool,
) -> None:
    try:
        from app.notifications import telegram
        flags_str = ", ".join(analysis["flags"]) if analysis["flags"] else "none"
        error_str = analysis.get("error_type") or "n/a"
        msg = (
            f"🧠 *Learning update — {total} trades*\n"
            f"Last: `{'WIN ✅' if won else 'LOSS ❌'}` "
            f"{analysis['pnl']:+.4f} USDT\n"
            f"Win rate (last 10): `{win_rate:.1%}`\n"
            f"Error type: `{error_str}`\n"
            f"Flags: _{flags_str}_"
        )
        await telegram.send(msg)
    except Exception as e:
        logger.error(f"Failed to send learning update: {e}")