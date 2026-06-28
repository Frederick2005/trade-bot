"""
Generates weekly performance report and sends to Telegram.
Called automatically by the bot every Sunday at 08:00 UTC,
or manually via /report Telegram command.

Usage:
    python scripts/weekly_report.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from loguru import logger
from app.learning.queries import overall_performance, win_rate_by_rsi_bucket, win_rate_by_hour
from app.database.trades import get_trades_since
from app.notifications import telegram
from app.notifications.messages import weekly_report as msg_weekly
from app.state import state
import app.ai.model as ai_model


async def generate_and_send() -> None:
    logger.info("Generating weekly report...")

    since  = datetime.now(timezone.utc) - timedelta(days=7)
    trades = await get_trades_since(since)

    if not trades:
        await telegram.send(
            "📊 *Weekly Report*\n\nNo trades this week."
        )
        return

    pnls    = [float(t.get("profit_loss") or 0) for t in trades]
    total   = len(trades)
    winning = sum(1 for p in pnls if p > 0)
    win_rate = winning / total if total else 0

    period = (
        f"{since.strftime('%b %d')} – "
        f"{datetime.now(timezone.utc).strftime('%b %d')}"
    )

    msg = msg_weekly(
        period=period,
        total_trades=total,
        winning=winning,
        win_rate=win_rate,
        total_pnl=sum(pnls),
        best_trade=max(pnls) if pnls else 0,
        worst_trade=min(pnls) if pnls else 0,
        balance=state.balance,
        model_version=ai_model.get_version(),
    )
    await telegram.send(msg)

    # Append learning insights
    await _send_learning_insights()

    # Trigger parameter tuning
    from app.learning.tuner import maybe_tune
    tuned = await maybe_tune()
    if tuned:
        await telegram.send("🔧 *Strategy parameters auto-tuned based on weekly data.*")

    # Trigger model retraining
    from app.database.trades import count_closed_trades
    from app.config import AI
    total_ever = await count_closed_trades()
    if total_ever >= AI.min_trades_to_train:
        from app.ai.trainer import run_training
        logger.info("Running weekly model retraining...")
        metrics = await run_training()
        if metrics:
            await telegram.send(
                f"🧠 *Weekly model retrained*\n"
                f"Version:  `{metrics.get('version')}`\n"
                f"Accuracy: `{metrics.get('accuracy', 0):.2%}`\n"
                f"Trained on `{metrics.get('trained_on')}` trades"
            )

    logger.info("Weekly report complete")


async def _send_learning_insights() -> None:
    try:
        rsi_data  = await win_rate_by_rsi_bucket()
        hour_data = await win_rate_by_hour()

        if rsi_data:
            best_rsi  = max(rsi_data, key=lambda x: x["win_rate"])
            worst_rsi = min(rsi_data, key=lambda x: x["win_rate"])
            await telegram.send(
                f"📈 *RSI Insights*\n"
                f"Best bucket:  RSI `{best_rsi['rsi_bucket']}-{best_rsi['rsi_bucket']+5}` "
                f"→ `{best_rsi['win_rate']:.0%}` win rate ({best_rsi['total']} trades)\n"
                f"Worst bucket: RSI `{worst_rsi['rsi_bucket']}-{worst_rsi['rsi_bucket']+5}` "
                f"→ `{worst_rsi['win_rate']:.0%}` win rate ({worst_rsi['total']} trades)"
            )

        if hour_data:
            best_hour = hour_data[0]
            await telegram.send(
                f"🕐 *Best trading hour (UTC): `{best_hour['hour']:02d}:00`*\n"
                f"Win rate: `{best_hour['win_rate']:.0%}` | "
                f"P&L: `{best_hour['total_pnl']:+.4f}`"
            )
    except Exception as e:
        logger.error(f"Failed to send learning insights: {e}")


if __name__ == "__main__":
    asyncio.run(generate_and_send())