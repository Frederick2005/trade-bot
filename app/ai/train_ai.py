"""
scripts/train_ai.py

Standalone entry point to run AI model training on-demand, without
waiting for the live engine's every-50-trades auto-trigger. Use this
right after a backtest run + clear_training_labels.py to train on the
now-validated signal.

Usage:
    python scripts/train_ai.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


async def main():
    from app.ai.trainer import run_training
    from app.ai.labels import get_label_stats

    stats = await get_label_stats()
    logger.info(
        f"Training data available: {stats['total']} trades | "
        f"win_rate={stats['win_rate']:.1%}"
    )

    if stats["total"] == 0:
        logger.error(
            "No training_labels found. Run scripts/backtest.py first "
            "(it populates training_labels when save_labels=True, which "
            "main() does for the winning combo)."
        )
        return

    metrics = await run_training()

    if metrics is None:
        logger.info(
            "Training ran but the model was NOT deployed — see the "
            "should_deploy() log line above for the exact reason "
            "(too few AI-approved trades in validation, doesn't beat "
            "baseline expectancy, or doesn't beat the current model)."
        )
        return

    logger.info("=" * 50)
    logger.info(f"Model deployed: {metrics.get('version')}")
    logger.info(f"  Accuracy:            {metrics.get('accuracy', 0):.2%}")
    logger.info(f"  F1 score:            {metrics.get('f1_score', 0):.2%}")
    logger.info(f"  Trained on:          {metrics.get('trained_on')} trades")
    if metrics.get("baseline_expectancy_pct") is not None:
        logger.info(f"  Baseline expectancy: {metrics.get('baseline_expectancy_pct'):+.3f}%")
        logger.info(f"  AI-approved expectancy: {metrics.get('approved_expectancy_pct'):+.3f}% "
                    f"(n={metrics.get('approved_trade_count')})")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())