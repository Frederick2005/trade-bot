import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from loguru import logger
from app.database.client import get_client
from app.config import AI


# New model must beat this accuracy to be deployed
MIN_ACCURACY = 0.55

# Minimum improvement over current model to justify replacement
MIN_IMPROVEMENT = 0.02   # 2 percentage points


def evaluate(
    model,
    X_val: list[list[float]],
    y_val: list[int],
    pnl_val: list[float] | None = None,
) -> dict:
    """Runs the model on validation data and returns all metrics.

    pnl_val: pnl_pct for each validation row, in the SAME order as X_val/
    y_val. If provided, this also computes the trading-relevant check —
    does the subset of trades the AI would actually APPROVE (confidence
    >= AI.min_confidence) have better expectancy than just taking every
    trade? Classification accuracy alone can't tell you that: a model can
    hit 55%+ accuracy on a roughly-balanced label set without necessarily
    being a good FILTER for this specific job. This is what should_deploy()
    actually gates on now, not raw accuracy.
    """
    try:
        import numpy as np
        X = np.array(X_val)
        y = np.array(y_val)

        preds  = model.predict(X)
        probas = model.predict_proba(X)[:, 1]

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)

        # Sharpe-like score on predicted winners vs losers
        # (not a real Sharpe — just a directional proxy)
        sharpe = _pseudo_sharpe(probas, y)

        metrics = {
            "accuracy":    round(acc, 4),
            "precision":   round(prec, 4),
            "recall":      round(rec, 4),
            "f1_score":    round(f1, 4),
            "sharpe":      round(sharpe, 4),
        }

        if pnl_val is not None and len(pnl_val) == len(y):
            pnl = np.array(pnl_val)
            baseline_expectancy = float(pnl.mean())

            approved = probas >= AI.min_confidence
            n_approved = int(approved.sum())
            if n_approved >= 20:   # need a minimally meaningful sample
                approved_expectancy = float(pnl[approved].mean())
                approved_win_rate = float((y[approved] == 1).mean())
            else:
                approved_expectancy = None
                approved_win_rate = None

            metrics.update({
                "baseline_expectancy_pct": round(baseline_expectancy * 100, 4),
                "approved_trade_count":    n_approved,
                "approved_win_rate":       round(approved_win_rate, 4) if approved_win_rate is not None else None,
                "approved_expectancy_pct": round(approved_expectancy * 100, 4) if approved_expectancy is not None else None,
            })

        return metrics
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {}


def should_deploy(new_metrics: dict, current_accuracy: float | None) -> tuple[bool, str]:
    """
    Decides whether a new model should replace the current one.

    Gates on THREE things now, not just accuracy:
      1. Minimum classification accuracy (sanity floor)
      2. Improvement over the current model's accuracy
      3. THE ACTUAL JOB: trades the AI would approve (confidence >=
         AI.min_confidence) must have HIGHER expectancy than the
         unfiltered baseline in the held-out validation set. A model
         that's accurate in general but doesn't improve on baseline when
         used as a filter isn't doing its job in this system, regardless
         of how good its accuracy number looks.

    Returns (deploy, reason).
    """
    acc = new_metrics.get("accuracy", 0.0)

    if acc < MIN_ACCURACY:
        return False, (
            f"Accuracy {acc:.2%} below minimum {MIN_ACCURACY:.2%}"
        )

    if current_accuracy is not None:
        improvement = acc - current_accuracy
        if improvement < MIN_IMPROVEMENT:
            return False, (
                f"Improvement {improvement:.2%} below threshold "
                f"{MIN_IMPROVEMENT:.2%} — keeping current model"
            )

    approved_expectancy = new_metrics.get("approved_expectancy_pct")
    baseline_expectancy = new_metrics.get("baseline_expectancy_pct")
    approved_count = new_metrics.get("approved_trade_count", 0)

    if approved_expectancy is not None and baseline_expectancy is not None:
        if approved_expectancy <= baseline_expectancy:
            return False, (
                f"AI-approved trades (n={approved_count}) expectancy "
                f"{approved_expectancy:+.3f}% does NOT beat baseline "
                f"{baseline_expectancy:+.3f}% — model doesn't improve on "
                f"just taking every signal, not deploying"
            )
        if approved_count < 20:
            return False, (
                f"Only {approved_count} validation trades would be AI-approved "
                f"— too few to trust the expectancy comparison, not deploying"
            )
    else:
        logger.warning(
            "should_deploy() called without pnl_val — skipping the "
            "trading-relevant expectancy check. Pass pnl_val to evaluate() "
            "to enable it; deploying on accuracy alone is a weaker guarantee."
        )

    return True, (
        f"Model approved: accuracy={acc:.2%} f1={new_metrics.get('f1_score', 0):.2%} "
        f"approved-trade expectancy={approved_expectancy:+.3f}% "
        f"(vs baseline {baseline_expectancy:+.3f}%)"
        if approved_expectancy is not None else
        f"Model approved: accuracy={acc:.2%} f1={new_metrics.get('f1_score', 0):.2%}"
    )


def _pseudo_sharpe(probas: np.ndarray, labels: np.ndarray) -> float:
    """
    Rough quality signal: difference in average confidence between
    correctly predicted wins and everything else.
    """
    try:
        wins  = probas[labels == 1]
        other = probas[labels == 0]
        if len(wins) == 0 or len(other) == 0:
            return 0.0
        diff = wins.mean() - other.mean()
        std  = probas.std()
        return float(diff / std) if std > 0 else 0.0
    except Exception:
        return 0.0


async def get_current_model_accuracy() -> float | None:
    """Fetches the accuracy of the currently active model from Supabase."""
    try:
        client = get_client()
        result = (
            client.table("model_versions")
            .select("accuracy")
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        if result.data:
            return float(result.data[0]["accuracy"])
        return None
    except Exception as e:
        logger.error(f"Failed to fetch current model accuracy: {e}")
        return None