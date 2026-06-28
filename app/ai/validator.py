import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from loguru import logger
from app.database.client import get_client


# New model must beat this accuracy to be deployed
MIN_ACCURACY = 0.55

# Minimum improvement over current model to justify replacement
MIN_IMPROVEMENT = 0.02   # 2 percentage points


def evaluate(
    model,
    X_val: list[list[float]],
    y_val: list[int],
) -> dict:
    """Runs the model on validation data and returns all metrics."""
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

        return {
            "accuracy":    round(acc, 4),
            "precision":   round(prec, 4),
            "recall":      round(rec, 4),
            "f1_score":    round(f1, 4),
            "sharpe":      round(sharpe, 4),
        }
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {}


def should_deploy(new_metrics: dict, current_accuracy: float | None) -> tuple[bool, str]:
    """
    Decides whether a new model should replace the current one.
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

    return True, (
        f"Model approved: accuracy={acc:.2%} "
        f"f1={new_metrics.get('f1_score', 0):.2%}"
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