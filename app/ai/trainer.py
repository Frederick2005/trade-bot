import os
from datetime import datetime, timezone
import numpy as np
import joblib
from xgboost import XGBClassifier
from loguru import logger
from app.config import AI
from app.ai.labels import get_all_labels, get_label_stats
from app.ai.validator import evaluate, should_deploy, get_current_model_accuracy
from app.database.context import save_model_version


async def run_training() -> dict | None:
    """
    Full training pipeline:
      1. Load all labelled trades from Supabase
      2. Check we have enough data
      3. Time-ordered 80/20 train/val split
      4. Train XGBoost
      5. Validate — must beat current model
      6. Save and activate if approved

    Returns metrics dict on success, None on failure/skip.
    """
    logger.info("=" * 40)
    logger.info("AI training run started")

    # ── 1. Check data volume ───────────────────────────────────────
    stats = await get_label_stats()
    total = stats["total"]
    logger.info(
        f"Training data: {total} trades | "
        f"wins={stats['wins']} losses={stats['losses']} "
        f"win_rate={stats['win_rate']:.1%}"
    )

    if total < AI.min_trades_to_train:
        logger.warning(
            f"Not enough data: {total}/{AI.min_trades_to_train} trades. "
            f"Skipping training."
        )
        return None

    # ── 2. Load features and labels ────────────────────────────────
    X, y = await get_all_labels()
    if not X:
        logger.error("No training data returned — aborting")
        return None

    X = np.array(X)
    y = np.array(y)

    # ── 3. Time-ordered train/val split (no data leakage) ──────────
    split      = int(len(X) * 0.80)
    X_train    = X[:split]
    y_train    = y[:split]
    X_val      = X[split:]
    y_val      = y[split:]

    logger.info(
        f"Split: train={len(X_train)} val={len(X_val)} "
        f"(80/20 time-ordered)"
    )

    # ── 4. Train XGBoost ───────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    logger.info("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── 5. Validate ────────────────────────────────────────────────
    metrics         = evaluate(model, X_val.tolist(), y_val.tolist())
    current_acc     = await get_current_model_accuracy()
    deploy, reason  = should_deploy(metrics, current_acc)

    logger.info(
        f"Validation: acc={metrics.get('accuracy', 0):.2%} "
        f"f1={metrics.get('f1_score', 0):.2%} "
        f"precision={metrics.get('precision', 0):.2%}"
    )
    logger.info(f"Deploy decision: {deploy} — {reason}")

    if not deploy:
        return None

    # ── 6. Save model file ─────────────────────────────────────────
    os.makedirs(AI.model_dir, exist_ok=True)
    version    = _next_version()
    model_path = os.path.join(AI.model_dir, f"{version}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # ── 7. Register in Supabase ────────────────────────────────────
    await save_model_version(
        version=version,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        trained_on=len(X_train),
        model_path=model_path,
        notes=reason,
    )

    # ── 8. Load the new model into memory ─────────────────────────
    from app.ai import model as ai_model
    ai_model.load_model(model_path, version)

    logger.info(f"New model activated: {version}")
    logger.info("=" * 40)

    return {**metrics, "version": version, "trained_on": len(X_train)}


def _next_version() -> str:
    """Generates a version string like v1.0, v1.1, v2.0 based on existing files."""
    try:
        from app.database.client import get_client
        client  = get_client()
        result  = (
            client.table("model_versions")
            .select("version")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            last    = result.data[0]["version"]   # e.g. 'v1.3'
            parts   = last.lstrip("v").split(".")
            major   = int(parts[0])
            minor   = int(parts[1]) + 1
            return f"v{major}.{minor}"
    except Exception:
        pass
    return "v1.0"