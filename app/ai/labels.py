from loguru import logger
from app.database.client import get_client
from app.database.context import save_training_label


async def label_closed_trade(
    trade_id: str,
    profit_loss: float,
    profit_pct: float,
    features: dict,
) -> bool:
    """
    Converts a closed trade into a binary training label.
    label = 1 if the trade was profitable, 0 if it was a loss.
    Saves the label + feature vector to the training_labels table.
    """
    try:
        label = 1 if profit_loss > 0 else 0
        success = await save_training_label(
            trade_id=trade_id,
            features=features,
            label=label,
            pnl_pct=profit_pct,
        )
        logger.info(
            f"Trade labelled: id={trade_id} "
            f"label={label} pnl={profit_loss:+.4f}"
        )
        return success
    except Exception as e:
        logger.error(f"Failed to label trade {trade_id}: {e}")
        return False


async def get_all_labels() -> tuple[list[list[float]], list[int], list[float]]:
    """
    Fetches all training labels from Supabase.
    Returns (X, y, pnl) where:
        X   = list of feature vectors (each a list of floats)
        y   = list of labels (0 or 1)
        pnl = list of pnl_pct values, same order — needed by
              app.ai.validator.evaluate() to check whether AI-approved
              trades actually beat baseline expectancy, not just whether
              the classifier is accurate in the abstract.
    Ordered by created_at to preserve time order for train/val split.
    """
    from app.ai.features import vector_to_list

    try:
        client = get_client()
        result = (
            client.table("training_labels")
            .select("features,label,pnl_pct")
            .order("created_at")
            .execute()
        )
        data = result.data or []

        if not data:
            logger.warning("No training labels found in database")
            return [], [], []

        X, y, pnl = [], [], []
        for row in data:
            try:
                vec = vector_to_list(row["features"])
                X.append(vec)
                y.append(int(row["label"]))
                pnl.append(float(row.get("pnl_pct", 0.0)))
            except Exception as e:
                logger.warning(f"Skipping malformed label row: {e}")
                continue

        logger.info(f"Loaded {len(X)} training examples | wins={sum(y)} losses={len(y)-sum(y)}")
        return X, y, pnl

    except Exception as e:
        logger.error(f"Failed to load training labels: {e}")
        return [], [], []


async def get_label_stats() -> dict:
    """Returns basic stats about the training set."""
    try:
        client = get_client()
        result = (
            client.table("training_labels")
            .select("label,pnl_pct")
            .execute()
        )
        data = result.data or []
        if not data:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        total  = len(data)
        wins   = sum(1 for r in data if r["label"] == 1)
        losses = total - wins
        return {
            "total":    total,
            "wins":     wins,
            "losses":   losses,
            "win_rate": wins / total if total else 0.0,
        }
    except Exception as e:
        logger.error(f"Failed to fetch label stats: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}