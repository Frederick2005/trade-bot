from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from app.database.client import get_client


async def save_trade_context(trade_id: str, context: dict) -> bool:
    try:
        client = get_client()
        client.table("trade_context").insert({
            "trade_id":       trade_id,
            "ema50_1h":       context.get("ema50_1h"),
            "ema200_1h":      context.get("ema200_1h"),
            "rsi_1h":         context.get("rsi_1h"),
            "atr_1h":         context.get("atr_1h"),
            "ema50_4h":       context.get("ema50_4h"),
            "ema200_4h":      context.get("ema200_4h"),
            "rsi_4h":         context.get("rsi_4h"),
            "atr_4h":         context.get("atr_4h"),
            "price_vs_ema50": context.get("price_vs_ema50"),
            "trend_strength": context.get("trend_strength"),
            "volatility_pct": context.get("volatility_pct"),
            "volume_ratio":   context.get("volume_ratio"),
            "ema_gap_pct":    context.get("ema_gap_pct"),
            "candle_body_pct":context.get("candle_body_pct"),
            "rsi_divergence": context.get("rsi_divergence"),
            "hour_of_day":    context.get("hour_of_day"),
            "day_of_week":    context.get("day_of_week"),
            "trend_4h":       context.get("trend_4h"),
        }).execute()
        logger.debug(f"Trade context saved for trade_id={trade_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save trade context: {e}")
        return False


async def log_decision(
    symbol: str,
    action: str,
    reason: str,
    signal_type: Optional[str] = None,
    rsi: Optional[float] = None,
    atr: Optional[float] = None,
    confidence: Optional[float] = None,
) -> bool:
    try:
        client = get_client()
        client.table("decision_log").insert({
            "symbol":        symbol,
            "signal_type":   signal_type,
            "action":        action,
            "reason":        reason,
            "rsi_at_signal": rsi,
            "atr_at_signal": atr,
            "confidence":    confidence,
            "created_at":    datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to log decision: {e}")
        return False


async def save_training_label(
    trade_id: str,
    features: dict,
    label: int,
    pnl_pct: float,
) -> bool:
    try:
        client = get_client()
        client.table("training_labels").insert({
            "trade_id":   trade_id,
            "features":   features,
            "label":      label,
            "pnl_pct":    pnl_pct,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.debug(f"Training label saved: trade_id={trade_id} label={label}")
        return True
    except Exception as e:
        logger.error(f"Failed to save training label: {e}")
        return False


async def get_training_labels(min_count: int = 0) -> list[dict]:
    try:
        client = get_client()
        result = (
            client.table("training_labels")
            .select("features,label,pnl_pct,created_at")
            .order("created_at")
            .execute()
        )
        data = result.data or []
        if len(data) < min_count:
            logger.warning(
                f"Only {len(data)} training labels available, need {min_count}"
            )
            return []
        return data
    except Exception as e:
        logger.error(f"Failed to fetch training labels: {e}")
        return []


async def save_model_version(
    version: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    trained_on: int,
    model_path: str,
    notes: str = "",
) -> bool:
    try:
        client = get_client()
        # deactivate all existing versions first
        client.table("model_versions").update(
            {"is_active": False}
        ).eq("is_active", True).execute()
        # insert new active version
        client.table("model_versions").insert({
            "version":    version,
            "is_active":  True,
            "accuracy":   accuracy,
            "precision":  precision,
            "recall":     recall,
            "f1_score":   f1_score,
            "trained_on": trained_on,
            "model_path": model_path,
            "notes":      notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.info(f"Model version saved and activated: {version} | accuracy={accuracy:.2%}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model version: {e}")
        return False


async def get_active_model_version() -> Optional[dict]:
    try:
        client = get_client()
        result = (
            client.table("model_versions")
            .select("*")
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Failed to fetch active model version: {e}")
        return None


async def log_bot_event(level: str, message: str, context: Optional[dict] = None) -> None:
    try:
        client = get_client()
        client.table("bot_logs").insert({
            "level":      level,
            "message":    message,
            "context":    context,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error(f"Failed to write bot log to Supabase: {e}")