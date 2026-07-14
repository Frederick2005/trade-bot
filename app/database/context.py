"""
app/database/context.py
Saves trade context, decision log, training labels, model versions.

WHY FULL CONTEXT MATTERS:
  Every field saved here becomes a feature the AI can learn from.
  The more context we capture at entry, the better the model
  understands which conditions produce wins vs losses.
  
  New fields added:
  - MACD histogram → tells AI if momentum is accelerating or decelerating
  - ADX           → tells AI how strong the trend is (>25 = strong)
  - BB width      → tells AI if market is expanding or contracting
  - Supertrend    → confirms or denies the EMA trend signal
  - Session name  → your data proves NY session wins 57-66% vs 44% other
  - Market regime → AI learns different strategies work in different regimes
  - Stoch RSI     → finer momentum detail than plain RSI
"""
import json
import math
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from app.database.client import get_client


def _session_name(hour: int) -> str:
    """
    Classify trading session by UTC hour.
    WHY: Your data shows NY session (14-19 UTC) wins 57-66%.
    AI learns session is one of the most predictive features.
    """
    if 0 <= hour < 7:
        return "ASIAN"
    elif 7 <= hour < 13:
        return "LONDON"
    elif 13 <= hour < 19:
        return "NEW_YORK"
    elif 19 <= hour < 22:
        return "LONDON_NY_CLOSE"
    else:
        return "ASIAN_OPEN"


def _clean_for_json(obj):
    """Recursively convert numpy/non-serializable types to plain Python."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean_for_json(i) for i in obj]
    return str(obj)


async def save_trade_context(trade_id: str, context: dict) -> bool:
    try:
        now      = datetime.now(timezone.utc)
        hour     = context.get("hour_of_day", now.hour)
        weekday  = context.get("day_of_week", now.weekday())
        month    = now.month
        quarter  = (month - 1) // 3 + 1
        week_num = now.isocalendar()[1]

        client = get_client()
        client.table("trade_context").insert({
            "trade_id":        trade_id,
            # Core indicators
            "ema50_1h":        context.get("ema50_1h"),
            "ema200_1h":       context.get("ema200_1h"),
            "rsi_1h":          context.get("rsi_1h"),
            "atr_1h":          context.get("atr_1h"),
            "ema50_4h":        context.get("ema50_4h"),
            "ema200_4h":       context.get("ema200_4h"),
            "rsi_4h":          context.get("rsi_4h"),
            "atr_4h":          context.get("atr_4h"),
            # Derived
            "price_vs_ema50":  context.get("price_vs_ema50"),
            "trend_strength":  context.get("trend_strength"),
            "volatility_pct":  context.get("volatility_pct"),
            "volume_ratio":    context.get("volume_ratio"),
            "ema_gap_pct":     context.get("ema_gap_pct"),
            "candle_body_pct": context.get("candle_body_pct"),
            "rsi_divergence":  context.get("rsi_divergence"),
            # Time
            "hour_of_day":     hour,
            "day_of_week":     weekday,
            "trend_4h":        context.get("trend_4h"),
            # NEW — extended indicators
            "macd":            context.get("macd"),
            "macd_signal":     context.get("macd_signal"),
            "macd_histogram":  context.get("macd_histogram"),
            "stoch_rsi":       context.get("stoch_rsi_k"),
            "adx":             context.get("adx"),
            "cci":             context.get("cci"),
            "obv":             context.get("obv"),
            "bb_upper":        context.get("bb_upper"),
            "bb_lower":        context.get("bb_lower"),
            "bb_width":        context.get("bb_width"),
            "supertrend":      context.get("supertrend"),
            "supertrend_dir":  context.get("supertrend_dir"),
            "vwap_1h":         context.get("vwap"),
            "realized_vol":    context.get("realized_vol"),
            "atr_percentile":  context.get("atr_percentile"),
            "market_regime":   context.get("market_regime"),
            # NEW — session + time features
            "session_name":    _session_name(hour),
            "is_weekend":      weekday >= 5,
            "is_month_end":    now.day >= 28,
            "is_quarter_end":  (month in [3, 6, 9, 12]) and now.day >= 28,
            "week_of_year":    week_num,
            "quarter":         quarter,
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
        # Full numpy-safe JSON conversion
        clean_features = _clean_for_json(features)
        clean_features = json.loads(json.dumps(clean_features))

        client = get_client()
        client.table("training_labels").insert({
            "trade_id":   trade_id,
            "features":   clean_features,
            "label":      int(label),
            "pnl_pct":    float(pnl_pct) if not math.isnan(float(pnl_pct)) else 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.debug(f"Training label saved: trade_id={trade_id} label={label}")
        return True
    except Exception as e:
        logger.error(f"Failed to save training label: {e}")
        return False


async def get_training_labels(min_count: int = 0) -> list[dict]:
    """Fetch ALL training labels using pagination — no 1000 row limit."""
    try:
        client   = get_client()
        all_data = []
        batch    = 1000
        offset   = 0

        while True:
            result = (
                client.table("training_labels")
                .select("features,label,pnl_pct,created_at")
                .order("created_at")
                .range(offset, offset + batch - 1)
                .execute()
            )
            chunk = result.data or []
            all_data.extend(chunk)
            logger.info(f"Fetched {len(all_data)} training labels so far...")
            if len(chunk) < batch:
                break
            offset += batch

        if len(all_data) < min_count:
            logger.warning(f"Only {len(all_data)} labels, need {min_count}")
            return []

        logger.info(f"Total training labels loaded: {len(all_data)}")
        return all_data
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
        client.table("model_versions").update(
            {"is_active": False}
        ).eq("is_active", True).execute()
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
        logger.info(f"Model version saved and activated: {version} accuracy={accuracy:.2%}")
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


async def save_error_analysis(
    trade_id: str,
    symbol: str,
    error_type: str,
    confidence: float,
    description: str,
    context: dict,
) -> bool:
    """
    Auto-classify every losing trade.
    WHY: Pattern of errors tells us what to fix in the strategy.
    If 40% of losses are STOP_TOO_TIGHT, we increase ATR multiplier.
    If 30% are LOW_VOLUME, we raise the volume ratio threshold.
    """
    try:
        client = get_client()
        client.table("error_analysis").insert({
            "trade_id":       trade_id,
            "symbol":         symbol,
            "error_type":     error_type,
            "confidence":     confidence,
            "description":    description,
            "rsi_at_entry":   context.get("rsi_1h"),
            "volume_at_entry":context.get("volume_ratio"),
            "volatility":     context.get("volatility_pct"),
            "market_regime":  context.get("market_regime"),
            "auto_classified":True,
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save error analysis: {e}")
        return False


async def save_ai_prediction_log(
    trade_id: Optional[str],
    model_version: str,
    confidence: float,
    probability_win: float,
    chosen_action: str,
    feature_vector: dict,
    feature_importance: dict,
    inference_time_ms: float = 0.0,
) -> bool:
    """
    Store every AI prediction.
    WHY: Allows us to track model accuracy over time, detect drift,
    and compare model versions on the same market conditions.
    """
    try:
        clean_fv = _clean_for_json(feature_vector)
        clean_fi = _clean_for_json(feature_importance)
        clean_fv = json.loads(json.dumps(clean_fv))
        clean_fi = json.loads(json.dumps(clean_fi))

        client = get_client()
        client.table("ai_prediction_logs").insert({
            "trade_id":          trade_id,
            "model_version":     model_version,
            "prediction_time":   datetime.now(timezone.utc).isoformat(),
            "confidence":        float(confidence),
            "probability_win":   float(probability_win),
            "probability_loss":  float(1 - probability_win),
            "chosen_action":     chosen_action,
            "feature_vector":    clean_fv,
            "feature_importance":clean_fi,
            "inference_time_ms": float(inference_time_ms),
            "created_at":        datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save AI prediction log: {e}")
        return False


async def update_prediction_outcome(trade_id: str, outcome: str, result: str) -> bool:
    """Update prediction log when trade closes — marks correct/incorrect."""
    try:
        client = get_client()
        client.table("ai_prediction_logs").update({
            "prediction_outcome": outcome,
            "actual_result":      result,
            "outcome_recorded_at":datetime.now(timezone.utc).isoformat(),
        }).eq("trade_id", trade_id).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to update prediction outcome: {e}")
        return False


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