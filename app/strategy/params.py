from dataclasses import dataclass
from loguru import logger
from app.database.client import get_client
from app.config import STRATEGY


@dataclass
class StrategyParams:
    version: str
    rsi_lower: float
    rsi_upper: float
    atr_multiplier: float
    ema_gap_min: float
    min_volume_ratio: float
    min_rr: float


_cached: StrategyParams | None = None


def get_active_params() -> StrategyParams:
    global _cached
    if _cached is not None:
        return _cached
    _cached = _load_from_db()
    return _cached


def refresh() -> StrategyParams:
    global _cached
    _cached = _load_from_db()
    logger.info(f"Strategy params refreshed: {_cached.version}")
    return _cached


def _load_from_db() -> StrategyParams:
    try:
        client = get_client()
        result = (
            client.table("param_history")
            .select("*")
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            return StrategyParams(
                version=row["version"],
                rsi_lower=float(row["rsi_lower"]),
                rsi_upper=float(row["rsi_upper"]),
                atr_multiplier=float(row["atr_multiplier"]),
                ema_gap_min=float(row["ema_gap_min"]),
                min_volume_ratio=float(row["min_volume_ratio"]),
                min_rr=float(row["min_rr"]),
            )
    except Exception as e:
        logger.warning(f"Could not load params from DB, using defaults: {e}")

    # Fallback to config defaults
    return StrategyParams(
        version="v1.0-default",
        rsi_lower=STRATEGY.rsi_lower,
        rsi_upper=STRATEGY.rsi_upper,
        atr_multiplier=STRATEGY.atr_multiplier,
        ema_gap_min=0.0,
        min_volume_ratio=1.0,
        min_rr=STRATEGY.min_risk_reward,
    )