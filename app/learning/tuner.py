"""
app/learning/tuner.py
Adjusts strategy parameters based on error analysis + win rate.

HOW DATA-DRIVEN TUNING WORKS:
──────────────────────────────
Instead of guessing what to change, the tuner reads the error_analysis
table and makes targeted adjustments:

  If >30% of losses are HIGH_VOLATILITY_LOSS  → raise ATR multiplier
  If >30% are LOW_VOLUME_LOSS                 → raise volume threshold
  If >30% are STOP_TOO_TIGHT                  → raise ATR multiplier
  If >30% are RANGING_MARKET                  → raise ADX threshold
  If >30% are RSI_OVERBOUGHT/OVERSOLD         → tighten RSI zone
  If win_rate < 40%                           → tighten all conditions
  If win_rate > 65%                           → slightly relax to get more signals

This is self-improving parameter management driven by real outcomes.
"""
from datetime import datetime, timezone
from loguru import logger
from app.database.client import get_client
from app.strategy.params import get_active_params, refresh, StrategyParams
from app.learning.analyser import get_error_summary

MIN_TRADES_TO_TUNE  = 20
WIN_RATE_LOW        = 0.40
WIN_RATE_HIGH       = 0.65
ERROR_TYPE_THRESHOLD = 0.30   # >30% of losses = systemic problem


async def maybe_tune() -> bool:
    from app.learning.queries import overall_performance
    stats = await overall_performance(days=14)
    if not stats:
        return False

    total    = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0.5)

    if total < MIN_TRADES_TO_TUNE:
        logger.info(f"Tuner: {total}/{MIN_TRADES_TO_TUNE} trades — skipping")
        return False

    params   = get_active_params()
    losses   = total - int(total * win_rate)
    errors   = await get_error_summary(last_n=min(losses, 100))

    logger.info(
        f"Tuner: win_rate={win_rate:.1%} total={total} "
        f"errors={errors} version={params.version}"
    )

    # ── Targeted fixes based on error patterns ────────────────────
    changes  = {}
    reasons  = []

    total_errors = sum(errors.values()) or 1

    if errors.get("HIGH_VOLATILITY_LOSS", 0) / total_errors > ERROR_TYPE_THRESHOLD:
        changes["atr_multiplier"] = min(params.atr_multiplier + 0.3, 3.5)
        reasons.append(f"ATR raised: {errors['HIGH_VOLATILITY_LOSS']} high-vol losses")

    if errors.get("LOW_VOLUME_LOSS", 0) / total_errors > ERROR_TYPE_THRESHOLD:
        changes["min_volume_ratio"] = min(params.min_volume_ratio + 0.1, 1.8)
        reasons.append(f"Volume threshold raised: {errors['LOW_VOLUME_LOSS']} low-vol losses")

    if errors.get("STOP_TOO_TIGHT", 0) / total_errors > ERROR_TYPE_THRESHOLD:
        changes["atr_multiplier"] = min(
            changes.get("atr_multiplier", params.atr_multiplier) + 0.2, 3.5
        )
        reasons.append(f"ATR raised for tight stops: {errors['STOP_TOO_TIGHT']} early SL hits")

    if (errors.get("RSI_OVERBOUGHT_ENTRY", 0) + errors.get("RSI_OVERSOLD_ENTRY", 0)) / total_errors > ERROR_TYPE_THRESHOLD:
        changes["rsi_upper"] = max(params.rsi_upper - 2, 56)
        changes["rsi_lower"] = min(params.rsi_lower + 1, 46)
        reasons.append("RSI zone tightened: too many RSI extreme losses")

    # ── Overall win rate adjustments ──────────────────────────────
    if win_rate < WIN_RATE_LOW and not changes:
        changes["rsi_lower"] = min(params.rsi_lower + 2, 52)
        changes["rsi_upper"] = max(params.rsi_upper - 2, 54)
        reasons.append(f"Win rate {win_rate:.1%} too low — tightening RSI")

    if win_rate > WIN_RATE_HIGH and not changes:
        changes["rsi_lower"] = max(params.rsi_lower - 1, 40)
        changes["rsi_upper"] = min(params.rsi_upper + 1, 65)
        reasons.append(f"Win rate {win_rate:.1%} strong — relaxing RSI for more signals")

    if not changes:
        logger.info(f"Tuner: no changes needed at {win_rate:.1%} win rate")
        return False

    return await _create_new_version(params, changes, " | ".join(reasons))


async def _create_new_version(
    current: StrategyParams,
    changes: dict,
    reason: str,
) -> bool:
    try:
        client  = get_client()
        version = _next_version(current.version)

        client.table("param_history").update(
            {"is_active": False}
        ).eq("is_active", True).execute()

        client.table("param_history").insert({
            "version":           version,
            "is_active":         True,
            "rsi_lower":         changes.get("rsi_lower", current.rsi_lower),
            "rsi_upper":         changes.get("rsi_upper", current.rsi_upper),
            "atr_multiplier":    changes.get("atr_multiplier", current.atr_multiplier),
            "ema_gap_min":       current.ema_gap_min,
            "min_volume_ratio":  changes.get("min_volume_ratio", current.min_volume_ratio),
            "min_rr":            changes.get("min_rr", current.min_rr),
            "reason":            reason,
            "created_at":        datetime.now(timezone.utc).isoformat(),
        }).execute()

        refresh()
        logger.info(f"New strategy version: {version} — {reason}")
        return True
    except Exception as e:
        logger.error(f"Tuner failed to create new version: {e}")
        return False


def _next_version(current: str) -> str:
    try:
        parts = current.lstrip("v").split(".")
        return f"v{parts[0]}.{int(parts[1]) + 1}"
    except Exception:
        return "v1.1"