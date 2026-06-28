from datetime import datetime, timezone
from loguru import logger
from app.database.client import get_client
from app.strategy.params import get_active_params, refresh, StrategyParams


# Minimum trades before tuner is allowed to act
MIN_TRADES_TO_TUNE = 20

# Win rate below this triggers a tighten of entry conditions
WIN_RATE_LOW_THRESHOLD  = 0.40

# Win rate above this allows a slight relaxation of conditions
WIN_RATE_HIGH_THRESHOLD = 0.65


async def maybe_tune() -> bool:
    """
    Checks recent performance and adjusts strategy parameters if needed.
    Returns True if a new parameter version was created.
    """
    from app.learning.queries import overall_performance

    stats = await overall_performance(days=14)
    if not stats:
        return False

    total    = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0.5)

    if total < MIN_TRADES_TO_TUNE:
        logger.info(
            f"Tuner: only {total} trades in last 14 days "
            f"(need {MIN_TRADES_TO_TUNE}) — skipping"
        )
        return False

    params = get_active_params()
    logger.info(
        f"Tuner evaluating: win_rate={win_rate:.1%} "
        f"trades={total} current_version={params.version}"
    )

    if win_rate < WIN_RATE_LOW_THRESHOLD:
        return await _tighten_conditions(params, win_rate, total)

    if win_rate > WIN_RATE_HIGH_THRESHOLD:
        return await _relax_conditions(params, win_rate, total)

    logger.info(f"Tuner: win_rate {win_rate:.1%} is healthy — no change")
    return False


async def _tighten_conditions(
    params: StrategyParams,
    win_rate: float,
    trade_count: int,
) -> bool:
    """Win rate too low — make entry conditions stricter."""
    new_rsi_lower = min(params.rsi_lower + 2, 52)
    new_rsi_upper = max(params.rsi_upper - 2, 54)

    if new_rsi_lower >= new_rsi_upper:
        logger.warning("Tuner: RSI window already at minimum — cannot tighten further")
        return False

    reason = (
        f"Win rate {win_rate:.1%} < {WIN_RATE_LOW_THRESHOLD:.1%} "
        f"over {trade_count} trades — tightening RSI window "
        f"from [{params.rsi_lower}-{params.rsi_upper}] "
        f"to [{new_rsi_lower}-{new_rsi_upper}]"
    )
    return await _create_new_version(params, new_rsi_lower, new_rsi_upper, reason)


async def _relax_conditions(
    params: StrategyParams,
    win_rate: float,
    trade_count: int,
) -> bool:
    """Win rate very high — slightly widen entry conditions for more signals."""
    new_rsi_lower = max(params.rsi_lower - 1, 40)
    new_rsi_upper = min(params.rsi_upper + 1, 65)

    reason = (
        f"Win rate {win_rate:.1%} > {WIN_RATE_HIGH_THRESHOLD:.1%} "
        f"over {trade_count} trades — relaxing RSI window "
        f"from [{params.rsi_lower}-{params.rsi_upper}] "
        f"to [{new_rsi_lower}-{new_rsi_upper}]"
    )
    return await _create_new_version(params, new_rsi_lower, new_rsi_upper, reason)


async def _create_new_version(
    current: StrategyParams,
    new_rsi_lower: float,
    new_rsi_upper: float,
    reason: str,
) -> bool:
    try:
        client  = get_client()
        version = _next_param_version(current.version)

        # Deactivate current version
        client.table("param_history").update(
            {"is_active": False}
        ).eq("is_active", True).execute()

        # Insert new version
        client.table("param_history").insert({
            "version":          version,
            "is_active":        True,
            "rsi_lower":        new_rsi_lower,
            "rsi_upper":        new_rsi_upper,
            "atr_multiplier":   current.atr_multiplier,
            "ema_gap_min":      current.ema_gap_min,
            "min_volume_ratio": current.min_volume_ratio,
            "min_rr":           current.min_rr,
            "reason":           reason,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        }).execute()

        # Reload params into memory
        refresh()
        logger.info(f"New strategy version created: {version} — {reason}")
        return True

    except Exception as e:
        logger.error(f"Tuner failed to create new param version: {e}")
        return False


def _next_param_version(current: str) -> str:
    try:
        parts = current.lstrip("v").split(".")
        major = int(parts[0])
        minor = int(parts[1]) + 1
        return f"v{major}.{minor}"
    except Exception:
        return "v1.1"