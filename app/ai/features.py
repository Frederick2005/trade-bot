from datetime import datetime


def _safe_pct(numerator, denominator) -> float:
    """Returns (numerator / denominator) * 100, or 0.0 if denominator is None/zero."""
    if not denominator:
        return 0.0
    return (numerator - denominator) / denominator * 100


def build_feature_vector(
    indicators_1h: dict,
    indicators_4h: dict,
    candle_time: datetime,
    recent_win_rate: float = 0.5,
    current_drawdown: float = 0.0,
) -> dict:
    """
    Converts raw indicator snapshots into a flat feature vector
    for the ML model. All features are numeric.
    """
    price     = indicators_1h["price"]
    ema50_1h  = indicators_1h["ema50"]
    ema200_1h = indicators_1h["ema200"]
    rsi_1h    = indicators_1h["rsi"]
    atr_1h    = indicators_1h["atr"]

    ema50_4h  = indicators_4h["ema50"]
    ema200_4h = indicators_4h["ema200"]
    rsi_4h    = indicators_4h["rsi"]

    return {
        # ── Trend features ──────────────────────────────────────────
        "ema_gap_pct_1h":    _safe_pct(ema50_1h, ema200_1h),
        "price_vs_ema50":    _safe_pct(price, ema50_1h),
        "ema50_slope":       indicators_1h.get("ema50_slope", 0.0),
        "trend_4h":          1.0 if (ema50_4h and ema200_4h and ema50_4h > ema200_4h) else -1.0,
        "ema_gap_pct_4h":    _safe_pct(ema50_4h, ema200_4h),

        # ── Momentum features ───────────────────────────────────────
        "rsi_1h":            rsi_1h,
        "rsi_4h":            rsi_4h,
        "rsi_divergence":    rsi_1h - rsi_4h,
        "rsi_vs_midpoint":   rsi_1h - 52.5,

        # ── Volatility features ─────────────────────────────────────
        "atr_pct":           (atr_1h / price * 100) if price else 0.0,
        "volatility_pct":    indicators_1h.get("volatility_pct", 0.0),
        "candle_body_pct":   indicators_1h.get("candle_body_pct", 0.0),

        # ── Volume features ─────────────────────────────────────────
        "volume_ratio":      indicators_1h.get("volume_ratio", 1.0),

        # ── Market structure ────────────────────────────────────────
        "trend_strength":    indicators_1h.get("trend_strength", 0.0),

        # ── Time context ────────────────────────────────────────────
        "hour_utc":          float(candle_time.hour),
        "day_of_week":       float(candle_time.weekday()),

        # ── Recent bot performance ──────────────────────────────────
        "win_rate_recent":   recent_win_rate,
        "current_drawdown":  current_drawdown,

        # ── Direction encoding ──────────────────────────────────────
        "is_bullish_candle": 1.0 if indicators_1h.get("is_bullish_candle") else -1.0,
    }


FEATURE_COLUMNS = [
    "ema_gap_pct_1h", "price_vs_ema50", "ema50_slope", "trend_4h",
    "ema_gap_pct_4h", "rsi_1h", "rsi_4h", "rsi_divergence",
    "rsi_vs_midpoint", "atr_pct", "volatility_pct", "candle_body_pct",
    "volume_ratio", "trend_strength", "hour_utc", "day_of_week",
    "win_rate_recent", "current_drawdown", "is_bullish_candle",
]


def vector_to_list(features: dict) -> list[float]:
    return [features.get(col, 0.0) for col in FEATURE_COLUMNS]