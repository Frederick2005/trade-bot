"""
app/ai/features.py
Converts raw indicator snapshots into the ML feature vector.

WHY EACH FEATURE TEACHES THE AI TO TRADE BETTER:
───────────────────────────────────────────────────────
ema_gap_pct     → Size of EMA50/200 gap. Wider gap = stronger trend.
                  AI learns minimum gap needed for reliable signals.

price_vs_ema50  → How far price is from EMA50. Your data shows
                  pullbacks within 1% win more. AI learns exact
                  optimal distance.

macd_histogram  → Momentum acceleration. Positive and rising means
                  momentum is building. AI learns MACD confirms RSI.

stoch_rsi_k     → Finer RSI detail. Reveals overbought/oversold
                  within the RSI zone. AI catches setups RSI alone misses.

adx             → Trend strength 0-100. AI learns trades taken when
                  ADX > 25 have significantly higher win rates.

bb_width        → Market compression/expansion. Narrow BB = compression
                  before breakout. AI learns to anticipate direction.

supertrend_dir  → 1 = uptrend, -1 = downtrend. Simple but powerful.
                  AI learns supertrend alignment adds ~5% win rate.

buy_vol_ratio   → % of volume from buyers. >60% = buyers in control.
                  AI learns buy-dominated candles have better follow through.

session_ny      → NY session flag (14-19 UTC). Your data: 57-66% win rate
                  vs 44% other sessions. Highest impact single feature.

is_weekend      → Saturday trades lose 44.5% — well below breakeven.
                  AI learns to strongly penalise weekend signals.

recent_win_rate → Feedback loop. When bot is on a losing streak,
                  AI becomes more conservative. When winning, stays active.

current_drawdown → Risk awareness. AI learns to be more selective
                  when account is under stress.

market_regime   → TRENDING/RANGING/VOLATILE encoding. AI learns
                  EMA strategy works in trending, fails in ranging.
"""
from datetime import datetime, timezone


def build_feature_vector(
    indicators_1h: dict,
    indicators_4h: dict,
    candle_time: datetime,
    recent_win_rate: float = 0.5,
    current_drawdown: float = 0.0,
) -> dict:
    price     = indicators_1h["price"]
    ema50_1h  = indicators_1h["ema50"]
    ema200_1h = indicators_1h["ema200"]
    rsi_1h    = indicators_1h["rsi"]
    atr_1h    = indicators_1h["atr"]

    ema50_4h  = indicators_4h["ema50"]
    ema200_4h = indicators_4h["ema200"]
    rsi_4h    = indicators_4h["rsi"]

    hour    = candle_time.hour
    weekday = candle_time.weekday()

    # Encode market regime numerically
    regime = indicators_1h.get("market_regime", "NEUTRAL")
    regime_map = {
        "TRENDING_UP":   2, "TRENDING_DOWN": -2,
        "RANGING":       0, "VOLATILE":      -1, "NEUTRAL": 1
    }
    regime_num = regime_map.get(regime, 0)

    return {
        # ── Trend features ────────────────────────────────────────
        "ema_gap_pct_1h":      (ema50_1h - ema200_1h) / ema200_1h * 100,
        "price_vs_ema50":      (price - ema50_1h) / ema50_1h * 100,
        "ema50_slope":         indicators_1h.get("ema50_slope", 0.0),
        "trend_4h":            1.0 if ema50_4h > ema200_4h else -1.0,
        "ema_gap_pct_4h":      (ema50_4h - ema200_4h) / ema200_4h * 100,
        "supertrend_dir":      float(indicators_1h.get("supertrend_dir", 1)),
        "adx":                 indicators_1h.get("adx", 20.0),
        "adx_trending":        1.0 if indicators_1h.get("adx", 0) > 25 else 0.0,

        # ── Momentum features ─────────────────────────────────────
        "rsi_1h":              rsi_1h,
        "rsi_4h":              rsi_4h,
        "rsi_divergence":      rsi_1h - rsi_4h,
        "rsi_vs_midpoint":     rsi_1h - 52.5,
        "stoch_rsi_k":         indicators_1h.get("stoch_rsi_k", 50.0),
        "stoch_rsi_d":         indicators_1h.get("stoch_rsi_d", 50.0),
        "macd":                indicators_1h.get("macd", 0.0),
        "macd_histogram":      indicators_1h.get("macd_histogram", 0.0),
        "macd_signal":         indicators_1h.get("macd_signal", 0.0),
        "cci":                 indicators_1h.get("cci", 0.0),

        # ── Volatility features ───────────────────────────────────
        "atr_pct":             atr_1h / price * 100,
        "volatility_pct":      indicators_1h.get("volatility_pct", 1.5),
        "bb_width":            indicators_1h.get("bb_width", 2.0),
        "bb_pct_b":            indicators_1h.get("bb_pct_b", 0.5),
        "realized_vol":        indicators_1h.get("realized_vol", 0.0),
        "atr_percentile":      indicators_1h.get("atr_percentile", 50.0),
        "candle_body_pct":     indicators_1h.get("candle_body_pct", 0.5),

        # ── Volume features ───────────────────────────────────────
        "volume_ratio":        indicators_1h.get("volume_ratio", 1.0),
        "buy_vol_ratio":       indicators_1h.get("buy_vol_ratio", 0.5),
        "obv_signal":          1.0 if indicators_1h.get("obv", 0) > 0 else -1.0,

        # ── Time features — proven high impact in your data ───────
        "hour_utc":            float(hour),
        "day_of_week":         float(weekday),
        "is_ny_session":       1.0 if 14 <= hour <= 19 else 0.0,
        "is_london_session":   1.0 if 7 <= hour <= 13 else 0.0,
        "is_weekend":          1.0 if weekday >= 5 else 0.0,
        "is_saturday":         1.0 if weekday == 5 else 0.0,

        # ── Market regime ─────────────────────────────────────────
        "market_regime":       float(regime_num),

        # ── Bot performance feedback ───────────────────────────────
        "win_rate_recent":     recent_win_rate,
        "current_drawdown":    current_drawdown,

        # ── Candle direction ──────────────────────────────────────
        "is_bullish_candle":   1.0 if indicators_1h.get("is_bullish_candle") else -1.0,
    }


# Column order must be fixed and consistent
# for the model to use the same feature order every time
FEATURE_COLUMNS = [
    "ema_gap_pct_1h", "price_vs_ema50", "ema50_slope",
    "trend_4h", "ema_gap_pct_4h", "supertrend_dir",
    "adx", "adx_trending",
    "rsi_1h", "rsi_4h", "rsi_divergence", "rsi_vs_midpoint",
    "stoch_rsi_k", "stoch_rsi_d",
    "macd", "macd_histogram", "macd_signal", "cci",
    "atr_pct", "volatility_pct", "bb_width", "bb_pct_b",
    "realized_vol", "atr_percentile", "candle_body_pct",
    "volume_ratio", "buy_vol_ratio", "obv_signal",
    "hour_utc", "day_of_week",
    "is_ny_session", "is_london_session",
    "is_weekend", "is_saturday",
    "market_regime",
    "win_rate_recent", "current_drawdown",
    "is_bullish_candle",
]


def vector_to_list(features: dict) -> list[float]:
    return [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]