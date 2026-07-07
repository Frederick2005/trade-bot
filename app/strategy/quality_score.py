"""
app/strategy/quality_score.py

Stage: Quality Score. A 0-100 composite score built from independently
scored categories, matching the AtlasQuant v2 spec's requirement that
"every category is scored independently" and "every sub-score is stored
separately" for explainability.

This does NOT include liquidity/order-block/FVG scoring (see
app/analysis/market_structure.py's docstring for why those are deferred).
The categories below are the ones with an honest, quantifiable definition
from data this project actually has.

Weights sum to 100. They're a starting point, not a claim of optimality —
these should be tuned via actual backtesting (e.g. the RR-sweep-style
approach), not treated as fixed truth.
"""

from dataclasses import dataclass, field
from typing import Optional


WEIGHTS = {
    "trend_alignment":     15,   # HTF + secondary + execution EMA trend agree
    "market_structure":    15,   # HH/HL or LH/LL structure, BOS supports direction
    "trend_quality":       10,   # ADX strength, EMA spacing/slope
    "pullback_quality":    15,   # distance from EMA, RSI zone, momentum reduction
    "volume":               10,   # volume ratio above average
    "momentum":             10,   # RSI/trend_strength direction agreement
    "volatility":           10,   # ATR in a reasonable (not extreme) range
    "entry_confirmation":   10,   # confirmation candle body/direction
    "confluence":            5,   # volume profile / supply-demand bonus (optional)
}
assert sum(WEIGHTS.values()) == 100


@dataclass
class QualityResult:
    total: float
    subscores: dict
    passed: bool
    threshold: float


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_setup(
    *,
    direction: str,
    ind_execution: dict,
    ind_secondary: dict,
    ind_primary: dict,
    structure_secondary,          # StructureState from market_structure.analyze_structure()
    confluence: Optional[dict] = None,
    threshold: float = 80.0,
) -> QualityResult:
    """
    All ind_* dicts follow the schema in app/market/candles.py /
    scripts/backtest.py (price, ema50, ema200, rsi, atr, volume_ratio,
    price_vs_ema50, trend_strength, candle_body_pct, is_bullish_candle,
    ema50_slope).

    structure_secondary: output of market_structure.analyze_structure() run
    on the secondary-trend timeframe (e.g. 1H) — the timeframe the spec
    calls out for market-structure analysis.
    """
    is_long = direction == "LONG"
    sub = {}

    # ── Trend alignment (15) — do execution, secondary, and primary all agree? ──
    exec_up = ind_execution.get("ema50", 0) > ind_execution.get("ema200", 0)
    sec_up = ind_secondary.get("ema50", 0) > ind_secondary.get("ema200", 0)
    pri_up = ind_primary.get("ema50", 0) > ind_primary.get("ema200", 0)
    agree_count = sum([
        exec_up == is_long,
        sec_up == is_long,
        pri_up == is_long,
    ])
    sub["trend_alignment"] = WEIGHTS["trend_alignment"] * (agree_count / 3)

    # ── Market structure (15) ──
    struct_score = 0.0
    if structure_secondary is not None:
        wants_uptrend = is_long
        if structure_secondary.structure == ("uptrend" if wants_uptrend else "downtrend"):
            struct_score += 0.6
        if structure_secondary.last_bos == ("up" if is_long else "down"):
            struct_score += 0.4
        elif structure_secondary.last_choch == ("down" if is_long else "up"):
            struct_score -= 0.4   # a CHOCH against your direction is a real warning sign
    sub["market_structure"] = WEIGHTS["market_structure"] * _clamp(struct_score)

    # ── Trend quality (10) — ADX-like proxy via trend_strength + EMA slope ──
    trend_strength = abs(ind_secondary.get("trend_strength", 0.0))
    slope = ind_secondary.get("ema50_slope", 0.0)
    slope_agrees = (slope > 0) == is_long
    ts_component = _clamp(trend_strength / 3.0)          # saturates around 3% EMA gap
    sub["trend_quality"] = WEIGHTS["trend_quality"] * (0.5 * ts_component + 0.5 * slope_agrees)

    # ── Pullback quality (15) ──
    price_vs_ema50 = abs(ind_execution.get("price_vs_ema50", 99))
    pullback_component = _clamp(1 - (price_vs_ema50 / 1.5))  # closer to EMA50 = better, up to the 1.5% band
    rsi = ind_execution.get("rsi", 50)
    rsi_mid_distance = abs(rsi - 50)
    rsi_component = _clamp(1 - (rsi_mid_distance / 20))       # closer to neutral RSI = calmer pullback
    sub["pullback_quality"] = WEIGHTS["pullback_quality"] * (0.6 * pullback_component + 0.4 * rsi_component)

    # ── Volume (10) ──
    vol_ratio = ind_execution.get("volume_ratio", 1.0) or 1.0
    sub["volume"] = WEIGHTS["volume"] * _clamp((vol_ratio - 0.8) / 0.8)

    # ── Momentum (10) — does RSI position agree with intended direction? ──
    if is_long:
        momentum_component = _clamp((rsi - 40) / 20)
    else:
        momentum_component = _clamp((60 - rsi) / 20)
    sub["momentum"] = WEIGHTS["momentum"] * momentum_component

    # ── Volatility (10) — penalize extremes, reward a normal ATR% range ──
    volatility_pct = ind_execution.get("volatility_pct", 1.0) or 1.0
    if volatility_pct < 0.15:
        vol_component = volatility_pct / 0.15            # too quiet — unreliable moves
    elif volatility_pct > 4.0:
        vol_component = _clamp(1 - ((volatility_pct - 4.0) / 4.0))   # too wild — unstable stops
    else:
        vol_component = 1.0
    sub["volatility"] = WEIGHTS["volatility"] * _clamp(vol_component)

    # ── Entry confirmation (10) ──
    body_pct = ind_execution.get("candle_body_pct", 0.0) or 0.0
    bullish = ind_execution.get("is_bullish_candle", False)
    direction_confirmed = (bullish == is_long)
    sub["entry_confirmation"] = WEIGHTS["entry_confirmation"] * (
        _clamp(body_pct / 0.7) * (1.0 if direction_confirmed else 0.0)
    )

    # ── Confluence (5) — optional volume-profile/supply-demand bonus ──
    if confluence:
        vp = confluence.get("volume_profile_score", 0.0) or 0.0
        zone = confluence.get("zone_score", 0.0) or 0.0
        sub["confluence"] = WEIGHTS["confluence"] * _clamp(max(vp, zone))
    else:
        sub["confluence"] = 0.0

    total = sum(sub.values())
    return QualityResult(total=round(total, 2), subscores={k: round(v, 2) for k, v in sub.items()}, passed=total >= threshold, threshold=threshold)