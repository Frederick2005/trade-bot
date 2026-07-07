"""
app/strategy/decision_engine.py

The multi-stage decision pipeline from the AtlasQuant v2 spec, built from
pieces that are genuinely implementable and testable from OHLCV data:

  Stage 1  Market regime               -> regime.py-style classification (inline below)
  Stage 2  HTF trend (primary+secondary) -> EMA50/200 across 3 timeframes
  Stage 3  Market structure             -> app.analysis.market_structure
  Stage 4  Liquidity                    -> DEFERRED, see market_structure.py docstring
  Stage 5  Trend quality                -> folded into quality_score.py
  Stage 6  Pullback quality             -> folded into quality_score.py
  Stage 7  Volume                       -> folded into quality_score.py
  Stage 8  Momentum                     -> folded into quality_score.py
  Stage 9  Volatility                   -> folded into quality_score.py
  Stage 10 Entry confirmation           -> signal_logic.evaluate_entry() + quality_score.py

This does NOT replace signal_logic.evaluate_entry() — it wraps it. That
function is still what proposes a candidate LONG/SHORT trade at all (it's
the "is there even a technical trigger here" check); this module adds the
regime gate, 3-timeframe trend alignment, market structure, and the 0-100
quality score gate on top, and computes the final stop/target using
dynamic stop-loss (max of ATR-based and structure-based, per spec) and a
configurable minimum reward:risk.

IMPORTANT — read before changing MIN_REWARD_RATIO:
The spec asks for a 3:1 minimum target. At a 3:1 R:R, the breakeven win
rate is 25%. That's an easier bar mathematically, but it does NOT mean
raising the target automatically makes the strategy better — a further
target is statistically harder for price to reach before reversing, all
else equal. The correct way to find out if 3:1 helps THIS strategy is to
backtest it and compare expectancy against 2:1 and other candidates, not
to assume the wider target is free money. Whatever number you configure
here is a hypothesis, not a guarantee, until validated on real data.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.strategy.signal_logic import evaluate_entry, ATR_MULTIPLIER
from app.strategy import quality_score
from app.analysis import market_structure


@dataclass
class DecisionResult:
    accepted: bool
    side: Optional[str] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quality_score: Optional[float] = None
    quality_subscores: dict = field(default_factory=dict)
    regime: Optional[str] = None
    structure: Optional[str] = None
    rejection_reason: Optional[str] = None
    reason: Optional[str] = None
    raw_confidence: Optional[float] = None   # from signal_logic, pre-quality-gate


def classify_regime(ind_primary: dict, ind_secondary: dict) -> str:
    """
    Stage 1 — lightweight regime classification from indicators already
    computed (no extra data source). Categories collapsed from the spec's
    7 down to what's distinguishable from EMA/ATR/trend_strength alone:
    strong_trend_up / trend_up / sideways / trend_down / strong_trend_down /
    high_volatility / low_volatility.
    """
    trend_strength = ind_primary.get("trend_strength", 0.0) or 0.0
    volatility_pct = ind_primary.get("volatility_pct", 1.0) or 1.0

    if volatility_pct > 5.0:
        return "high_volatility"
    if volatility_pct < 0.1:
        return "low_volatility"

    if trend_strength > 3.0:
        return "strong_trend_up"
    if trend_strength > 0.8:
        return "trend_up"
    if trend_strength < -3.0:
        return "strong_trend_down"
    if trend_strength < -0.8:
        return "trend_down"
    return "sideways"


def evaluate_setup(
    df_execution: pd.DataFrame,
    df_secondary: pd.DataFrame,
    df_primary: pd.DataFrame,
    ind_execution: dict,
    ind_secondary: dict,
    ind_primary: dict,
    quality_threshold: float = 80.0,
    min_reward_ratio: float = 3.0,
    confluence: Optional[dict] = None,
) -> DecisionResult:
    """
    Runs the full pipeline for the CURRENT (last-closed-candle) state of
    all three timeframes and returns an accept/reject decision with full
    reasoning attached for logging.

    df_execution/df_secondary/df_primary: enough trailing OHLCV history on
    each timeframe to run market-structure swing detection (100+ candles
    recommended for df_secondary, since structure runs on the secondary
    timeframe per the spec).
    """
    # ── Stage 1: Regime ──
    regime = classify_regime(ind_primary, ind_secondary)
    if regime in ("sideways", "high_volatility", "low_volatility"):
        return DecisionResult(
            accepted=False, regime=regime,
            rejection_reason=f"regime '{regime}' unsuitable for trend trading",
        )

    # ── Stage 2 + 10: base technical trigger (trend + pullback + entry candle) ──
    candidate = evaluate_entry(ind_execution, ind_secondary, confluence=confluence)
    if candidate is None:
        return DecisionResult(accepted=False, regime=regime, rejection_reason="no technical trigger")

    side = candidate["side"]

    # Regime must agree with the proposed direction.
    regime_wants_long = regime in ("strong_trend_up", "trend_up")
    if (side == "LONG") != regime_wants_long:
        return DecisionResult(
            accepted=False, side=side, regime=regime,
            rejection_reason=f"regime '{regime}' conflicts with {side} signal",
        )

    # ── Stage 3: Market structure (on the secondary timeframe, per spec) ──
    structure_state = market_structure.analyze_structure(df_secondary)

    # ── Stages 5-9 + confirmation: Quality score ──
    quality = quality_score.score_setup(
        direction=side,
        ind_execution=ind_execution,
        ind_secondary=ind_secondary,
        ind_primary=ind_primary,
        structure_secondary=structure_state,
        confluence=confluence,
        threshold=quality_threshold,
    )

    if not quality.passed:
        return DecisionResult(
            accepted=False, side=side, regime=regime,
            structure=structure_state.structure,
            quality_score=quality.total, quality_subscores=quality.subscores,
            raw_confidence=candidate["confidence"],
            rejection_reason=f"quality score {quality.total} below threshold {quality_threshold}",
        )

    # ── Dynamic stop-loss: max(ATR-based, structure-based) per spec ──
    entry_price = candidate["entry"]
    atr_stop = candidate["sl"]
    structure_stop = market_structure.nearest_swing_stop(df_secondary, side, entry_price)

    if structure_stop is not None:
        if side == "LONG":
            # "whichever is larger" = whichever gives the WIDER stop distance
            final_stop = min(atr_stop, structure_stop)  # lower stop = wider distance for LONG
        else:
            final_stop = max(atr_stop, structure_stop)  # higher stop = wider distance for SHORT
    else:
        final_stop = atr_stop

    stop_distance = abs(entry_price - final_stop)
    if stop_distance <= 0:
        return DecisionResult(accepted=False, side=side, regime=regime, rejection_reason="invalid stop distance")

    if side == "LONG":
        take_profit = entry_price + stop_distance * min_reward_ratio
    else:
        take_profit = entry_price - stop_distance * min_reward_ratio

    return DecisionResult(
        accepted=True,
        side=side,
        entry=entry_price,
        stop_loss=final_stop,
        take_profit=take_profit,
        quality_score=quality.total,
        quality_subscores=quality.subscores,
        regime=regime,
        structure=structure_state.structure,
        raw_confidence=candidate["confidence"],
        reason=candidate["reason"],
    )