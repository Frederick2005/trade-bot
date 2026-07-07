"""
app/analysis/market_structure.py

Implements Stage 3 of the AtlasQuant v2 spec: Break of Structure (BOS),
Change of Character (CHOCH), and swing-based Higher-High/Higher-Low/
Lower-High/Lower-Low classification.

This is the "market structure" concept done honestly: it's a fractal swing
detector plus a rule for what counts as a structural break. It does NOT
attempt liquidity sweeps, stop hunts, fair value gaps, or order blocks
(Stage 4 in the spec) — those concepts are defined in terms of order-book/
liquidity behavior that isn't reliably inferable from OHLCV candles alone.
Approximating them from candles tends to produce a lot of false-positive
"detections" that look sophisticated but add noise, not signal. If you want
those later, the honest way in is via the exchange's real order book /
trade tape, not candle pattern-matching — that's a deliberate scope
decision, not an oversight.

Definitions used here:
- Swing high: a candle whose high is higher than `lookback` candles on
  each side (a local fractal peak).
- Swing low: symmetric, local fractal trough.
- Trend structure: sequence of swing highs/lows is classified as HH/HL
  (uptrend structure) or LH/LL (downtrend structure).
- BOS (Break of Structure): price closes beyond the most recent swing
  point IN THE DIRECTION OF the prevailing structure — confirms trend
  continuation.
- CHOCH (Change of Character): price closes beyond the most recent swing
  point AGAINST the prevailing structure — the first warning sign of a
  potential trend change. This is intentionally the FIRST break against
  structure, not full confirmation of a reversal.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    kind: str            # "high" or "low"
    price: float
    index: int
    time: pd.Timestamp


@dataclass
class StructureState:
    swings: List[SwingPoint] = field(default_factory=list)
    structure: str = "undefined"   # "uptrend", "downtrend", "undefined"
    last_bos: Optional[str] = None       # "up" / "down" / None
    last_choch: Optional[str] = None     # "up" / "down" / None
    hh: bool = False
    hl: bool = False
    lh: bool = False
    ll: bool = False


def find_swings(df: pd.DataFrame, lookback: int = 3) -> List[SwingPoint]:
    """
    Fractal swing detection: a candle at index i is a swing high if its
    high is the maximum within [i-lookback, i+lookback], and symmetric
    for swing lows. Requires `lookback` candles of confirmation AFTER the
    swing, so the most recent `lookback` candles can never produce a
    confirmed swing yet — that's correct and unavoidable (you can't know a
    local peak is a peak until price has moved away from it).
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["open_time"].to_numpy()
    n = len(df)
    swings: List[SwingPoint] = []

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback: i + lookback + 1]
        window_low = lows[i - lookback: i + lookback + 1]

        if highs[i] == window_high.max() and np.argmax(window_high) == lookback:
            swings.append(SwingPoint("high", float(highs[i]), i, pd.Timestamp(times[i])))
        if lows[i] == window_low.min() and np.argmin(window_low) == lookback:
            swings.append(SwingPoint("low", float(lows[i]), i, pd.Timestamp(times[i])))

    swings.sort(key=lambda s: s.index)
    return swings


def analyze_structure(df: pd.DataFrame, lookback: int = 3, max_swings: int = 8) -> StructureState:
    """
    Runs swing detection and classifies the current market structure as of
    the LAST closed candle in df. Also checks whether the last candle's
    close broke the most recent opposing-direction swing point (BOS/CHOCH).
    """
    swings = find_swings(df, lookback=lookback)
    state = StructureState(swings=swings[-max_swings:])

    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]

    if len(highs) >= 2:
        state.hh = highs[-1].price > highs[-2].price
        state.lh = highs[-1].price < highs[-2].price
    if len(lows) >= 2:
        state.hl = lows[-1].price > lows[-2].price
        state.ll = lows[-1].price < lows[-2].price

    if state.hh and state.hl:
        state.structure = "uptrend"
    elif state.lh and state.ll:
        state.structure = "downtrend"
    else:
        state.structure = "undefined"

    # BOS/CHOCH: does the LAST CLOSE break the most recent swing high/low?
    last_close = df["close"].iloc[-1]
    most_recent_high = highs[-1] if highs else None
    most_recent_low = lows[-1] if lows else None

    if most_recent_high and last_close > most_recent_high.price:
        if state.structure == "uptrend":
            state.last_bos = "up"
        elif state.structure == "downtrend":
            state.last_choch = "up"   # break against a downtrend = potential reversal warning

    if most_recent_low and last_close < most_recent_low.price:
        if state.structure == "downtrend":
            state.last_bos = "down"
        elif state.structure == "uptrend":
            state.last_choch = "down"

    return state


def nearest_swing_stop(
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
    lookback: int = 3,
    max_lookback_candles: int = 100,
) -> Optional[float]:
    """
    For dynamic stop-loss placement (spec: "Stop Loss: Dynamic. Use
    whichever is larger: ATR / Structure / Swing"). Returns the nearest
    swing low (for LONG) or swing high (for SHORT) below/above entry_price,
    which is the structural level a stop should sit beyond. Returns None
    if no suitable swing is found within max_lookback_candles.
    """
    recent = df.tail(max_lookback_candles)
    swings = find_swings(recent, lookback=lookback)

    if direction == "LONG":
        candidates = [s.price for s in swings if s.kind == "low" and s.price < entry_price]
        return max(candidates) if candidates else None   # nearest (highest) swing low below entry
    else:
        candidates = [s.price for s in swings if s.kind == "high" and s.price > entry_price]
        return min(candidates) if candidates else None   # nearest (lowest) swing high above entry