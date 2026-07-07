"""
app/analysis/supply_demand.py

Supply/Demand zone detection based on LTA Concepts Ch. 21-28: the four
base-and-expansion patterns (Rally-Base-Rally, Drop-Base-Drop,
Drop-Base-Rally, Rally-Base-Drop).

The book's own words on what actually makes a zone (Ch. 24): "the shape of
the candle doesn't matter as much as what it represents... the clearest
areas of supply and demand almost always formed right before a breakout".
That's the part that's actually quantifiable:

  1. A BASE: one or more candles with a small range relative to recent
     volatility (consolidation/indecision — the "coiling" the book
     describes).
  2. An EXPANSION: a candle that breaks out of the base with a range
     meaningfully larger than the base (the book's "three-candle bar
     play": the expansion candle's range exceeds the base).

The zone itself is the base's high-low range. Direction is determined by
which way the expansion candle broke (up = demand zone below, down =
supply zone above). This collapses the book's four named patterns
(RBR/DBD/DBR/RBD) into one detector, since algorithmically they're the same
base+expansion structure — only the preceding trend context differentiates
continuation (RBR/DBD) from reversal (DBR/RBD), which callers can layer on
top using their own trend read (e.g. from signal_logic's ema50/ema200).

Caveat, stated plainly: the book explicitly runs this on HIGH timeframes
(monthly/weekly/daily/12H/8H) for a reason — that's where it claims
institutional footprints actually show up. Running this detector on 15m/1h
candles will find many more "zones" that are mostly noise. The intended
usage in this project is: run this on 4H/1D data to build the zone map,
and use your existing 15m/1h data only to time entries once price is near
a zone found on the higher timeframe.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Zone:
    kind: str          # "demand" or "supply"
    high: float
    low: float
    formed_at: pd.Timestamp
    base_candles: int
    expansion_ratio: float   # how many multiples of the base range the expansion candle covered
    tested_count: int = 0    # how many times price has revisited this zone since formation


def detect_zones(
    df: pd.DataFrame,
    atr: Optional[pd.Series] = None,
    max_base_range_atr: float = 0.5,
    max_base_candles: int = 4,
    min_expansion_ratio: float = 1.8,
    lookback: Optional[int] = None,
) -> List[Zone]:
    """
    df: higher-timeframe OHLCV (4H/1D recommended — see module docstring).
    atr: precomputed ATR series aligned to df's index. If None, computed
         internally with a 14-period true range average.
    max_base_range_atr: a candle counts as part of a "base" if its
         high-low range is <= this many ATRs (small/coiled candle).
    max_base_candles: look back at most this many consecutive base candles
         before the expansion candle.
    min_expansion_ratio: the expansion candle's range must be at least this
         many times the average base-candle range to count as a genuine
         breakout (the book's "third candle exceeds the first").
    lookback: only scan the last N candles (None = scan all of df).

    Returns zones sorted newest-first.
    """
    if df.empty or len(df) < max_base_candles + 2:
        return []

    work = df.tail(lookback).reset_index(drop=True) if lookback else df.reset_index(drop=True)

    if atr is None:
        prev_close = work["close"].shift(1)
        tr = pd.concat([
            work["high"] - work["low"],
            (work["high"] - prev_close).abs(),
            (work["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    else:
        atr = atr.reset_index(drop=True)

    ranges = (work["high"] - work["low"]).to_numpy()
    closes = work["close"].to_numpy()
    opens = work["open"].to_numpy()
    highs = work["high"].to_numpy()
    lows = work["low"].to_numpy()
    atr_vals = atr.to_numpy()
    times = work["open_time"].to_numpy()

    zones: List[Zone] = []

    i = max_base_candles
    n = len(work)
    while i < n:
        atr_i = atr_vals[i]
        if not atr_i or np.isnan(atr_i) or atr_i <= 0:
            i += 1
            continue

        expansion_range = ranges[i]
        if expansion_range < min_expansion_ratio * 0.0001:  # guard against degenerate zero-range data
            i += 1
            continue

        # walk backward collecting consecutive "small range" base candles
        base_start = i - 1
        base_count = 0
        while (
            base_start >= 0
            and base_count < max_base_candles
            and ranges[base_start] <= max_base_range_atr * atr_i
        ):
            base_count += 1
            base_start -= 1
        base_start += 1  # step back to first valid base candle

        if base_count == 0:
            i += 1
            continue

        base_range_avg = ranges[base_start:i].mean()
        if base_range_avg <= 0:
            i += 1
            continue

        expansion_ratio = expansion_range / base_range_avg
        if expansion_ratio < min_expansion_ratio:
            i += 1
            continue

        bullish_expansion = closes[i] > opens[i]
        base_high = highs[base_start:i].max()
        base_low = lows[base_start:i].min()

        zones.append(Zone(
            kind="demand" if bullish_expansion else "supply",
            high=float(base_high),
            low=float(base_low),
            formed_at=pd.Timestamp(times[i]),
            base_candles=base_count,
            expansion_ratio=float(expansion_ratio),
        ))
        i += 1

    zones.sort(key=lambda z: z.formed_at, reverse=True)
    return zones


def zone_confluence(price: float, direction: str, zones: List[Zone], tolerance_pct: float = 0.5) -> float:
    """
    0.0-1.0 confidence bonus for `price` sitting inside/near a fresh,
    direction-aligned zone. Same "stacking" philosophy as
    volume_profile.confluence_score() — a bonus, not a hard gate.

    direction: "LONG" wants a nearby DEMAND zone below/at price.
               "SHORT" wants a nearby SUPPLY zone above/at price.
    """
    if not zones:
        return 0.0

    wanted_kind = "demand" if direction == "LONG" else "supply"
    best = 0.0

    for zone in zones:
        if zone.kind != wanted_kind:
            continue
        pad = (zone.high - zone.low) * (tolerance_pct / 100) if zone.high > zone.low else 0
        inside = (zone.low - pad) <= price <= (zone.high + pad)
        if not inside:
            continue

        # fresher and untested zones score higher, per the book's
        # "untested zones tend to be more reliable" (Ch. 24)
        freshness_bonus = 0.3 if zone.tested_count == 0 else 0.1
        strength_bonus = min(zone.expansion_ratio / 5.0, 0.4)
        score = 0.3 + freshness_bonus + strength_bonus
        best = max(best, min(score, 1.0))

    return best
