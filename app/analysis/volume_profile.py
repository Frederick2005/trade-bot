"""
app/analysis/volume_profile.py

Fixed-Range Volume Profile, computed from candle OHLCV data you already have
— no new data source required. Implements the definitions from the LTA
Concepts book (Ch. 1-6):

- POC (Point of Control): the price level with the most traded volume.
- VAH / VAL (Value Area High/Low): the boundaries of the range containing
  ~70% of total volume (standard Market Profile convention; the book uses
  70-80%, we use 70% as the more conservative/common default).
- HVN (High Volume Node): a local volume peak — likely support/resistance.
- LVN (Low Volume Node): a local volume trough — price tends to move
  through these quickly (low "acceptance").

Approximation note: real volume-at-price requires trade-by-trade or
tick-level data. We approximate the volume of each candle as uniformly
distributed across its high-low range, which is the same approximation
most retail charting platforms (TradingView included) use for the
"Fixed Range Volume Profile" tool referenced throughout the book. It is
good enough to find POC/VAH/VAL levels that line up with real
support/resistance, but treat it as an approximation, not exact
order-book truth.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolumeProfileResult:
    poc: float
    vah: float
    val: float
    hvn_prices: list      # local high-volume-node price levels, strongest first
    lvn_prices: list      # local low-volume-node price levels
    price_bins: np.ndarray
    volumes: np.ndarray


def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
) -> Optional[VolumeProfileResult]:
    """
    df: OHLCV window to profile (e.g. previous day's 15m candles, or
        previous week's 1h candles — the caller decides the window,
        matching the book's "Daily/Weekly Volume Profile" concept).
    """
    if df.empty or len(df) < 2:
        return None

    lo = df["low"].min()
    hi = df["high"].max()
    if hi <= lo:
        return None

    bin_edges = np.linspace(lo, hi, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume_at_price = np.zeros(num_bins)

    # Distribute each candle's volume uniformly across the bins its
    # high-low range overlaps (the standard Fixed Range Volume Profile
    # approximation described above).
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    vols = df["volume"].to_numpy()

    for h, l, v in zip(highs, lows, vols):
        if h <= l or v <= 0:
            continue
        lo_bin = np.searchsorted(bin_edges, l, side="right") - 1
        hi_bin = np.searchsorted(bin_edges, h, side="left")
        lo_bin = max(0, min(lo_bin, num_bins - 1))
        hi_bin = max(0, min(hi_bin, num_bins))
        span = max(hi_bin - lo_bin, 1)
        volume_at_price[lo_bin:hi_bin if hi_bin > lo_bin else lo_bin + 1] += v / span

    total_volume = volume_at_price.sum()
    if total_volume <= 0:
        return None

    poc_idx = int(np.argmax(volume_at_price))
    poc = float(bin_centers[poc_idx])

    # Expand outward from POC until value_area_pct of volume is captured —
    # standard Market Profile value-area algorithm.
    included = {poc_idx}
    captured = volume_at_price[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while captured < total_volume * value_area_pct and (lo_idx > 0 or hi_idx < num_bins - 1):
        vol_below = volume_at_price[lo_idx - 1] if lo_idx > 0 else -1
        vol_above = volume_at_price[hi_idx + 1] if hi_idx < num_bins - 1 else -1
        if vol_above >= vol_below:
            hi_idx += 1
            captured += volume_at_price[hi_idx]
            included.add(hi_idx)
        else:
            lo_idx -= 1
            captured += volume_at_price[lo_idx]
            included.add(lo_idx)

    val = float(bin_centers[lo_idx])
    vah = float(bin_centers[hi_idx])

    # HVN/LVN: local maxima/minima in the volume histogram (simple
    # neighbor comparison — good enough for confluence checks, not meant
    # to replace visual chart reading).
    hvn_prices, lvn_prices = [], []
    for i in range(1, num_bins - 1):
        if volume_at_price[i] > volume_at_price[i - 1] and volume_at_price[i] > volume_at_price[i + 1]:
            hvn_prices.append((float(bin_centers[i]), float(volume_at_price[i])))
        elif volume_at_price[i] < volume_at_price[i - 1] and volume_at_price[i] < volume_at_price[i + 1]:
            lvn_prices.append((float(bin_centers[i]), float(volume_at_price[i])))

    hvn_prices.sort(key=lambda x: -x[1])
    lvn_prices.sort(key=lambda x: x[1])

    return VolumeProfileResult(
        poc=poc,
        vah=vah,
        val=val,
        hvn_prices=[p for p, _ in hvn_prices[:5]],
        lvn_prices=[p for p, _ in lvn_prices[:5]],
        price_bins=bin_centers,
        volumes=volume_at_price,
    )


def near_level(price: float, level: float, tolerance_pct: float = 0.3) -> bool:
    """Is `price` within tolerance_pct percent of `level`?"""
    if level == 0:
        return False
    return abs(price - level) / level * 100 <= tolerance_pct


def confluence_score(price: float, profile: Optional[VolumeProfileResult], tolerance_pct: float = 0.3) -> float:
    """
    0.0-1.0 score for how much volume-profile confluence supports a trade
    at `price` right now. Used as a CONFIDENCE BONUS, not a hard gate — in
    keeping with the book's "stacking" philosophy (Ch. 26, Data In Zones):
    you don't need every confluence to line up, each one just adds
    conviction.
    """
    if profile is None:
        return 0.0
    score = 0.0
    if near_level(price, profile.poc, tolerance_pct):
        score += 0.5  # POC is the single strongest level
    if near_level(price, profile.vah, tolerance_pct) or near_level(price, profile.val, tolerance_pct):
        score += 0.3  # value area boundaries are the next strongest
    for hvn in profile.hvn_prices[:3]:
        if near_level(price, hvn, tolerance_pct):
            score += 0.2
            break
    return min(score, 1.0)
