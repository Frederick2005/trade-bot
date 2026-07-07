"""
app/strategy/signal_logic.py

Single source of truth for the EMA/RSI pullback entry rule.

Both app/strategy/ema_rsi.py (live) and scripts/backtest.py (offline)
import evaluate_entry() from here. This is the fix for the most dangerous
class of trading-bot bug: backtest and live silently drifting apart because
the same logic got duplicated (and then edited) in two places. Now there is
exactly one place this logic lives.

Confluence layer (added after reading "LTA Concepts"):
Optional volume-profile / supply-demand confluence bumps CONFIDENCE only —
it never gates whether a trade fires. This matches the book's own framing
(Ch. 26, "Data In Zones"): "You don't always need all four to line up...
if even one supports your zone, it could be enough." The core EMA/RSI/
pullback gate below is unchanged and is still what decides whether a trade
happens at all — confluence only says how strongly to weight it, which
your AI confidence layer (Step 9) or a fixed threshold can use downstream.
"""

import math
from typing import Optional

# ── Tunable thresholds ──────────────────────────────────────────────────────
ATR_MULTIPLIER    = 1.5   # stop-loss distance = ATR * this
TP_RISK_REWARD    = 2.0   # take-profit distance = stop distance * this.
                           # LTA Concepts Ch. 32 backtested 1:1 vs 2:1 vs
                           # 4-5:1 and landed on 2:1 as the sweet spot: only
                           # needs a 35% win rate to break even, and doesn't
                           # require the razor-tight stops that make 4-5:1
                           # setups get wicked out constantly. Was 1.5 before
                           # this change — re-run your backtest after this
                           # edit, it WILL move your numbers.
RSI_LOW           = 42.0
RSI_HIGH          = 62.0
PULLBACK_BAND_PCT = 1.5   # price must be within this % of EMA50 (signal tf)
MIN_BODY_PCT      = 0.55  # signal-timeframe candle body must be at least this
                           # % of its range.
                           # CORRECTION: this was originally justified by a
                           # backtest finding attributed to "candle_body_pct",
                           # but that measurement had an argument-order bug in
                           # app/ai/features.py's caller — it was actually
                           # reading the TREND-timeframe candle's body, not
                           # this signal-timeframe one. The bug is fixed (see
                           # TREND_CANDLE_MIN_BODY_PCT below, which is the
                           # variable that finding was really about). Once this
                           # gate is combined with the trend-candle gate below,
                           # every trade in the tested data already satisfies
                           # MIN_BODY_PCT>=0.55, so its OWN marginal
                           # contribution on top of the trend-candle filter is
                           # currently unverified — kept at 0.55 as a
                           # reasonable floor, not because it's been proven to
                           # help on its own. Re-test if you want to isolate it.

TREND_CANDLE_MIN_BODY_PCT = 0.8
# Trend-timeframe (e.g. 1H) candle body must be at least this % of its range
# at the moment of entry — i.e. only take a signal when the higher timeframe
# is already showing strong directional conviction, not chopping.
#
# This is the single strongest validated finding across everything tested so
# far (9,200 BTCUSDT trades, clean/verified data after fixing the argument-
# order bug above): filtering to >=0.8 took win rate from 33.65% to 51.29%
# and expectancy from -0.11R to +0.34R. Verified three separate ways, not
# just once:
#   - chronological halves:  +0.38R / +0.30R (both positive)
#   - chronological thirds:  +0.47R / +0.28R / +0.28R (all positive)
#   - 95% CI on win rate stayed in the 46-59% range across every slice,
#     never dropping back toward the ~34% baseline
# Caveats, stated plainly:
#   - This drops trade FREQUENCY substantially (~8% of the original signal
#     count passed this filter) — expect far fewer trades live.
#   - Tested on BTCUSDT only in this run; ETHUSDT data was absent from the
#     export this was validated against — re-validate once you have a clean
#     multi-symbol run.
#   - Still a single continuous historical window, not truly independent
#     data (e.g. not a different market regime like a sustained bear market).
#     Keep re-validating as more/different data comes in.


def evaluate_entry(
    ind_signal: dict,
    ind_trend: dict,
    confluence: Optional[dict] = None,
) -> Optional[dict]:
    """
    ind_signal: indicators from the faster/entry timeframe (e.g. 15m)
    ind_trend:  indicators from the slower/trend-confirmation timeframe (e.g. 1h)

    Both dicts are expected in the schema produced by
    app/market/candles.py:get_indicators() / scripts/backtest.py:row_to_indicators().

    confluence: optional dict with any of:
        {"volume_profile_score": float 0-1, "zone_score": float 0-1}
      from app.analysis.volume_profile.confluence_score() and
      app.analysis.supply_demand.zone_confluence(). Purely additive to
      confidence — omit it (or pass None) and behavior is identical to
      before this was added.

    Returns {side, entry, sl, tp, confidence, reason} or None.
    """
    price          = ind_signal.get("price")
    ema50_s        = ind_signal.get("ema50")
    ema200_s       = ind_signal.get("ema200")
    rsi_s          = ind_signal.get("rsi")
    atr_s          = ind_signal.get("atr")
    body_pct       = ind_signal.get("candle_body_pct", 0.0)
    bullish        = ind_signal.get("is_bullish_candle", False)
    price_vs_ema50 = ind_signal.get("price_vs_ema50", 0.0)
    volume_ratio   = ind_signal.get("volume_ratio", 1.0)
    trend_strength_s = ind_signal.get("trend_strength", 0.0)

    ema50_t  = ind_trend.get("ema50")
    ema200_t = ind_trend.get("ema200")
    trend_body_pct = ind_trend.get("candle_body_pct", 0.0)

    required = [price, ema50_s, ema200_s, ema50_t, ema200_t, atr_s]
    if any(v is None for v in required):
        return None
    if rsi_s is None:
        return None
    try:
        if math.isnan(float(rsi_s)):
            return None
    except (TypeError, ValueError):
        return None

    if not (RSI_LOW <= rsi_s <= RSI_HIGH):
        return None
    if not (-PULLBACK_BAND_PCT <= price_vs_ema50 <= PULLBACK_BAND_PCT):
        return None
    if body_pct < MIN_BODY_PCT:
        return None
    if trend_body_pct < TREND_CANDLE_MIN_BODY_PCT:
        return None

    trend_up    = ema50_t > ema200_t
    trend_down  = ema50_t < ema200_t
    signal_up   = ema50_s > ema200_s
    signal_down = ema50_s < ema200_s

    if trend_up and signal_up and bullish:
        side = "LONG"
    elif trend_down and signal_down and not bullish:
        side = "SHORT"
    else:
        return None

    if side == "LONG":
        sl = price - (atr_s * ATR_MULTIPLIER)
        if (price - sl) <= 0:
            return None
        tp = price + ((price - sl) * TP_RISK_REWARD)
    else:
        sl = price + (atr_s * ATR_MULTIPLIER)
        if (sl - price) <= 0:
            return None
        tp = price - ((sl - price) * TP_RISK_REWARD)

    # Confidence: base rate + bonuses for stronger confirming conditions.
    # This is the RULE-BASED confidence only — engine.py overrides it with
    # the AI model's confidence when a trained model is loaded.
    confidence = 0.55
    if volume_ratio and volume_ratio > 1.0:
        confidence += 0.15
    if abs(trend_strength_s) >= 2.0:
        confidence += 0.15
    if body_pct >= 0.5:
        confidence += 0.15
    confidence = min(confidence, 1.0)

    confluence_note = ""
    if confluence:
        vp_score = confluence.get("volume_profile_score", 0.0) or 0.0
        zone_score = confluence.get("zone_score", 0.0) or 0.0
        # Additive bonus, capped so confluence alone can't exceed +0.3 total —
        # it's a tiebreaker/conviction booster, not a replacement for the
        # core technical gate above.
        bonus = min((vp_score * 0.15) + (zone_score * 0.15), 0.3)
        if bonus > 0:
            confidence = min(confidence + bonus, 1.0)
            confluence_note = f" [confluence: vp={vp_score:.2f} zone={zone_score:.2f}]"

    reason = (
        f"{side} pullback: RSI={rsi_s:.1f} price_vs_ema50={price_vs_ema50:+.2f}% "
        f"body={body_pct:.0%} vol_ratio={volume_ratio:.2f}{confluence_note}"
    )

    return {
        "side": side,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "confidence": confidence,
        "reason": reason,
    }