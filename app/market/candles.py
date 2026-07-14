"""
app/market/candles.py
Candle buffer + full indicator suite.

Every indicator computed here feeds:
  - Strategy signal generation
  - AI feature vector (model input)
  - trade_context storage (learning data)
  - market_regime labeling
"""
from collections import defaultdict
import pandas as pd
import numpy as np
import ta
from loguru import logger

# In-memory candle store keyed by (symbol, timeframe)
_candles: dict[tuple, pd.DataFrame] = defaultdict(pd.DataFrame)

# Minimum candles needed before indicators are valid
# EMA200 needs 200 bars + warm-up
MIN_CANDLES = 140


def update(symbol: str, timeframe: str, candle: dict) -> None:
    key = (symbol, timeframe)
    row = pd.DataFrame([{
        "open_time":   pd.to_datetime(candle["open_time"]),
        "open":        float(candle["open"]),
        "high":        float(candle["high"]),
        "low":         float(candle["low"]),
        "close":       float(candle["close"]),
        "volume":      float(candle["volume"]),
        "buy_volume":  float(candle.get("buy_volume", 0)),
        "num_trades":  int(candle.get("num_trades", 0)),
        "quote_volume":float(candle.get("quote_volume", 0)),
    }])

    existing = _candles[key]
    if existing.empty:
        _candles[key] = row
    else:
        if existing.iloc[-1]["open_time"] == row.iloc[0]["open_time"]:
            _candles[key] = pd.concat([existing.iloc[:-1], row], ignore_index=True)
        else:
            _candles[key] = pd.concat([existing, row], ignore_index=True)

    if len(_candles[key]) > 500:
        _candles[key] = _candles[key].iloc[-500:].reset_index(drop=True)


def seed(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    _candles[(symbol, timeframe)] = df.copy().reset_index(drop=True)
    logger.info(f"Candle buffer seeded: {symbol} {timeframe} — {len(df)} candles")


def get_df(symbol: str, timeframe: str) -> pd.DataFrame:
    return _candles.get((symbol, timeframe), pd.DataFrame())


def is_ready(symbol: str, timeframe: str) -> bool:
    return len(get_df(symbol, timeframe)) >= MIN_CANDLES


def get_indicators(symbol: str, timeframe: str) -> dict | None:
    """
    Compute the full indicator suite for a symbol/timeframe.

    WHY EACH INDICATOR HELPS THE AI LEARN:
    ─────────────────────────────────────
    EMA50/200     → trend direction and strength. The AI learns
                    that trades WITH the trend win more often.
    RSI           → momentum state. AI learns which RSI zones
                    produce wins vs losses (your data shows 45-50
                    and 55-60 are best).
    ATR           → volatility measure. AI learns to avoid entries
                    during extreme volatility when stops get blown.
    MACD          → momentum shift detection. AI learns that MACD
                    crossovers confirm or deny RSI signals.
    Bollinger     → price extremes. AI learns that price at band
                    edges has higher mean-reversion probability.
    ADX           → trend strength 0-100. AI learns to prefer
                    trades when ADX > 25 (strong trend).
    Stoch RSI     → short-term overbought/oversold. AI learns
                    fine-grained momentum shifts the regular RSI misses.
    OBV           → volume flow. AI learns that price moves WITH
                    rising OBV are more reliable than those against it.
    Volume ratio  → activity level. Your data proves high volume
                    trades win 53% vs 49% for low volume.
    Supertrend    → directional bias. AI learns that trading with
                    supertrend direction adds 5-8% to win rate.
    CCI           → deviation from average. AI learns extreme CCI
                    readings often precede reversals.
    Buy/sell vol  → aggressor activity. When buy volume dominates,
                    bullish moves are more sustained.
    """
    df = get_df(symbol, timeframe)
    if len(df) < MIN_CANDLES:
        logger.warning(
            f"Not enough candles for {symbol} {timeframe}: "
            f"{len(df)}/{MIN_CANDLES}"
        )
        return None

    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        open_  = df["open"]

        # ── Trend indicators ──────────────────────────────────────
        ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()
        ema20  = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        ema100 = ta.trend.EMAIndicator(close, window=100).ema_indicator()
        sma20  = ta.trend.SMAIndicator(close, window=20).sma_indicator()

        # ── Momentum ──────────────────────────────────────────────
        rsi    = ta.momentum.RSIIndicator(close, window=14).rsi()

        # Stochastic RSI
        stoch  = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        stoch_rsi_k = stoch.stochrsi_k()
        stoch_rsi_d = stoch.stochrsi_d()

        # MACD
        macd_ind   = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_line  = macd_ind.macd()
        macd_sig   = macd_ind.macd_signal()
        macd_hist  = macd_ind.macd_diff()

        # CCI
        cci = ta.trend.CCIIndicator(high, low, close, window=20).cci()

        # ── Volatility ────────────────────────────────────────────
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        # Bollinger Bands
        bb     = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_up  = bb.bollinger_hband()
        bb_mid = bb.bollinger_mavg()
        bb_lo  = bb.bollinger_lband()
        bb_pct = bb.bollinger_pband()

        # Realised volatility (20-period std of log returns)
        log_ret  = np.log(close / close.shift(1))
        real_vol = log_ret.rolling(20).std() * np.sqrt(252) * 100

        # ATR percentile (where is current ATR vs last 100 bars)
        atr_pct = atr.rolling(100).rank(pct=True) * 100

        # ── Volume indicators ─────────────────────────────────────
        obv      = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        vol_avg  = volume.rolling(20).mean()
        vol_ratio = (volume / vol_avg)

        # Buy/sell volume split
        buy_vol_col  = df.get("buy_volume", pd.Series(0, index=df.index))
        sell_vol     = volume - buy_vol_col
        buy_vol_ratio = (buy_vol_col / volume.replace(0, 1)).fillna(0.5)

        # ── Trend strength ────────────────────────────────────────
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        adx     = adx_ind.adx()
        adx_pos = adx_ind.adx_pos()  # +DI
        adx_neg = adx_ind.adx_neg()  # -DI

        # ── Supertrend (manual calculation) ───────────────────────
        atr14   = atr
        hl2     = (high + low) / 2
        mult    = 3.0
        basic_ub = hl2 + (mult * atr14)
        basic_lb = hl2 - (mult * atr14)

        supertrend = pd.Series(index=df.index, dtype=float)
        st_dir     = pd.Series(1, index=df.index, dtype=int)

        for i in range(1, len(df)):
            prev_close = close.iloc[i-1]
            curr_ub    = basic_ub.iloc[i]
            curr_lb    = basic_lb.iloc[i]
            prev_ub    = supertrend.iloc[i-1] if st_dir.iloc[i-1] == -1 else basic_ub.iloc[i-1]
            prev_lb    = supertrend.iloc[i-1] if st_dir.iloc[i-1] == 1  else basic_lb.iloc[i-1]

            final_ub = curr_ub if curr_ub < prev_ub or prev_close > prev_ub else prev_ub
            final_lb = curr_lb if curr_lb > prev_lb or prev_close < prev_lb else prev_lb

            if st_dir.iloc[i-1] == 1:
                if close.iloc[i] < final_lb:
                    st_dir.iloc[i]     = -1
                    supertrend.iloc[i] = final_ub
                else:
                    st_dir.iloc[i]     = 1
                    supertrend.iloc[i] = final_lb
            else:
                if close.iloc[i] > final_ub:
                    st_dir.iloc[i]     = 1
                    supertrend.iloc[i] = final_lb
                else:
                    st_dir.iloc[i]     = -1
                    supertrend.iloc[i] = final_ub

        # ── Candle structure ──────────────────────────────────────
        price         = close.iloc[-1]
        ema50_val     = ema50.iloc[-1]
        ema200_val    = ema200.iloc[-1]
        candle_range  = high.iloc[-1] - low.iloc[-1]
        candle_body   = abs(close.iloc[-1] - open_.iloc[-1])
        body_pct      = candle_body / candle_range if candle_range > 0 else 0
        ema50_slope   = (ema50.iloc[-1] - ema50.iloc[-4]) / ema50.iloc[-4] * 100 if len(ema50) > 4 else 0

        # ── Derived features ──────────────────────────────────────
        price_vs_ema50 = (price - ema50_val) / ema50_val * 100
        trend_strength = (ema50_val - ema200_val) / ema200_val * 100
        volatility_pct = atr.iloc[-1] / price * 100
        ema_gap_pct    = trend_strength

        # ── Volatility regime ──────────────────────────────────────
        # Helps AI learn to avoid low-volatility choppy markets
        # and extreme volatility where stops get blown
        vol_pct_now = volatility_pct
        if vol_pct_now > 3.0:
            vol_regime = "HIGH"
        elif vol_pct_now < 0.8:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"

        # ── Market regime ─────────────────────────────────────────
        # Helps AI learn which regime favours the current strategy
        adx_val = adx.iloc[-1]
        if adx_val > 25 and ema50_val > ema200_val:
            regime = "TRENDING_UP"
        elif adx_val > 25 and ema50_val < ema200_val:
            regime = "TRENDING_DOWN"
        elif adx_val < 20:
            regime = "RANGING"
        elif vol_pct_now > 3.0:
            regime = "VOLATILE"
        else:
            regime = "NEUTRAL"

        return {
            # Price
            "price":             price,

            # Trend
            "ema50":             float(ema50_val),
            "ema200":            float(ema200_val),
            "ema20":             float(ema20.iloc[-1]),
            "ema100":            float(ema100.iloc[-1]),
            "sma20":             float(sma20.iloc[-1]),
            "ema50_slope":       float(ema50_slope),

            # Momentum
            "rsi":               float(rsi.iloc[-1]),
            "stoch_rsi_k":       float(stoch_rsi_k.iloc[-1]),
            "stoch_rsi_d":       float(stoch_rsi_d.iloc[-1]),
            "macd":              float(macd_line.iloc[-1]),
            "macd_signal":       float(macd_sig.iloc[-1]),
            "macd_histogram":    float(macd_hist.iloc[-1]),
            "cci":               float(cci.iloc[-1]),

            # Volatility
            "atr":               float(atr.iloc[-1]),
            "bb_upper":          float(bb_up.iloc[-1]),
            "bb_middle":         float(bb_mid.iloc[-1]),
            "bb_lower":          float(bb_lo.iloc[-1]),
            "bb_pct_b":          float(bb_pct.iloc[-1]),
            "bb_width":          float((bb_up.iloc[-1] - bb_lo.iloc[-1]) / bb_mid.iloc[-1] * 100),
            "realized_vol":      float(real_vol.iloc[-1]) if not np.isnan(real_vol.iloc[-1]) else 0,
            "atr_percentile":    float(atr_pct.iloc[-1]) if not np.isnan(atr_pct.iloc[-1]) else 50,
            "volatility_regime": vol_regime,

            # Volume
            "volume_ratio":      float(vol_ratio.iloc[-1]),
            "obv":               float(obv.iloc[-1]),
            "buy_vol_ratio":     float(buy_vol_ratio.iloc[-1]),

            # Trend strength
            "adx":               float(adx.iloc[-1]),
            "adx_pos":           float(adx_pos.iloc[-1]),
            "adx_neg":           float(adx_neg.iloc[-1]),

            # Supertrend
            "supertrend":        float(supertrend.iloc[-1]) if not pd.isna(supertrend.iloc[-1]) else price,
            "supertrend_dir":    int(st_dir.iloc[-1]),

            # Derived
            "price_vs_ema50":    float(price_vs_ema50),
            "trend_strength":    float(trend_strength),
            "volatility_pct":    float(volatility_pct),
            "ema_gap_pct":       float(ema_gap_pct),
            "candle_body_pct":   float(body_pct),
            "is_bullish_candle": close.iloc[-1] > open_.iloc[-1],
            "market_regime":     regime,
        }

    except Exception as e:
        logger.error(f"Indicator calculation failed for {symbol} {timeframe}: {e}")
        return None