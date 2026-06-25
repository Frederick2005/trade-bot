from collections import defaultdict
import pandas as pd
import ta
from loguru import logger


# In-memory candle store keyed by (symbol, timeframe)
_candles: dict[tuple, pd.DataFrame] = defaultdict(pd.DataFrame)

# Minimum candles needed before indicators are valid
MIN_CANDLES = 210


def update(symbol: str, timeframe: str, candle: dict) -> None:
    key = (symbol, timeframe)
    row = pd.DataFrame([{
        "open_time": pd.to_datetime(candle["open_time"]),
        "open":      float(candle["open"]),
        "high":      float(candle["high"]),
        "low":       float(candle["low"]),
        "close":     float(candle["close"]),
        "volume":    float(candle["volume"]),
    }])

    existing = _candles[key]
    if existing.empty:
        _candles[key] = row
    else:
        # Replace last candle if same open_time (candle update), else append
        if existing.iloc[-1]["open_time"] == row.iloc[0]["open_time"]:
            _candles[key] = pd.concat(
                [existing.iloc[:-1], row], ignore_index=True
            )
        else:
            _candles[key] = pd.concat(
                [existing, row], ignore_index=True
            )

    # Keep rolling window — 500 candles max
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
    df = get_df(symbol, timeframe)
    return len(df) >= MIN_CANDLES


def get_indicators(symbol: str, timeframe: str) -> dict | None:
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

        # Trend
        ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()

        # Momentum
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

        # Volatility
        atr = ta.volatility.AverageTrueRange(
            high, low, close, window=14
        ).average_true_range()

        # Volume ratio vs 20-period average
        vol_avg    = volume.rolling(20).mean()
        vol_ratio  = (volume / vol_avg).iloc[-1]

        price      = close.iloc[-1]
        ema50_val  = ema50.iloc[-1]
        ema200_val = ema200.iloc[-1]
        rsi_val    = rsi.iloc[-1]
        atr_val    = atr.iloc[-1]

        # Derived
        price_vs_ema50  = (price - ema50_val) / ema50_val * 100
        trend_strength  = (ema50_val - ema200_val) / ema200_val * 100
        volatility_pct  = atr_val / price * 100
        ema_gap_pct     = (ema50_val - ema200_val) / ema200_val * 100

        last = df.iloc[-1]
        candle_range    = last["high"] - last["low"]
        candle_body     = abs(last["close"] - last["open"])
        candle_body_pct = candle_body / candle_range if candle_range > 0 else 0

        # EMA slope (change over last 3 bars)
        ema50_slope = (ema50_val - ema50.iloc[-4]) / ema50.iloc[-4] * 100

        return {
            "price":         price,
            "ema50":         ema50_val,
            "ema200":        ema200_val,
            "rsi":           rsi_val,
            "atr":           atr_val,
            "volume_ratio":  vol_ratio,
            "price_vs_ema50": price_vs_ema50,
            "trend_strength": trend_strength,
            "volatility_pct": volatility_pct,
            "ema_gap_pct":   ema_gap_pct,
            "candle_body_pct": candle_body_pct,
            "ema50_slope":   ema50_slope,
            "is_bullish_candle": last["close"] > last["open"],
        }

    except Exception as e:
        logger.error(f"Indicator calculation failed for {symbol} {timeframe}: {e}")
        return None