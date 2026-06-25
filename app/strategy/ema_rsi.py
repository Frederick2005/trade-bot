from typing import Optional
from loguru import logger
from app.strategy.base import BaseStrategy, Signal
from app.strategy.params import get_active_params


class EmaRsiStrategy(BaseStrategy):
    """
    EMA50/200 crossover + RSI momentum + 4H trend confluence.

    Entry conditions (LONG):
      - 4H: EMA50 > EMA200 (uptrend confirmed)
      - 1H: EMA50 > EMA200
      - 1H: Price within 1% of EMA50 (pullback zone)
      - 1H: RSI between rsi_lower and rsi_upper
      - 1H: Bullish candle (close > open, body > 60% of range)
      - Volume above min_volume_ratio × 20-period average

    SHORT is the mirror of the above.
    """

    def evaluate(
        self,
        symbol: str,
        indicators_1h: dict,
        indicators_4h: dict,
    ) -> Optional[Signal]:
        params = get_active_params()

        # ── Extract values ──────────────────────────────────────────
        price      = indicators_1h["price"]
        ema50_1h   = indicators_1h["ema50"]
        ema200_1h  = indicators_1h["ema200"]
        rsi_1h     = indicators_1h["rsi"]
        atr_1h     = indicators_1h["atr"]
        vol_ratio  = indicators_1h["volume_ratio"]
        body_pct   = indicators_1h["candle_body_pct"]
        bullish    = indicators_1h["is_bullish_candle"]

        ema50_4h   = indicators_4h["ema50"]
        ema200_4h  = indicators_4h["ema200"]

        # ── 4H trend filter ─────────────────────────────────────────
        trend_4h_bull = ema50_4h > ema200_4h

        # ── RSI zone check ──────────────────────────────────────────
        rsi_in_zone = params.rsi_lower <= rsi_1h <= params.rsi_upper

        # ── Volume filter ───────────────────────────────────────────
        vol_ok = vol_ratio >= params.min_volume_ratio

        # ── Candle confirmation ─────────────────────────────────────
        strong_candle = body_pct >= 0.60

        # ── Pullback to EMA50 (within 1%) ──────────────────────────
        near_ema50 = abs(indicators_1h["price_vs_ema50"]) <= 1.0

        # ── LONG signal ─────────────────────────────────────────────
        if (
            trend_4h_bull
            and ema50_1h > ema200_1h
            and near_ema50
            and rsi_in_zone
            and strong_candle
            and bullish
            and vol_ok
        ):
            sl  = price - (atr_1h * params.atr_multiplier)
            tp  = price + ((price - sl) * params.min_rr)
            rr  = (tp - price) / (price - sl)

            if rr < params.min_rr:
                logger.debug(f"{symbol} LONG skipped: RR {rr:.2f} < {params.min_rr}")
                return None

            logger.info(
                f"LONG signal: {symbol} | price={price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f} RSI={rsi_1h:.1f}"
            )
            return Signal(
                symbol=symbol,
                side="LONG",
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                confidence=self._confidence(rsi_1h, vol_ratio, body_pct, params),
                reason=(
                    f"4H uptrend, 1H EMA50>200, pullback to EMA50, "
                    f"RSI={rsi_1h:.1f}, vol_ratio={vol_ratio:.2f}"
                ),
                indicators={
                    **indicators_1h,
                    "ema50_4h": ema50_4h,
                    "ema200_4h": ema200_4h,
                },
            )

        # ── SHORT signal ─────────────────────────────────────────────
        if (
            not trend_4h_bull
            and ema50_1h < ema200_1h
            and near_ema50
            and rsi_in_zone
            and strong_candle
            and not bullish
            and vol_ok
        ):
            sl  = price + (atr_1h * params.atr_multiplier)
            tp  = price - ((sl - price) * params.min_rr)
            rr  = (price - tp) / (sl - price)

            if rr < params.min_rr:
                logger.debug(f"{symbol} SHORT skipped: RR {rr:.2f} < {params.min_rr}")
                return None

            logger.info(
                f"SHORT signal: {symbol} | price={price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f} RSI={rsi_1h:.1f}"
            )
            return Signal(
                symbol=symbol,
                side="SHORT",
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                confidence=self._confidence(rsi_1h, vol_ratio, body_pct, params),
                reason=(
                    f"4H downtrend, 1H EMA50<200, pullback to EMA50, "
                    f"RSI={rsi_1h:.1f}, vol_ratio={vol_ratio:.2f}"
                ),
                indicators={
                    **indicators_1h,
                    "ema50_4h": ema50_4h,
                    "ema200_4h": ema200_4h,
                },
            )

        return None

    def _confidence(
        self,
        rsi: float,
        vol_ratio: float,
        body_pct: float,
        params,
    ) -> float:
        score = 0.5
        rsi_mid = (params.rsi_lower + params.rsi_upper) / 2
        rsi_score = 1 - abs(rsi - rsi_mid) / (params.rsi_upper - params.rsi_lower)
        score += rsi_score * 0.2
        score += min(vol_ratio / 2, 0.15)
        score += body_pct * 0.15
        return round(min(score, 1.0), 4)