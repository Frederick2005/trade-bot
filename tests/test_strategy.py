import pytest
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.base import Signal


@pytest.fixture
def strategy():
    return EmaRsiStrategy(version="test")


def _make_indicators(
    price=100.0, ema50=99.0, ema200=95.0, rsi=52.0,
    atr=1.5, vol_ratio=1.2, body_pct=0.7, bullish=True,
    price_vs_ema50=1.0, trend_strength=4.2,
    volatility_pct=1.5, ema_gap_pct=4.2, ema50_slope=0.1,
) -> dict:
    return {
        "price": price, "ema50": ema50, "ema200": ema200,
        "rsi": rsi, "atr": atr, "volume_ratio": vol_ratio,
        "candle_body_pct": body_pct, "is_bullish_candle": bullish,
        "price_vs_ema50": price_vs_ema50, "trend_strength": trend_strength,
        "volatility_pct": volatility_pct, "ema_gap_pct": ema_gap_pct,
        "ema50_slope": ema50_slope,
    }


def _make_4h_bull(body_pct=0.85) -> dict:
    # body_pct defaults to 0.85 (was implicitly 0.7 via _make_indicators'
    # own default) — must be >= TREND_CANDLE_MIN_BODY_PCT (0.8) for tests
    # that expect a valid signal, since evaluate_entry() now gates on the
    # TREND-timeframe candle's body strength too (see signal_logic.py).
    return _make_indicators(ema50=105.0, ema200=100.0, body_pct=body_pct)


def _make_4h_bear(body_pct=0.85) -> dict:
    return _make_indicators(ema50=95.0, ema200=100.0, body_pct=body_pct)


class TestLongSignal:
    def test_valid_long(self, strategy):
        ind_1h = _make_indicators()
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is not None
        assert signal.side == "LONG"

    def test_no_signal_4h_downtrend(self, strategy):
        ind_1h = _make_indicators()
        ind_4h = _make_4h_bear()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None

    def test_no_signal_rsi_too_high(self, strategy):
        ind_1h = _make_indicators(rsi=65.0)
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None

    def test_no_signal_rsi_too_low(self, strategy):
        ind_1h = _make_indicators(rsi=40.0)
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None

    def test_no_signal_bearish_candle(self, strategy):
        ind_1h = _make_indicators(bullish=False)
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None

    def test_no_signal_price_far_from_ema50(self, strategy):
        ind_1h = _make_indicators(price_vs_ema50=3.0)
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None

    def test_no_signal_weak_trend_candle(self, strategy):
        # Trend-timeframe candle body below TREND_CANDLE_MIN_BODY_PCT (0.8)
        # should block the signal even though everything else is valid —
        # this is the gate added from real backtest evidence (see
        # signal_logic.py's TREND_CANDLE_MIN_BODY_PCT comment).
        ind_1h = _make_indicators()
        ind_4h = _make_4h_bull(body_pct=0.5)
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None

    def test_signal_has_correct_structure(self, strategy):
        ind_1h = _make_indicators()
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert isinstance(signal, Signal)
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.reason != ""


class TestShortSignal:
    def test_valid_short(self, strategy):
        ind_1h = _make_indicators(
            ema50=95.0, ema200=100.0, bullish=False,
            price_vs_ema50=-0.5, rsi=52.0,
        )
        ind_4h = _make_4h_bear()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is not None
        assert signal.side == "SHORT"
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit < signal.entry_price

    def test_no_short_in_uptrend(self, strategy):
        ind_1h = _make_indicators(ema50=95.0, ema200=100.0, bullish=False)
        ind_4h = _make_4h_bull()   # 4H says uptrend
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        assert signal is None


class TestConfidence:
    def test_confidence_in_range(self, strategy):
        ind_1h = _make_indicators()
        ind_4h = _make_4h_bull()
        signal = strategy.evaluate("BTCUSDT", ind_1h, ind_4h)
        if signal:
            assert 0.0 <= signal.confidence <= 1.0