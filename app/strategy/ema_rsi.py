"""
app/strategy/ema_rsi.py

Live strategy implementation. This file previously had its contents
accidentally overwritten with a copy of the backtest script, which meant
`from app.strategy.ema_rsi import EmaRsiStrategy` (used in app/engine.py
and tests/test_strategy.py) failed with an ImportError — the bot could not
start in this state.

This is now a thin wrapper around app.strategy.signal_logic, which is the
single shared source of truth for entry logic (also used directly by
scripts/backtest.py). Live and backtest can no longer drift apart.
"""
from typing import Optional

from app.strategy.base import BaseStrategy, Signal
from app.strategy import signal_logic


class EmaRsiStrategy(BaseStrategy):
    def evaluate(
        self,
        symbol: str,
        indicators_1h: dict,
        indicators_4h: dict,
    ) -> Optional[Signal]:
        """
        indicators_1h: indicators from the faster/entry (signal) timeframe.
        indicators_4h: indicators from the slower/trend-confirmation timeframe.
        (Parameter names kept for BaseStrategy compatibility — the actual
        timeframes used are whatever TIMEFRAMES.signal / TIMEFRAMES.trend
        resolve to in app/config.py, e.g. 15m / 1h.)
        """
        result = signal_logic.evaluate_entry(indicators_1h, indicators_4h)
        if result is None:
            return None

        return Signal(
            symbol=symbol,
            side=result["side"],
            entry_price=result["entry"],
            stop_loss=result["sl"],
            take_profit=result["tp"],
            confidence=result["confidence"],
            reason=result["reason"],
            indicators={
                **indicators_1h,
                "trend_ema50": indicators_4h.get("ema50"),
                "trend_ema200": indicators_4h.get("ema200"),
            },
        )
