from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    symbol: str
    side: str               # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float       # 0.0 – 1.0 (rule-based confidence before AI)
    reason: str
    indicators: dict        # full indicator snapshot for logging


class BaseStrategy(ABC):
    def __init__(self, version: str):
        self.version = version

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        indicators_1h: dict,
        indicators_4h: dict,
    ) -> Optional[Signal]:
        """
        Evaluate current market conditions and return a Signal or None.
        Must be synchronous — called on every candle close.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(version={self.version})"