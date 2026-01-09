"""Simple Moving Average Crossover Strategy."""

from datetime import datetime
from typing import Dict, List
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.strategies.base import Strategy, Parameter, ParameterType


class SMACrossover(Strategy):
    """
    SMA Crossover Strategy.

    Goes long when fast SMA crosses above slow SMA.
    Exits when fast SMA crosses below slow SMA.
    """

    name = "SMA Crossover"

    parameters = [
        Parameter(
            name="fast_period",
            param_type=ParameterType.INTEGER,
            default=10,
            min_value=5,
            max_value=50,
            step=5,
            description="Period for fast moving average",
        ),
        Parameter(
            name="slow_period",
            param_type=ParameterType.INTEGER,
            default=30,
            min_value=20,
            max_value=100,
            step=10,
            description="Period for slow moving average",
        ),
        Parameter(
            name="position_size",
            param_type=ParameterType.FLOAT,
            default=0.95,
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            description="Fraction of portfolio to use per trade",
        ),
    ]

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        position_size: float = 0.95,
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            position_size=position_size,
        )
        self._indicators = {}

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """Precompute SMAs for all symbols."""
        self._indicators = {}

        for symbol, df in data.items():
            self._indicators[symbol] = {
                "fast_sma": df["close"].rolling(window=self.fast_period).mean(),
                "slow_sma": df["close"].rolling(window=self.slow_period).mean(),
            }

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        current_date: datetime,
    ) -> List[Dict]:
        """Generate crossover signals."""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.slow_period + 1:
                continue

            # Get indicator values
            fast_sma = self._indicators[symbol]["fast_sma"]
            slow_sma = self._indicators[symbol]["slow_sma"]

            if current_date not in fast_sma.index:
                continue

            # Get current and previous values
            idx = fast_sma.index.get_loc(current_date)
            if idx < 1:
                continue

            prev_date = fast_sma.index[idx - 1]

            fast_current = fast_sma.loc[current_date]
            fast_prev = fast_sma.loc[prev_date]
            slow_current = slow_sma.loc[current_date]
            slow_prev = slow_sma.loc[prev_date]

            # Skip if NaN
            if pd.isna(fast_current) or pd.isna(slow_current):
                continue
            if pd.isna(fast_prev) or pd.isna(slow_prev):
                continue

            # Check for crossover
            in_position = symbol in portfolio.positions

            # Bullish crossover
            if fast_prev <= slow_prev and fast_current > slow_current:
                if not in_position:
                    signals.append({
                        "action": "buy",
                        "symbol": symbol,
                        "size_pct": self.position_size,
                    })

            # Bearish crossover
            elif fast_prev >= slow_prev and fast_current < slow_current:
                if in_position:
                    signals.append({
                        "action": "sell",
                        "symbol": symbol,
                    })

        return signals
