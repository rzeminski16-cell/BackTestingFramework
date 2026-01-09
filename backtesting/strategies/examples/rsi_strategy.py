"""RSI (Relative Strength Index) Strategy."""

from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.strategies.base import Strategy, Parameter, ParameterType


class RSIStrategy(Strategy):
    """
    RSI Overbought/Oversold Strategy.

    Goes long when RSI drops below oversold level.
    Exits when RSI rises above overbought level.
    """

    name = "RSI Strategy"

    parameters = [
        Parameter(
            name="rsi_period",
            param_type=ParameterType.INTEGER,
            default=14,
            min_value=5,
            max_value=30,
            step=1,
            description="RSI calculation period",
        ),
        Parameter(
            name="oversold",
            param_type=ParameterType.FLOAT,
            default=30.0,
            min_value=10.0,
            max_value=40.0,
            step=5.0,
            description="Oversold threshold (buy signal)",
        ),
        Parameter(
            name="overbought",
            param_type=ParameterType.FLOAT,
            default=70.0,
            min_value=60.0,
            max_value=90.0,
            step=5.0,
            description="Overbought threshold (sell signal)",
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
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        position_size: float = 0.95,
    ):
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            position_size=position_size,
        )
        self._indicators = {}

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for a price series."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """Precompute RSI for all symbols."""
        self._indicators = {}

        for symbol, df in data.items():
            self._indicators[symbol] = {
                "rsi": self._calculate_rsi(df["close"], self.rsi_period),
            }

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        current_date: datetime,
    ) -> List[Dict]:
        """Generate RSI-based signals."""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.rsi_period + 1:
                continue

            rsi = self._indicators[symbol]["rsi"]

            if current_date not in rsi.index:
                continue

            current_rsi = rsi.loc[current_date]

            if pd.isna(current_rsi):
                continue

            in_position = symbol in portfolio.positions

            # Buy when oversold
            if current_rsi < self.oversold and not in_position:
                signals.append({
                    "action": "buy",
                    "symbol": symbol,
                    "size_pct": self.position_size,
                })

            # Sell when overbought
            elif current_rsi > self.overbought and in_position:
                signals.append({
                    "action": "sell",
                    "symbol": symbol,
                })

        return signals
