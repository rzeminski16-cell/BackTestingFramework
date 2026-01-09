"""Bollinger Bands Strategy."""

from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.strategies.base import Strategy, Parameter, ParameterType


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Mean Reversion Strategy.

    Goes long when price touches lower band.
    Exits when price touches upper band or middle band.
    """

    name = "Bollinger Bands"

    parameters = [
        Parameter(
            name="bb_period",
            param_type=ParameterType.INTEGER,
            default=20,
            min_value=10,
            max_value=50,
            step=5,
            description="Period for Bollinger Bands calculation",
        ),
        Parameter(
            name="bb_std",
            param_type=ParameterType.FLOAT,
            default=2.0,
            min_value=1.0,
            max_value=3.0,
            step=0.25,
            description="Number of standard deviations for bands",
        ),
        Parameter(
            name="exit_at_middle",
            param_type=ParameterType.BOOLEAN,
            default=False,
            description="Exit at middle band instead of upper band",
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
        bb_period: int = 20,
        bb_std: float = 2.0,
        exit_at_middle: bool = False,
        position_size: float = 0.95,
    ):
        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            exit_at_middle=exit_at_middle,
            position_size=position_size,
        )
        self._indicators = {}

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """Precompute Bollinger Bands for all symbols."""
        self._indicators = {}

        for symbol, df in data.items():
            middle = df["close"].rolling(window=self.bb_period).mean()
            std = df["close"].rolling(window=self.bb_period).std()

            self._indicators[symbol] = {
                "middle": middle,
                "upper": middle + (std * self.bb_std),
                "lower": middle - (std * self.bb_std),
            }

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        current_date: datetime,
    ) -> List[Dict]:
        """Generate Bollinger Bands signals."""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.bb_period:
                continue

            if current_date not in df.index:
                continue

            indicators = self._indicators[symbol]
            current_price = df.loc[current_date, "close"]

            middle = indicators["middle"].loc[current_date]
            upper = indicators["upper"].loc[current_date]
            lower = indicators["lower"].loc[current_date]

            if pd.isna(middle) or pd.isna(upper) or pd.isna(lower):
                continue

            in_position = symbol in portfolio.positions

            # Buy when price touches lower band
            if current_price <= lower and not in_position:
                signals.append({
                    "action": "buy",
                    "symbol": symbol,
                    "size_pct": self.position_size,
                })

            # Sell when price touches upper/middle band
            elif in_position:
                exit_level = middle if self.exit_at_middle else upper
                if current_price >= exit_level:
                    signals.append({
                        "action": "sell",
                        "symbol": symbol,
                    })

        return signals
