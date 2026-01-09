"""Example trading strategies."""

from backtesting.strategies.examples.sma_crossover import SMACrossover
from backtesting.strategies.examples.rsi_strategy import RSIStrategy
from backtesting.strategies.examples.bollinger_bands import BollingerBandsStrategy

__all__ = ["SMACrossover", "RSIStrategy", "BollingerBandsStrategy"]
