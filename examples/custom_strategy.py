#!/usr/bin/env python3
"""
Example: Creating a Custom Strategy

This example shows how to create a custom trading strategy with
proper parameter definitions for optimization.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from backtesting import BacktestEngine, Optimizer
from backtesting.optimization import OptimizationConfig, ParameterRange
from backtesting.strategies.base import Strategy, Parameter, ParameterType
from backtesting.core.portfolio import Portfolio
from backtesting.data import DataLoader
from backtesting.metrics import PerformanceMetrics


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Goes long when MACD line crosses above signal line.
    Exits when MACD line crosses below signal line.

    This demonstrates how to create a custom strategy with
    proper parameter definitions for optimization.
    """

    name = "MACD Strategy"

    # Define all parameters with their metadata
    parameters = [
        Parameter(
            name="fast_period",
            param_type=ParameterType.INTEGER,
            default=12,
            min_value=5,
            max_value=20,
            step=1,
            description="Fast EMA period",
        ),
        Parameter(
            name="slow_period",
            param_type=ParameterType.INTEGER,
            default=26,
            min_value=20,
            max_value=40,
            step=2,
            description="Slow EMA period",
        ),
        Parameter(
            name="signal_period",
            param_type=ParameterType.INTEGER,
            default=9,
            min_value=5,
            max_value=15,
            step=1,
            description="Signal line EMA period",
        ),
        Parameter(
            name="position_size",
            param_type=ParameterType.FLOAT,
            default=0.95,
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            description="Fraction of portfolio per trade",
        ),
    ]

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        position_size: float = 0.95,
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            position_size=position_size,
        )
        self._indicators = {}

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """Precompute MACD indicators."""
        self._indicators = {}

        for symbol, df in data.items():
            fast_ema = self._calculate_ema(df["close"], self.fast_period)
            slow_ema = self._calculate_ema(df["close"], self.slow_period)

            macd_line = fast_ema - slow_ema
            signal_line = self._calculate_ema(macd_line, self.signal_period)
            histogram = macd_line - signal_line

            self._indicators[symbol] = {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram,
            }

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        current_date: datetime,
    ) -> List[Dict]:
        """Generate MACD crossover signals."""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.slow_period + self.signal_period:
                continue

            indicators = self._indicators[symbol]
            macd = indicators["macd"]
            signal = indicators["signal"]

            if current_date not in macd.index:
                continue

            idx = macd.index.get_loc(current_date)
            if idx < 1:
                continue

            prev_date = macd.index[idx - 1]

            macd_curr = macd.loc[current_date]
            macd_prev = macd.loc[prev_date]
            signal_curr = signal.loc[current_date]
            signal_prev = signal.loc[prev_date]

            if pd.isna(macd_curr) or pd.isna(signal_curr):
                continue
            if pd.isna(macd_prev) or pd.isna(signal_prev):
                continue

            in_position = symbol in portfolio.positions

            # Bullish crossover: MACD crosses above signal
            if macd_prev <= signal_prev and macd_curr > signal_curr:
                if not in_position:
                    signals.append({
                        "action": "buy",
                        "symbol": symbol,
                        "size_pct": self.position_size,
                    })

            # Bearish crossover: MACD crosses below signal
            elif macd_prev >= signal_prev and macd_curr < signal_curr:
                if in_position:
                    signals.append({
                        "action": "sell",
                        "symbol": symbol,
                    })

        return signals


def main():
    print("=" * 60)
    print("Custom Strategy Example - MACD Strategy")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    loader = DataLoader()
    data = loader.generate_sample_data(
        days=504,
        start_price=100.0,
        volatility=0.015,
        drift=0.0003,
        seed=42,
    )
    print(f"Loaded {len(data)} bars")
    print()

    # Show strategy parameters
    print("MACD Strategy Parameters:")
    for param in MACDStrategy.parameters:
        print(f"  {param.name}: {param.description}")
        print(f"    Type: {param.param_type.value}")
        print(f"    Default: {param.default}")
        if param.min_value is not None:
            print(f"    Range: {param.min_value} - {param.max_value}")
        print()

    # Run basic backtest
    print("Running backtest with default parameters...")
    strategy = MACDStrategy()
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run(strategy, data)

    metrics = PerformanceMetrics(result)
    print(metrics.summary())
    print()

    # Run optimization
    print("Running parameter optimization...")
    print()

    config = OptimizationConfig(
        strategy_class=MACDStrategy,
        parameters_to_optimize=[
            ParameterRange("fast_period", values=[8, 10, 12, 14, 16]),
            ParameterRange("slow_period", values=[22, 26, 30, 34]),
            ParameterRange("signal_period", values=[7, 9, 11]),
        ],
        control_values={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "position_size": 0.95,
        },
        metrics=["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
    )

    def progress(param_name, current, total):
        print(f"\r  Optimizing {param_name}: {current}/{total}", end="", flush=True)

    optimizer = Optimizer(config, progress_callback=progress)
    results = optimizer.run(data)
    print()
    print()

    # Show best values
    print("Best parameter values by metric:")
    for param_name, param_results in results.parameter_results.items():
        print(f"\n  {param_name}:")
        for metric in config.metrics:
            best = param_results.get_best_value(metric)
            print(f"    Best for {metric}: {best}")

    # Export results
    output_file = "macd_optimization.xlsx"
    optimizer.export_to_excel(results, output_file)
    print(f"\nResults exported to: {output_file}")

    print()
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
