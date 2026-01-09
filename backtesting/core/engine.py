"""Main backtesting engine."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.core.position import PositionSide, Trade

if TYPE_CHECKING:
    from backtesting.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Container for backtest results."""

    portfolio: Portfolio
    trades: List[Trade]
    equity_curve: pd.DataFrame
    data: pd.DataFrame
    strategy_name: str
    parameters: Dict
    start_date: datetime
    end_date: datetime

    def __repr__(self) -> str:
        return (
            f"BacktestResult(strategy={self.strategy_name}, "
            f"trades={len(self.trades)}, "
            f"final_equity={self.equity_curve['equity'].iloc[-1]:.2f})"
        )


class BacktestEngine:
    """
    Main backtesting engine that runs strategies against historical data.

    Example usage:
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(strategy, data)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        strategy: "Strategy",
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run a backtest with the given strategy and data.

        Args:
            strategy: Strategy instance to test
            data: DataFrame with OHLCV data or dict of DataFrames for portfolio
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult containing all backtest information
        """
        # Handle single vs multiple securities
        if isinstance(data, pd.DataFrame):
            data_dict = {"default": data.copy()}
        else:
            data_dict = {k: v.copy() for k, v in data.items()}

        # Normalize column names
        for symbol, df in data_dict.items():
            data_dict[symbol] = self._normalize_columns(df)

        # Apply date filters
        for symbol, df in data_dict.items():
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            data_dict[symbol] = df

        # Initialize portfolio
        portfolio = Portfolio(initial_capital=self.initial_capital)

        # Initialize strategy
        strategy.initialize(data_dict)

        # Get combined index of all dates
        all_dates = sorted(set().union(*[set(df.index) for df in data_dict.values()]))

        # Run through each bar
        for i, current_date in enumerate(all_dates):
            # Build current bar data for all symbols
            current_data = {}
            current_prices = {}

            for symbol, df in data_dict.items():
                if current_date in df.index:
                    current_data[symbol] = df.loc[:current_date]
                    current_prices[symbol] = df.loc[current_date, "close"]

            # Check stop loss / take profit for open positions
            self._check_exits(portfolio, current_data, current_date)

            # Generate signals
            signals = strategy.generate_signals(current_data, portfolio, current_date)

            # Execute signals
            for signal in signals:
                self._execute_signal(portfolio, signal, current_data, current_date)

            # Record equity
            portfolio.record_equity(current_date, current_prices)

        # Close any remaining positions at last price
        final_prices = {
            symbol: df.iloc[-1]["close"]
            for symbol, df in data_dict.items()
        }
        final_date = all_dates[-1]

        for symbol in list(portfolio.positions.keys()):
            if symbol in final_prices:
                portfolio.close_position(symbol, final_prices[symbol], final_date)

        # Build combined data for result
        if len(data_dict) == 1:
            combined_data = list(data_dict.values())[0]
        else:
            combined_data = pd.concat(data_dict, axis=1)

        return BacktestResult(
            portfolio=portfolio,
            trades=portfolio.closed_trades,
            equity_curve=portfolio.get_equity_dataframe(),
            data=combined_data,
            strategy_name=strategy.name,
            parameters=strategy.get_parameter_values(),
            start_date=all_dates[0] if all_dates else None,
            end_date=all_dates[-1] if all_dates else None,
        )

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns exist
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df

    def _check_exits(
        self,
        portfolio: Portfolio,
        current_data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> None:
        """Check and execute stop loss / take profit exits."""
        for symbol in list(portfolio.positions.keys()):
            if symbol not in current_data:
                continue

            position = portfolio.positions[symbol]
            bar = current_data[symbol].iloc[-1]

            # Check stop loss
            if position.stop_loss is not None:
                if position.side == PositionSide.LONG and bar["low"] <= position.stop_loss:
                    exit_price = position.stop_loss * (1 - self.slippage)
                    portfolio.close_position(symbol, exit_price, current_date)
                    continue
                elif position.side == PositionSide.SHORT and bar["high"] >= position.stop_loss:
                    exit_price = position.stop_loss * (1 + self.slippage)
                    portfolio.close_position(symbol, exit_price, current_date)
                    continue

            # Check take profit
            if position.take_profit is not None:
                if position.side == PositionSide.LONG and bar["high"] >= position.take_profit:
                    exit_price = position.take_profit * (1 - self.slippage)
                    portfolio.close_position(symbol, exit_price, current_date)
                    continue
                elif position.side == PositionSide.SHORT and bar["low"] <= position.take_profit:
                    exit_price = position.take_profit * (1 + self.slippage)
                    portfolio.close_position(symbol, exit_price, current_date)
                    continue

    def _execute_signal(
        self,
        portfolio: Portfolio,
        signal: Dict,
        current_data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> None:
        """Execute a trading signal."""
        action = signal.get("action")
        symbol = signal.get("symbol", "default")

        if symbol not in current_data:
            return

        bar = current_data[symbol].iloc[-1]
        price = bar["close"]

        if action == "buy":
            # Apply slippage
            entry_price = price * (1 + self.slippage)

            # Calculate position size
            size = signal.get("size")
            if size is None:
                # Default: use percentage of portfolio
                pct = signal.get("size_pct", 0.95)
                size = (portfolio.cash * pct) / entry_price

            # Apply commission
            commission_cost = entry_price * size * self.commission
            if entry_price * size + commission_cost > portfolio.cash:
                size = (portfolio.cash - commission_cost) / entry_price

            if size > 0:
                portfolio.open_position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    price=entry_price,
                    quantity=size,
                    date=current_date,
                    stop_loss=signal.get("stop_loss"),
                    take_profit=signal.get("take_profit"),
                )
                portfolio.cash -= commission_cost

        elif action == "sell":
            if symbol in portfolio.positions:
                exit_price = price * (1 - self.slippage)
                trade = portfolio.close_position(symbol, exit_price, current_date)
                if trade:
                    commission_cost = exit_price * trade.quantity * self.commission
                    portfolio.cash -= commission_cost

        elif action == "short":
            entry_price = price * (1 - self.slippage)
            size = signal.get("size")
            if size is None:
                pct = signal.get("size_pct", 0.95)
                size = (portfolio.cash * pct) / entry_price

            commission_cost = entry_price * size * self.commission

            if size > 0:
                portfolio.open_position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    price=entry_price,
                    quantity=size,
                    date=current_date,
                    stop_loss=signal.get("stop_loss"),
                    take_profit=signal.get("take_profit"),
                )
                portfolio.cash -= commission_cost

        elif action == "cover":
            if symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                if position.side == PositionSide.SHORT:
                    exit_price = price * (1 + self.slippage)
                    trade = portfolio.close_position(symbol, exit_price, current_date)
                    if trade:
                        commission_cost = exit_price * trade.quantity * self.commission
                        portfolio.cash -= commission_cost
