"""
Single security backtesting engine.
"""
import pandas as pd
from typing import List, Optional, Callable
from datetime import datetime

from ..Strategy.base_strategy import BaseStrategy
from ..Strategy.strategy_context import StrategyContext
from ..Models.signal import Signal, SignalType
from ..Models.order import Order, OrderSide, OrderType
from ..Config.config import BacktestConfig
from .position_manager import PositionManager
from .trade_executor import TradeExecutor
from .backtest_result import BacktestResult


class SingleSecurityEngine:
    """
    Backtesting engine for single security.

    Executes strategy bar-by-bar with:
    - No lookahead bias (strategies only see past data)
    - Execution at close prices
    - Commission tracking
    - Detailed trade logging
    - Stop loss and take profit checks
    - Partial exits
    - Trailing stops
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.position_manager = PositionManager()
        self.trade_executor = TradeExecutor(config.commission)

    def run(self, symbol: str, data: pd.DataFrame, strategy: BaseStrategy,
            progress_callback: Optional[Callable[[int, int], None]] = None) -> BacktestResult:
        """
        Run backtest for a single security.

        Args:
            symbol: Security symbol
            data: Price/indicator data (must be sorted by date)
            strategy: Trading strategy
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            BacktestResult with trades and equity curve

        Raises:
            ValueError: If data is invalid
        """
        # Validate data
        if len(data) == 0:
            raise ValueError(f"No data provided for {symbol}")

        if 'date' not in data.columns or 'close' not in data.columns:
            raise ValueError("Data must contain 'date' and 'close' columns")

        # Initialize tracking variables
        capital = self.config.initial_capital
        equity_history = []

        # Filter by date range if specified
        data = self._filter_by_date_range(data)

        if len(data) == 0:
            raise ValueError(f"No data for {symbol} in specified date range")

        # Process each bar
        total_bars = len(data)
        for i in range(total_bars):
            # Update progress callback (every 50 bars or at the end)
            if progress_callback and (i % 50 == 0 or i == total_bars - 1):
                progress_callback(i + 1, total_bars)

            current_bar = data.iloc[i]
            current_date = current_bar['date']
            current_price = current_bar['close']

            # Create strategy context (point-in-time data only)
            historical_data = data.iloc[:i+1].copy()
            position_value = self.position_manager.get_position_value(current_price)
            total_equity = capital + position_value

            context = StrategyContext(
                data=historical_data,
                current_index=i,
                current_price=current_price,
                current_date=current_date,
                position=self.position_manager.get_position(),
                available_capital=capital,
                total_equity=total_equity
            )

            # Check stop loss first (before strategy signal)
            if self.position_manager.has_position:
                if strategy.should_check_stop_loss(context):
                    if self.position_manager.check_stop_loss(current_price):
                        # Stop loss hit - close position
                        capital = self._close_position(
                            current_date, current_price, "Stop loss hit", capital
                        )
                        position_value = 0
                        total_equity = capital
                        # Record equity and continue to next bar
                        equity_history.append({
                            'date': current_date,
                            'equity': total_equity,
                            'capital': capital,
                            'position_value': position_value
                        })
                        continue

                # Check take profit
                if strategy.should_check_take_profit(context):
                    if self.position_manager.check_take_profit(current_price):
                        # Take profit hit - close position
                        capital = self._close_position(
                            current_date, current_price, "Take profit hit", capital
                        )
                        position_value = 0
                        total_equity = capital
                        # Record equity and continue to next bar
                        equity_history.append({
                            'date': current_date,
                            'equity': total_equity,
                            'capital': capital,
                            'position_value': position_value
                        })
                        continue

                # Check for trailing stop adjustment
                new_stop = strategy.should_adjust_stop(context)
                if new_stop is not None:
                    # Only allow stop to move in favorable direction (up for LONG)
                    current_stop = self.position_manager.position.stop_loss
                    if current_stop is None or new_stop > current_stop:
                        self.position_manager.adjust_stop_loss(new_stop)

                # Check for partial exit
                partial_fraction = strategy.should_partial_exit(context)
                if partial_fraction is not None and partial_fraction > 0:
                    capital = self._partial_exit(
                        current_date, current_price, partial_fraction,
                        "Partial profit taking", capital
                    )
                    position_value = self.position_manager.get_position_value(current_price)
                    total_equity = capital + position_value

            # Generate signal from strategy
            signal = strategy.generate_signal(context)

            # Process signal
            if signal.type == SignalType.BUY and not self.position_manager.has_position:
                # Open new position
                capital = self._open_position(
                    symbol, current_date, current_price, signal, strategy, context, capital
                )
                position_value = self.position_manager.get_position_value(current_price)
                total_equity = capital + position_value

            elif signal.type == SignalType.SELL and self.position_manager.has_position:
                # Close position
                capital = self._close_position(
                    current_date, current_price, signal.reason or "Strategy exit signal", capital
                )
                position_value = 0
                total_equity = capital

            elif signal.type == SignalType.PARTIAL_EXIT and self.position_manager.has_position:
                # Partial exit
                capital = self._partial_exit(
                    current_date, current_price, signal.size, signal.reason, capital
                )
                position_value = self.position_manager.get_position_value(current_price)
                total_equity = capital + position_value

            elif signal.type == SignalType.ADJUST_STOP and self.position_manager.has_position:
                # Adjust stop loss
                if signal.new_stop_loss is not None:
                    current_stop = self.position_manager.position.stop_loss
                    if current_stop is None or signal.new_stop_loss > current_stop:
                        self.position_manager.adjust_stop_loss(signal.new_stop_loss)

            # Record equity
            position_value = self.position_manager.get_position_value(current_price)
            total_equity = capital + position_value

            equity_history.append({
                'date': current_date,
                'equity': total_equity,
                'capital': capital,
                'position_value': position_value
            })

        # Close any remaining position at end
        if self.position_manager.has_position:
            final_bar = data.iloc[-1]
            capital = self._close_position(
                final_bar['date'], final_bar['close'],
                "End of backtest period", capital
            )

        # Create equity curve
        equity_df = pd.DataFrame(equity_history)

        # Calculate returns
        final_equity = capital
        total_return = final_equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100

        # Create result
        result = BacktestResult(
            symbol=symbol,
            strategy_name=strategy.get_name(),
            trades=self.trade_executor.get_trades(),
            equity_curve=equity_df,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            strategy_params=strategy.get_parameters()
        )

        return result

    def _open_position(self, symbol: str, date: datetime, price: float,
                      signal: Signal, strategy: BaseStrategy,
                      context: StrategyContext, capital: float) -> float:
        """
        Open a new position.

        Args:
            symbol: Security symbol
            date: Entry date
            price: Entry price
            signal: BUY signal
            strategy: Strategy instance
            context: Current context
            capital: Available capital

        Returns:
            Remaining capital after entry
        """
        # Calculate position size
        quantity = strategy.position_size(context, signal)

        if quantity <= 0:
            return capital

        # Check position size limit
        max_capital = capital * self.config.position_size_limit
        order_value = quantity * price

        if order_value > max_capital:
            # Adjust quantity to fit within limit
            quantity = max_capital / price

        # Create entry order
        entry_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            timestamp=date,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reason=signal.reason
        )

        # Execute order
        entry_commission = self.trade_executor.execute_order(entry_order)
        total_cost = entry_order.total_value() + entry_commission

        # Check if we have enough capital
        if total_cost > capital:
            # Insufficient capital - skip this trade
            return capital

        # Open position
        self.position_manager.open_position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_reason=signal.reason,
            commission_paid=entry_commission
        )

        # Deduct from capital
        capital -= total_cost

        return capital

    def _close_position(self, date: datetime, price: float,
                       reason: str, capital: float) -> float:
        """
        Close current position.

        Args:
            date: Exit date
            price: Exit price
            reason: Exit reason
            capital: Current capital

        Returns:
            Updated capital after exit
        """
        position = self.position_manager.get_position()
        quantity = position.current_quantity

        # Create exit order
        exit_order = Order(
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            timestamp=date,
            reason=reason
        )

        # Execute order
        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission

        # Add proceeds to capital
        capital += proceeds

        # Create trade record
        self.trade_executor.create_trade(
            position=position,
            exit_date=date,
            exit_price=price,
            exit_reason=reason,
            exit_commission=exit_commission
        )

        # Close position
        self.position_manager.close_position()

        return capital

    def _partial_exit(self, date: datetime, price: float, fraction: float,
                     reason: str, capital: float) -> float:
        """
        Partially exit position.

        Args:
            date: Exit date
            price: Exit price
            fraction: Fraction to exit (0.0-1.0)
            reason: Exit reason
            capital: Current capital

        Returns:
            Updated capital
        """
        position = self.position_manager.get_position()
        exit_quantity = position.current_quantity * fraction

        # Create exit order
        exit_order = Order(
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=exit_quantity,
            order_type=OrderType.MARKET,
            price=price,
            timestamp=date,
            reason=reason
        )

        # Execute order
        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission

        # Add proceeds to capital
        capital += proceeds

        # Record partial exit
        self.position_manager.add_partial_exit(
            exit_date=date,
            quantity=exit_quantity,
            price=price,
            reason=reason,
            commission_paid=exit_commission
        )

        return capital

    def _filter_by_date_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data by configured date range.

        Args:
            data: Full data

        Returns:
            Filtered data
        """
        filtered = data.copy()

        if self.config.start_date:
            filtered = filtered[filtered['date'] >= self.config.start_date]

        if self.config.end_date:
            filtered = filtered[filtered['date'] <= self.config.end_date]

        return filtered.reset_index(drop=True)
