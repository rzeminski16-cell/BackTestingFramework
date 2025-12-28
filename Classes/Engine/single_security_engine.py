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
from ..Models.trade_direction import TradeDirection
from ..Config.config import BacktestConfig
from .position_manager import PositionManager
from .trade_executor import TradeExecutor
from .backtest_result import BacktestResult
from ..Data.currency_converter import CurrencyConverter
from ..Data.security_registry import SecurityRegistry
from ..Data.historical_data_view import HistoricalDataView


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

    def __init__(self, config: BacktestConfig,
                 currency_converter: Optional[CurrencyConverter] = None,
                 security_registry: Optional[SecurityRegistry] = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            currency_converter: Optional currency converter for multi-currency support
            security_registry: Optional security registry for metadata
        """
        self.config = config
        self.position_manager = PositionManager()
        # Set commission rate for break-even calculation
        self.position_manager.commission_rate = config.commission.value if config.commission else 0.0
        self.trade_executor = TradeExecutor(config.commission)
        self.currency_converter = currency_converter
        self.security_registry = security_registry
        self._fx_rate_warnings = set()  # Track which currency pairs have been warned about

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

        # PERFORMANCE OPTIMIZATION: Pre-calculate all indicators ONCE before backtest
        # This eliminates O(n²) complexity from repeated calculations
        data = strategy.prepare_data(data)

        # Process each bar
        total_bars = len(data)
        for i in range(total_bars):
            # Update progress callback (every 50 bars or at the end)
            if progress_callback and (i % 50 == 0 or i == total_bars - 1):
                progress_callback(i + 1, total_bars)

            current_bar = data.iloc[i]
            current_date = current_bar['date']
            current_price = current_bar['close']

            # DATA LEAKAGE FIX: Only pass historical data up to current bar
            # This prevents strategies from accessing future data and eliminates look-ahead bias
            # HistoricalDataView enforces look-ahead protection without expensive copying
            # The view allows access to bars 0 to i (inclusive), so current_index=i is the last valid index
            # PERFORMANCE: This avoids O(n²) memory operations from copying data every bar
            historical_data = HistoricalDataView(data, valid_end_index=i)

            position_value = self.position_manager.get_position_value(current_price)
            # Convert position value to base currency (GBP)
            position_value = self._convert_to_base_currency(position_value, symbol, current_date)
            total_equity = capital + position_value

            # Get FX rate for currency conversion
            fx_rate = self._get_fx_rate(symbol, current_date)

            context = StrategyContext(
                data=historical_data,
                current_index=i,
                current_price=current_price,
                current_date=current_date,
                position=self.position_manager.get_position(),
                available_capital=capital,
                total_equity=total_equity,
                symbol=symbol,
                fx_rate=fx_rate
            )

            # Check stop loss first (before strategy signal)
            if self.position_manager.has_position:
                if strategy.should_check_stop_loss(context):
                    if self.position_manager.check_stop_loss(current_price):
                        # Stop loss hit - close position
                        capital = self._close_position(
                            symbol, current_date, current_price, "Stop loss hit", capital
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
                            symbol, current_date, current_price, "Take profit hit", capital
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
                    # Direction-aware stop adjustment
                    current_stop = self.position_manager.position.stop_loss
                    direction = self.position_manager.position.direction

                    if direction == TradeDirection.LONG:
                        # For LONG: stop can only move up
                        if current_stop is None or new_stop > current_stop:
                            self.position_manager.adjust_stop_loss(new_stop)
                    else:
                        # For SHORT: stop can only move down
                        if current_stop is None or new_stop < current_stop:
                            self.position_manager.adjust_stop_loss(new_stop)

                # Check for partial exit
                partial_fraction = strategy.should_partial_exit(context)
                if partial_fraction is not None and partial_fraction > 0:
                    capital = self._partial_exit(
                        symbol, current_date, current_price, partial_fraction,
                        "Partial profit taking", capital
                    )
                    position_value = self.position_manager.get_position_value(current_price)
                    position_value = self._convert_to_base_currency(position_value, symbol, current_date)
                    total_equity = capital + position_value

            # Generate signal from strategy
            signal = strategy.generate_signal(context)

            # Process signal
            if signal.type == SignalType.BUY and not self.position_manager.has_position:
                # Calculate entry equity before opening position
                entry_equity = capital  # Capital represents total equity before position is opened
                # Open new position
                capital = self._open_position(
                    symbol, current_date, current_price, signal, strategy, context, capital, entry_equity
                )
                position_value = self.position_manager.get_position_value(current_price)
                position_value = self._convert_to_base_currency(position_value, symbol, current_date)
                total_equity = capital + position_value

            elif signal.type == SignalType.SELL and self.position_manager.has_position:
                # Close position
                capital = self._close_position(
                    symbol, current_date, current_price, signal.reason or "Strategy exit signal", capital
                )
                position_value = 0
                total_equity = capital

            elif signal.type == SignalType.PARTIAL_EXIT and self.position_manager.has_position:
                # Partial exit
                capital = self._partial_exit(
                    symbol, current_date, current_price, signal.size, signal.reason, capital
                )
                position_value = self.position_manager.get_position_value(current_price)
                position_value = self._convert_to_base_currency(position_value, symbol, current_date)
                total_equity = capital + position_value

            elif signal.type == SignalType.ADJUST_STOP and self.position_manager.has_position:
                # Adjust stop loss (direction-aware)
                if signal.new_stop_loss is not None:
                    current_stop = self.position_manager.position.stop_loss
                    direction = self.position_manager.position.direction

                    if direction == TradeDirection.LONG:
                        if current_stop is None or signal.new_stop_loss > current_stop:
                            self.position_manager.adjust_stop_loss(signal.new_stop_loss)
                    else:
                        if current_stop is None or signal.new_stop_loss < current_stop:
                            self.position_manager.adjust_stop_loss(signal.new_stop_loss)

            elif signal.type == SignalType.PYRAMID and self.position_manager.has_position:
                # Pyramid - add to position with break-even stop
                capital = self._pyramid_position(
                    symbol, current_date, current_price, signal, strategy, context, capital
                )
                position_value = self.position_manager.get_position_value(current_price)
                position_value = self._convert_to_base_currency(position_value, symbol, current_date)
                total_equity = capital + position_value

            # Record equity
            position_value = self.position_manager.get_position_value(current_price)
            position_value = self._convert_to_base_currency(position_value, symbol, current_date)
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
                symbol, final_bar['date'], final_bar['close'],
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

    def _get_fx_rate(self, symbol: str, date: datetime) -> float:
        """
        Get FX rate to convert from security currency to base currency.

        Args:
            symbol: Security symbol
            date: Date for the FX rate

        Returns:
            FX rate (defaults to 1.0 if no conversion needed/available)
        """
        if self.currency_converter is None or self.security_registry is None:
            return 1.0

        # Get security currency
        metadata = self.security_registry.get_metadata(symbol)
        if metadata is None:
            return 1.0

        security_currency = metadata.currency
        base_currency = self.config.base_currency

        # No conversion needed if same currency
        if security_currency == base_currency:
            return 1.0

        # Get conversion rate
        rate = self.currency_converter.get_rate(
            from_currency=security_currency,
            to_currency=base_currency,
            date=date
        )

        # Default to 1.0 if rate not available (and warn)
        if rate is None:
            currency_pair = f"{security_currency}/{base_currency}"
            if currency_pair not in self._fx_rate_warnings:
                self._fx_rate_warnings.add(currency_pair)
                print(f"\n⚠️  WARNING: No FX rate available for {currency_pair} on {date.date()}")
                print(f"   FX rates will default to 1.0 (no conversion)")
                print(f"   Please ensure your currency_rates/{security_currency}{base_currency}.csv")
                print(f"   file covers the full date range of your backtest data.\n")
            return 1.0

        return rate

    def _convert_to_base_currency(self, amount: float, symbol: str, date: datetime) -> float:
        """
        Convert amount from security currency to base currency.

        Args:
            amount: Amount in security currency
            symbol: Security symbol
            date: Date for conversion

        Returns:
            Amount in base currency
        """
        fx_rate = self._get_fx_rate(symbol, date)
        return amount * fx_rate

    def _open_position(self, symbol: str, date: datetime, price: float,
                      signal: Signal, strategy: BaseStrategy,
                      context: StrategyContext, capital: float,
                      entry_equity: float = 0.0) -> float:
        """
        Open a new position.

        Args:
            symbol: Security symbol
            date: Entry date
            price: Entry price (expected price before slippage)
            signal: BUY signal
            strategy: Strategy instance
            context: Current context
            capital: Available capital
            entry_equity: Total portfolio equity at time of entry

        Returns:
            Remaining capital after entry
        """
        # Apply slippage to BUY orders (pay more due to slippage)
        execution_price = price * (1 + self.config.slippage_percent / 100)

        # Get FX rate for currency conversion
        fx_rate = self._get_fx_rate(symbol, date)

        # Calculate position size based on original price
        # Strategy's position_size() now handles currency conversion internally
        # using context.fx_rate, so no additional adjustment needed here
        quantity = strategy.position_size(context, signal)

        if quantity <= 0:
            return capital

        # Adjust quantity to account for slippage (reduce shares to maintain same capital allocation)
        # Since execution_price is higher than price, we need fewer shares to spend the same amount
        quantity = quantity * (price / execution_price)

        # Check position size limit (in base currency)
        max_capital_base = capital * self.config.position_size_limit
        order_value_base = quantity * execution_price * fx_rate  # Convert to base currency

        if order_value_base > max_capital_base:
            # Adjust quantity to fit within limit
            max_capital_security = max_capital_base / fx_rate
            quantity = max_capital_security / execution_price

        # Create entry order with slippage-adjusted price
        entry_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=execution_price,
            timestamp=date,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reason=signal.reason
        )

        # Execute order
        entry_commission = self.trade_executor.execute_order(entry_order)
        total_cost = entry_order.total_value() + entry_commission

        # Convert cost to base currency (GBP)
        total_cost_base = self._convert_to_base_currency(total_cost, symbol, date)

        # Check if we have enough capital (with small epsilon for floating point precision)
        epsilon = 0.01  # Allow $0.01 tolerance for floating point errors
        if total_cost_base > capital + epsilon:
            # Insufficient capital - skip this trade
            return capital

        # Get entry FX rate and currency
        entry_fx_rate = self._get_fx_rate(symbol, date)
        security_currency = "GBP"
        if self.security_registry:
            metadata = self.security_registry.get_metadata(symbol)
            if metadata:
                security_currency = metadata.currency

        # Open position with actual execution price (including slippage)
        self.position_manager.open_position(
            symbol=symbol,
            entry_date=date,
            entry_price=execution_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_reason=signal.reason,
            commission_paid=entry_commission,
            entry_fx_rate=entry_fx_rate,
            security_currency=security_currency,
            entry_equity=entry_equity,
            direction=signal.direction
        )

        # Deduct from capital (in base currency)
        capital -= total_cost_base

        return capital

    def _close_position(self, symbol: str, date: datetime, price: float,
                       reason: str, capital: float) -> float:
        """
        Close current position.

        Args:
            symbol: Security symbol
            date: Exit date
            price: Exit price (expected price before slippage)
            reason: Exit reason
            capital: Current capital

        Returns:
            Updated capital after exit
        """
        # Apply slippage to SELL orders (receive less due to slippage)
        execution_price = price * (1 - self.config.slippage_percent / 100)

        position = self.position_manager.get_position()
        quantity = position.current_quantity

        # Create exit order with slippage-adjusted price
        exit_order = Order(
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=execution_price,
            timestamp=date,
            reason=reason
        )

        # Execute order
        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission

        # Convert proceeds to base currency (GBP)
        proceeds_base = self._convert_to_base_currency(proceeds, symbol, date)

        # Add proceeds to capital (in base currency)
        capital += proceeds_base

        # Get exit FX rate
        exit_fx_rate = self._get_fx_rate(symbol, date)

        # Calculate slippage cost in security currency
        # Entry slippage: paid extra when buying
        # For BUY: execution_price = original_price * (1 + slippage%)
        # Entry slippage per share = execution_price * slippage% / (1 + slippage%)
        slippage_pct = self.config.slippage_percent / 100
        entry_slippage_per_share = position.entry_price * slippage_pct / (1 + slippage_pct)
        entry_slippage = entry_slippage_per_share * position.initial_quantity

        # Exit slippage: received less when selling
        # For SELL: execution_price = original_price * (1 - slippage%)
        # Exit slippage per share = execution_price * slippage% / (1 - slippage%)
        exit_slippage_per_share = execution_price * slippage_pct / (1 - slippage_pct)
        exit_slippage = exit_slippage_per_share * quantity

        # Total slippage in security currency
        total_slippage_sec = entry_slippage + exit_slippage

        # Convert to base currency using average FX rate
        avg_fx_rate = (position.entry_fx_rate + exit_fx_rate) / 2
        total_slippage_base = total_slippage_sec * avg_fx_rate

        # Create trade record with FX information (using execution price with slippage)
        self.trade_executor.create_trade(
            position=position,
            exit_date=date,
            exit_price=execution_price,
            exit_reason=reason,
            exit_commission=exit_commission,
            entry_fx_rate=position.entry_fx_rate,
            exit_fx_rate=exit_fx_rate,
            security_currency=position.security_currency,
            slippage_cost=total_slippage_base
        )

        # Close position
        self.position_manager.close_position()

        return capital

    def _partial_exit(self, symbol: str, date: datetime, price: float, fraction: float,
                     reason: str, capital: float) -> float:
        """
        Partially exit position.

        Args:
            symbol: Security symbol
            date: Exit date
            price: Exit price (expected price before slippage)
            fraction: Fraction to exit (0.0-1.0)
            reason: Exit reason
            capital: Current capital

        Returns:
            Updated capital
        """
        # Apply slippage to SELL orders (receive less due to slippage)
        execution_price = price * (1 - self.config.slippage_percent / 100)

        position = self.position_manager.get_position()
        exit_quantity = position.current_quantity * fraction

        # Create exit order with slippage-adjusted price
        exit_order = Order(
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=exit_quantity,
            order_type=OrderType.MARKET,
            price=execution_price,
            timestamp=date,
            reason=reason
        )

        # Execute order
        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission

        # Convert proceeds to base currency (GBP)
        proceeds_base = self._convert_to_base_currency(proceeds, symbol, date)

        # Add proceeds to capital (in base currency)
        capital += proceeds_base

        # Record partial exit (using execution price with slippage)
        self.position_manager.add_partial_exit(
            exit_date=date,
            quantity=exit_quantity,
            price=execution_price,
            reason=reason,
            commission_paid=exit_commission
        )

        return capital

    def _pyramid_position(self, symbol: str, date: datetime, price: float,
                          signal: Signal, strategy: BaseStrategy,
                          context: StrategyContext, capital: float) -> float:
        """
        Add to existing position (pyramid).

        After pyramiding:
        - Position quantity is increased
        - Average entry price is updated (weighted average)
        - Stop loss is moved to break-even (accounting for commissions)

        Args:
            symbol: Security symbol
            date: Pyramid date
            price: Current price (expected price before slippage)
            signal: PYRAMID signal
            strategy: Strategy instance
            context: Current context
            capital: Available capital

        Returns:
            Remaining capital after pyramid
        """
        # Apply slippage to pyramid (same as BUY)
        execution_price = price * (1 + self.config.slippage_percent / 100)

        # Get FX rate
        fx_rate = self._get_fx_rate(symbol, date)

        # Calculate pyramid size - use signal.size as fraction of available capital
        capital_to_use = capital * signal.size
        quantity = capital_to_use / (execution_price * fx_rate)

        if quantity <= 0:
            return capital

        # Adjust quantity for slippage
        quantity = quantity * (price / execution_price)

        # Create pyramid order
        pyramid_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=execution_price,
            timestamp=date,
            reason=signal.reason
        )

        # Execute order
        pyramid_commission = self.trade_executor.execute_order(pyramid_order)
        total_cost = pyramid_order.total_value() + pyramid_commission

        # Convert cost to base currency
        total_cost_base = self._convert_to_base_currency(total_cost, symbol, date)

        # Check if we have enough capital
        epsilon = 0.01
        if total_cost_base > capital + epsilon:
            return capital

        # Add pyramid to position (updates average price and sets break-even stop)
        self.position_manager.add_pyramid(
            pyramid_date=date,
            quantity=quantity,
            price=execution_price,
            commission=pyramid_commission,
            reason=signal.reason
        )

        # Deduct from capital
        capital -= total_cost_base

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
