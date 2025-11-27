"""
Portfolio-level backtesting engine.
"""
import pandas as pd
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

from ..Strategy.base_strategy import BaseStrategy
from ..Strategy.strategy_context import StrategyContext
from ..Models.signal import Signal, SignalType
from ..Models.order import Order, OrderSide, OrderType
from ..Models.trade import Trade
from ..Config.config import PortfolioConfig
from .position_manager import PositionManager
from .trade_executor import TradeExecutor
from .backtest_result import BacktestResult


class PortfolioEngine:
    """
    Portfolio-level backtesting engine.

    Manages multiple securities with:
    - Shared capital pool
    - Maximum position limits
    - Position sizing across portfolio
    - All features of single-security engine
    """

    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio engine.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.position_managers: Dict[str, PositionManager] = {}
        self.trade_executor = TradeExecutor(config.commission)

    def run(self, data_dict: Dict[str, pd.DataFrame],
            strategy: BaseStrategy) -> Dict[str, BacktestResult]:
        """
        Run portfolio backtest across multiple securities.

        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            strategy: Trading strategy (same strategy for all securities)

        Returns:
            Dictionary mapping symbol to BacktestResult
        """
        # Initialize position managers for each symbol
        for symbol in data_dict.keys():
            self.position_managers[symbol] = PositionManager()

        # Get unified date range across all securities
        all_dates = self._get_unified_dates(data_dict)

        # Initialize tracking
        capital = self.config.initial_capital
        equity_history = []

        # Process each date
        for current_date in all_dates:
            # Track daily actions
            day_trades = []

            # Get current prices for all securities
            current_prices = {}
            for symbol, data in data_dict.items():
                date_data = data[data['date'] == current_date]
                if len(date_data) > 0:
                    current_prices[symbol] = date_data.iloc[0]['close']

            # Calculate total position value
            total_position_value = 0
            for symbol, pm in self.position_managers.items():
                if pm.has_position and symbol in current_prices:
                    total_position_value += pm.get_position_value(current_prices[symbol])

            total_equity = capital + total_position_value

            # Process each security
            for symbol in data_dict.keys():
                if symbol not in current_prices:
                    continue  # No data for this symbol on this date

                current_price = current_prices[symbol]
                data = data_dict[symbol]
                pm = self.position_managers[symbol]

                # Get bar index for this date
                date_mask = data['date'] == current_date
                if not date_mask.any():
                    continue

                bar_index = data[date_mask].index[0]
                historical_data = data.iloc[:bar_index+1].copy()

                # Create context
                context = StrategyContext(
                    data=historical_data,
                    current_index=bar_index,
                    current_price=current_price,
                    current_date=current_date,
                    position=pm.get_position(),
                    available_capital=capital,
                    total_equity=total_equity
                )

                # Check stop loss
                if pm.has_position:
                    if strategy.should_check_stop_loss(context):
                        if pm.check_stop_loss(current_price):
                            capital = self._close_position(
                                symbol, current_date, current_price,
                                "Stop loss hit", capital, pm
                            )
                            continue

                    # Check take profit
                    if strategy.should_check_take_profit(context):
                        if pm.check_take_profit(current_price):
                            capital = self._close_position(
                                symbol, current_date, current_price,
                                "Take profit hit", capital, pm
                            )
                            continue

                    # Check trailing stop
                    new_stop = strategy.should_adjust_stop(context)
                    if new_stop is not None:
                        current_stop = pm.position.stop_loss
                        if current_stop is None or new_stop > current_stop:
                            pm.adjust_stop_loss(new_stop)

                    # Check partial exit
                    partial_fraction = strategy.should_partial_exit(context)
                    if partial_fraction is not None and partial_fraction > 0:
                        capital = self._partial_exit(
                            symbol, current_date, current_price,
                            partial_fraction, "Partial profit taking",
                            capital, pm
                        )

                # Generate signal
                signal = strategy.generate_signal(context)

                # Process signal
                if signal.type == SignalType.BUY and not pm.has_position:
                    # Check portfolio constraints
                    num_positions = sum(1 for pm in self.position_managers.values() if pm.has_position)

                    if self.config.max_positions is None or num_positions < self.config.max_positions:
                        capital = self._open_position(
                            symbol, current_date, current_price,
                            signal, strategy, context, capital, pm
                        )

                elif signal.type == SignalType.SELL and pm.has_position:
                    capital = self._close_position(
                        symbol, current_date, current_price,
                        signal.reason or "Strategy exit signal",
                        capital, pm
                    )

                elif signal.type == SignalType.PARTIAL_EXIT and pm.has_position:
                    capital = self._partial_exit(
                        symbol, current_date, current_price,
                        signal.size, signal.reason, capital, pm
                    )

                elif signal.type == SignalType.ADJUST_STOP and pm.has_position:
                    if signal.new_stop_loss is not None:
                        current_stop = pm.position.stop_loss
                        if current_stop is None or signal.new_stop_loss > current_stop:
                            pm.adjust_stop_loss(signal.new_stop_loss)

            # Recalculate equity at end of day
            total_position_value = 0
            for symbol, pm in self.position_managers.items():
                if pm.has_position and symbol in current_prices:
                    total_position_value += pm.get_position_value(current_prices[symbol])

            total_equity = capital + total_position_value

            equity_history.append({
                'date': current_date,
                'equity': total_equity,
                'capital': capital,
                'position_value': total_position_value
            })

        # Close remaining positions
        for symbol, pm in self.position_managers.items():
            if pm.has_position:
                final_data = data_dict[symbol]
                final_bar = final_data.iloc[-1]
                capital = self._close_position(
                    symbol, final_bar['date'], final_bar['close'],
                    "End of backtest period", capital, pm
                )

        # Create equity curve
        equity_df = pd.DataFrame(equity_history)

        # Calculate returns
        final_equity = capital
        total_return = final_equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100

        # Group trades by symbol
        trades_by_symbol = defaultdict(list)
        for trade in self.trade_executor.get_trades():
            trades_by_symbol[trade.symbol].append(trade)

        # Create results per symbol
        results = {}
        for symbol in data_dict.keys():
            symbol_trades = trades_by_symbol[symbol]

            # Calculate symbol-specific returns (from its trades)
            symbol_pl = sum(t.pl for t in symbol_trades)

            result = BacktestResult(
                symbol=symbol,
                strategy_name=strategy.get_name(),
                trades=symbol_trades,
                equity_curve=equity_df,  # Portfolio equity (same for all)
                final_equity=final_equity,  # Portfolio final equity
                total_return=symbol_pl,  # Symbol-specific P/L
                total_return_pct=(symbol_pl / self.config.initial_capital * 100) if symbol_trades else 0.0
            )
            results[symbol] = result

        return results

    def _open_position(self, symbol: str, date: datetime, price: float,
                      signal: Signal, strategy: BaseStrategy,
                      context: StrategyContext, capital: float,
                      pm: PositionManager) -> float:
        """Open position with portfolio constraints."""
        # Calculate base quantity
        quantity = strategy.position_size(context, signal)

        if quantity <= 0:
            return capital

        # Apply portfolio position size limit
        max_capital_per_position = capital * self.config.position_size_limit
        order_value = quantity * price

        if order_value > max_capital_per_position:
            quantity = max_capital_per_position / price

        # Apply total allocation limit
        total_positions_value = sum(
            pm.get_position_value(price) for pm in self.position_managers.values()
            if pm.has_position
        )
        max_total_value = (capital + total_positions_value) * self.config.total_allocation_limit
        available_for_new = max_total_value - total_positions_value

        if order_value > available_for_new:
            if available_for_new > 0:
                quantity = available_for_new / price
            else:
                return capital  # No room for new position

        # Create and execute order
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

        entry_commission = self.trade_executor.execute_order(entry_order)
        total_cost = entry_order.total_value() + entry_commission

        if total_cost > capital:
            return capital  # Insufficient capital

        # Open position
        pm.open_position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_reason=signal.reason,
            commission_paid=entry_commission
        )

        capital -= total_cost
        return capital

    def _close_position(self, symbol: str, date: datetime, price: float,
                       reason: str, capital: float, pm: PositionManager) -> float:
        """Close position."""
        position = pm.get_position()
        quantity = position.current_quantity

        exit_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            timestamp=date,
            reason=reason
        )

        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission

        capital += proceeds

        self.trade_executor.create_trade(
            position=position,
            exit_date=date,
            exit_price=price,
            exit_reason=reason,
            exit_commission=exit_commission
        )

        pm.close_position()
        return capital

    def _partial_exit(self, symbol: str, date: datetime, price: float,
                     fraction: float, reason: str, capital: float,
                     pm: PositionManager) -> float:
        """Partial exit."""
        position = pm.get_position()
        exit_quantity = position.current_quantity * fraction

        exit_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=exit_quantity,
            order_type=OrderType.MARKET,
            price=price,
            timestamp=date,
            reason=reason
        )

        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission

        capital += proceeds

        pm.add_partial_exit(
            exit_date=date,
            quantity=exit_quantity,
            price=price,
            reason=reason,
            commission_paid=exit_commission
        )

        return capital

    def _get_unified_dates(self, data_dict: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get unified list of all unique dates across securities."""
        all_dates = set()

        for data in data_dict.values():
            # Filter by date range
            filtered = data.copy()
            if self.config.start_date:
                filtered = filtered[filtered['date'] >= self.config.start_date]
            if self.config.end_date:
                filtered = filtered[filtered['date'] <= self.config.end_date]

            all_dates.update(filtered['date'].tolist())

        return sorted(all_dates)
