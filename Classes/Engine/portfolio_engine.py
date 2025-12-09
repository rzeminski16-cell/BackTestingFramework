"""
Portfolio-level backtesting engine with capital contention management.
"""
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from ..Strategy.base_strategy import BaseStrategy
from ..Strategy.strategy_context import StrategyContext
from ..Models.signal import Signal, SignalType
from ..Models.order import Order, OrderSide, OrderType
from ..Models.trade import Trade
from ..Models.position import Position
from ..Config.config import PortfolioConfig
from ..Config.capital_contention import CapitalContentionMode
from .position_manager import PositionManager
from .trade_executor import TradeExecutor
from .backtest_result import BacktestResult
from .vulnerability_score import VulnerabilityScoreCalculator, VulnerabilityResult, SwapDecision
from ..Data.currency_converter import CurrencyConverter
from ..Data.security_registry import SecurityRegistry


@dataclass
class SignalRejection:
    """Record of a signal that was rejected due to capital constraints."""
    date: datetime
    symbol: str
    signal_type: str
    reason: str
    available_capital: float
    required_capital: float
    vulnerability_decision: Optional[SwapDecision] = None


@dataclass
class VulnerabilitySwap:
    """Record of a position swap due to vulnerability score."""
    date: datetime
    closed_symbol: str
    closed_score: float
    new_symbol: str
    all_scores: Dict[str, VulnerabilityResult] = field(default_factory=dict)


@dataclass
class PortfolioBacktestResult:
    """
    Extended backtest result with portfolio-specific information.
    """
    # Per-symbol results
    symbol_results: Dict[str, BacktestResult]

    # Portfolio-level metrics
    final_equity: float
    total_return: float
    total_return_pct: float
    portfolio_equity_curve: pd.DataFrame

    # Capital contention tracking
    signal_rejections: List[SignalRejection]
    vulnerability_swaps: List[VulnerabilitySwap]
    vulnerability_history: List[Dict[str, VulnerabilityResult]]  # Per-day scores

    # Configuration used
    config: PortfolioConfig
    strategy_name: str


class PortfolioEngine:
    """
    Portfolio-level backtesting engine.

    Manages multiple securities with:
    - Shared capital pool
    - Capital contention resolution (default or vulnerability score)
    - Position value cannot exceed total equity
    - Comprehensive logging of signal rejections and swaps
    """

    def __init__(self, config: PortfolioConfig,
                 currency_converter: Optional[CurrencyConverter] = None,
                 security_registry: Optional[SecurityRegistry] = None):
        """
        Initialize portfolio engine.

        Args:
            config: Portfolio configuration
            currency_converter: Optional currency converter for multi-currency support
            security_registry: Optional security registry for metadata
        """
        self.config = config
        self.position_managers: Dict[str, PositionManager] = {}
        self.trade_executor = TradeExecutor(config.commission)
        self.currency_converter = currency_converter
        self.security_registry = security_registry
        self._fx_rate_warnings = set()

        # Capital contention components
        self.vulnerability_calculator: Optional[VulnerabilityScoreCalculator] = None
        if config.capital_contention.mode == CapitalContentionMode.VULNERABILITY_SCORE:
            self.vulnerability_calculator = VulnerabilityScoreCalculator(
                config.capital_contention.vulnerability_config
            )

        # Tracking
        self.signal_rejections: List[SignalRejection] = []
        self.vulnerability_swaps: List[VulnerabilitySwap] = []
        self.vulnerability_history: List[Dict[str, VulnerabilityResult]] = []

    def run(self, data_dict: Dict[str, pd.DataFrame],
            strategy: BaseStrategy,
            progress_callback: Optional[Callable[[int, int], None]] = None) -> PortfolioBacktestResult:
        """
        Run portfolio backtest across multiple securities.

        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            strategy: Trading strategy (same strategy for all securities)
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            PortfolioBacktestResult with complete results and tracking data
        """
        # Reset tracking
        self.signal_rejections = []
        self.vulnerability_swaps = []
        self.vulnerability_history = []
        self.trade_executor = TradeExecutor(self.config.commission)

        # Initialize position managers for each symbol
        self.position_managers = {}
        for symbol in data_dict.keys():
            self.position_managers[symbol] = PositionManager()

        # CRITICAL: Pre-calculate indicators for each security
        prepared_data_dict = {}
        for symbol, data in data_dict.items():
            prepared_data_dict[symbol] = strategy.prepare_data(data)
        data_dict = prepared_data_dict

        # Get unified date range across all securities
        all_dates = self._get_unified_dates(data_dict)

        # Initialize tracking
        capital = self.config.initial_capital
        equity_history = []

        # Process each date
        total_dates = len(all_dates)
        for date_idx, current_date in enumerate(all_dates):
            # Update progress callback
            if progress_callback and (date_idx % 50 == 0 or date_idx == total_dates - 1):
                progress_callback(date_idx + 1, total_dates)

            # Get current prices for all securities
            current_prices = {}
            for symbol, data in data_dict.items():
                date_data = data[data['date'] == current_date]
                if len(date_data) > 0:
                    current_prices[symbol] = date_data.iloc[0]['close']

            # Calculate total position value (converted to base currency)
            total_position_value = 0
            for symbol, pm in self.position_managers.items():
                if pm.has_position and symbol in current_prices:
                    pos_value = pm.get_position_value(current_prices[symbol])
                    pos_value_base = self._convert_to_base_currency(pos_value, symbol, current_date)
                    total_position_value += pos_value_base

            total_equity = capital + total_position_value

            # Track vulnerability scores for the day (if using vulnerability mode)
            if self.vulnerability_calculator:
                open_positions = {
                    sym: pm.get_position()
                    for sym, pm in self.position_managers.items()
                    if pm.has_position
                }
                if open_positions:
                    day_scores = self.vulnerability_calculator.calculate_all_scores(
                        open_positions, current_prices, current_date
                    )
                    self.vulnerability_history.append(day_scores)

            # Collect new signals for this date
            new_signals: List[tuple] = []  # (symbol, signal, context)

            # Process each security
            for symbol in data_dict.keys():
                if symbol not in current_prices:
                    continue

                current_price = current_prices[symbol]
                data = data_dict[symbol]
                pm = self.position_managers[symbol]

                # Get bar index for this date
                date_mask = data['date'] == current_date
                if not date_mask.any():
                    continue

                # DATA LEAKAGE FIX: Get positional index
                label_index = data[date_mask].index[0]
                if isinstance(data.index, pd.RangeIndex):
                    bar_position = label_index
                else:
                    bar_position = data.index.get_loc(label_index)

                # Only pass historical data up to current bar (no future data access)
                historical_data = data.iloc[:bar_position+1].copy()

                # Get FX rate for currency conversion
                fx_rate = self._get_fx_rate(symbol, current_date)

                # Create context
                context = StrategyContext(
                    data=historical_data,
                    current_index=bar_position,
                    current_price=current_price,
                    current_date=current_date,
                    position=pm.get_position(),
                    available_capital=capital,
                    total_equity=total_equity,
                    symbol=symbol,
                    fx_rate=fx_rate
                )

                # PRIORITY 1: Check stop loss (natural exits always take priority)
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

                # Process SELL/PARTIAL_EXIT/ADJUST_STOP immediately
                if signal.type == SignalType.SELL and pm.has_position:
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
                elif signal.type == SignalType.BUY and not pm.has_position:
                    # Collect BUY signals for capital contention processing
                    new_signals.append((symbol, signal, context, strategy))

            # Process BUY signals with capital contention
            capital = self._process_buy_signals(
                new_signals, capital, total_equity, current_prices, current_date
            )

            # Recalculate equity at end of day
            total_position_value = 0
            for symbol, pm in self.position_managers.items():
                if pm.has_position and symbol in current_prices:
                    pos_value = pm.get_position_value(current_prices[symbol])
                    pos_value_base = self._convert_to_base_currency(pos_value, symbol, current_date)
                    total_position_value += pos_value_base

            total_equity = capital + total_position_value

            equity_history.append({
                'date': current_date,
                'equity': total_equity,
                'capital': capital,
                'position_value': total_position_value,
                'num_positions': sum(1 for pm in self.position_managers.values() if pm.has_position)
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
        symbol_results = {}
        for symbol in data_dict.keys():
            symbol_trades = trades_by_symbol[symbol]
            symbol_pl = sum(t.pl for t in symbol_trades)

            result = BacktestResult(
                symbol=symbol,
                strategy_name=strategy.get_name(),
                trades=symbol_trades,
                equity_curve=equity_df,
                final_equity=final_equity,
                total_return=symbol_pl,
                total_return_pct=(symbol_pl / self.config.initial_capital * 100) if symbol_trades else 0.0,
                strategy_params=strategy.get_parameters()
            )
            symbol_results[symbol] = result

        return PortfolioBacktestResult(
            symbol_results=symbol_results,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            portfolio_equity_curve=equity_df,
            signal_rejections=self.signal_rejections,
            vulnerability_swaps=self.vulnerability_swaps,
            vulnerability_history=self.vulnerability_history,
            config=self.config,
            strategy_name=strategy.get_name()
        )

    def _process_buy_signals(self, signals: List[tuple], capital: float,
                             total_equity: float, current_prices: Dict[str, float],
                             current_date: datetime) -> float:
        """
        Process BUY signals with capital contention logic.

        Args:
            signals: List of (symbol, signal, context, strategy) tuples
            capital: Current available capital
            total_equity: Total portfolio equity
            current_prices: Current prices for all securities
            current_date: Current date

        Returns:
            Updated capital after processing signals
        """
        for symbol, signal, context, strategy in signals:
            pm = self.position_managers[symbol]
            current_price = current_prices[symbol]

            # Calculate required capital for this position
            quantity = strategy.position_size(context, signal)
            if quantity <= 0:
                continue

            fx_rate = self._get_fx_rate(symbol, current_date)
            required_capital = quantity * current_price * fx_rate

            # Check if we have enough capital
            if required_capital <= capital:
                # CONSTRAINT: Position value cannot exceed total equity
                if required_capital > total_equity:
                    # Reduce position size to fit within equity
                    quantity = (total_equity * 0.99) / (current_price * fx_rate)  # 99% to leave buffer
                    required_capital = quantity * current_price * fx_rate

                if required_capital <= capital and quantity > 0:
                    capital = self._open_position(
                        symbol, current_date, current_price,
                        signal, strategy, context, capital, pm
                    )
                else:
                    self._record_signal_rejection(
                        current_date, symbol, "BUY",
                        f"Position size exceeds total equity (required: {required_capital:.2f}, equity: {total_equity:.2f})",
                        capital, required_capital
                    )
            else:
                # Not enough capital - apply capital contention logic
                capital = self._handle_capital_contention(
                    symbol, signal, context, strategy, capital, total_equity,
                    current_prices, current_date, required_capital
                )

        return capital

    def _handle_capital_contention(self, new_symbol: str, signal: Signal,
                                   context: StrategyContext, strategy: BaseStrategy,
                                   capital: float, total_equity: float,
                                   current_prices: Dict[str, float],
                                   current_date: datetime,
                                   required_capital: float) -> float:
        """
        Handle capital contention when a BUY signal arrives with insufficient capital.

        Args:
            new_symbol: Symbol of the new signal
            signal: The BUY signal
            context: Strategy context
            strategy: Strategy instance
            capital: Current available capital
            total_equity: Total portfolio equity
            current_prices: Current prices for all securities
            current_date: Current date
            required_capital: Capital required for the new position

        Returns:
            Updated capital after processing
        """
        mode = self.config.capital_contention.mode

        if mode == CapitalContentionMode.DEFAULT:
            # DEFAULT MODE: Simply reject the signal
            self._record_signal_rejection(
                current_date, new_symbol, "BUY",
                f"Insufficient capital (available: {capital:.2f}, required: {required_capital:.2f})",
                capital, required_capital
            )
            return capital

        elif mode == CapitalContentionMode.VULNERABILITY_SCORE:
            # VULNERABILITY SCORE MODE: Check if we can swap a weak position
            open_positions = {
                sym: pm.get_position()
                for sym, pm in self.position_managers.items()
                if pm.has_position
            }

            if not open_positions:
                # No positions to swap
                self._record_signal_rejection(
                    current_date, new_symbol, "BUY",
                    "Insufficient capital and no positions to swap",
                    capital, required_capital
                )
                return capital

            # Get swap decision
            swap_decision = self.vulnerability_calculator.should_swap(
                open_positions, current_prices, current_date, new_symbol
            )

            if swap_decision.should_swap:
                # Close the weak position
                weak_symbol = swap_decision.position_to_close
                weak_pm = self.position_managers[weak_symbol]
                weak_price = current_prices[weak_symbol]

                # Record the swap
                self.vulnerability_swaps.append(VulnerabilitySwap(
                    date=current_date,
                    closed_symbol=weak_symbol,
                    closed_score=swap_decision.position_score,
                    new_symbol=new_symbol,
                    all_scores=swap_decision.all_scores
                ))

                # Close the weak position with vulnerability score exit reason
                capital = self._close_position(
                    weak_symbol, current_date, weak_price,
                    f"Vulnerability score swap (score: {swap_decision.position_score:.1f}) for {new_symbol}",
                    capital, weak_pm
                )

                # Recalculate total equity after closing
                total_position_value = 0
                for sym, pm in self.position_managers.items():
                    if pm.has_position and sym in current_prices:
                        pos_value = pm.get_position_value(current_prices[sym])
                        pos_value_base = self._convert_to_base_currency(pos_value, sym, current_date)
                        total_position_value += pos_value_base
                total_equity = capital + total_position_value

                # Now try to open the new position
                new_pm = self.position_managers[new_symbol]

                # Recalculate position size with updated capital
                new_context = StrategyContext(
                    data=context.data,
                    current_index=context.current_index,
                    current_price=context.current_price,
                    current_date=current_date,
                    position=None,
                    available_capital=capital,
                    total_equity=total_equity,
                    symbol=new_symbol,
                    fx_rate=context.fx_rate
                )

                capital = self._open_position(
                    new_symbol, current_date, current_prices[new_symbol],
                    signal, strategy, new_context, capital, new_pm
                )
            else:
                # Cannot swap - record rejection with vulnerability info
                self._record_signal_rejection(
                    current_date, new_symbol, "BUY",
                    swap_decision.reason,
                    capital, required_capital,
                    swap_decision
                )

        return capital

    def _record_signal_rejection(self, date: datetime, symbol: str, signal_type: str,
                                 reason: str, available_capital: float,
                                 required_capital: float,
                                 vulnerability_decision: Optional[SwapDecision] = None):
        """Record a signal rejection."""
        self.signal_rejections.append(SignalRejection(
            date=date,
            symbol=symbol,
            signal_type=signal_type,
            reason=reason,
            available_capital=available_capital,
            required_capital=required_capital,
            vulnerability_decision=vulnerability_decision
        ))

    def _get_fx_rate(self, symbol: str, date: datetime) -> float:
        """Get FX rate to convert from security currency to base currency."""
        if self.currency_converter is None or self.security_registry is None:
            return 1.0

        metadata = self.security_registry.get_metadata(symbol)
        if metadata is None:
            return 1.0

        security_currency = metadata.currency
        base_currency = self.config.base_currency

        if security_currency == base_currency:
            return 1.0

        rate = self.currency_converter.get_rate(
            from_currency=security_currency,
            to_currency=base_currency,
            date=date
        )

        if rate is None:
            currency_pair = f"{security_currency}/{base_currency}"
            if currency_pair not in self._fx_rate_warnings:
                self._fx_rate_warnings.add(currency_pair)
                print(f"\nWARNING: No FX rate available for {currency_pair} on {date.date()}")
                print(f"   FX rates will default to 1.0 (no conversion)")
            return 1.0

        return rate

    def _convert_to_base_currency(self, amount: float, symbol: str, date: datetime) -> float:
        """Convert amount from security currency to base currency."""
        fx_rate = self._get_fx_rate(symbol, date)
        return amount * fx_rate

    def _open_position(self, symbol: str, date: datetime, price: float,
                       signal: Signal, strategy: BaseStrategy,
                       context: StrategyContext, capital: float,
                       pm: PositionManager) -> float:
        """Open position with capital constraints."""
        fx_rate = self._get_fx_rate(symbol, date)
        quantity = strategy.position_size(context, signal)

        if quantity <= 0:
            return capital

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
        total_cost_base = self._convert_to_base_currency(total_cost, symbol, date)

        if total_cost_base > capital:
            return capital  # Insufficient capital

        # Get entry FX rate and currency
        entry_fx_rate = self._get_fx_rate(symbol, date)
        security_currency = "GBP"
        if self.security_registry:
            metadata = self.security_registry.get_metadata(symbol)
            if metadata:
                security_currency = metadata.currency

        # Open position
        pm.open_position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_reason=signal.reason,
            commission_paid=entry_commission,
            entry_fx_rate=entry_fx_rate,
            security_currency=security_currency
        )

        capital -= total_cost_base
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
        proceeds_base = self._convert_to_base_currency(proceeds, symbol, date)

        capital += proceeds_base

        exit_fx_rate = self._get_fx_rate(symbol, date)

        self.trade_executor.create_trade(
            position=position,
            exit_date=date,
            exit_price=price,
            exit_reason=reason,
            exit_commission=exit_commission,
            entry_fx_rate=position.entry_fx_rate,
            exit_fx_rate=exit_fx_rate,
            security_currency=position.security_currency
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
        proceeds_base = self._convert_to_base_currency(proceeds, symbol, date)

        capital += proceeds_base

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
            filtered = data.copy()
            if self.config.start_date:
                filtered = filtered[filtered['date'] >= self.config.start_date]
            if self.config.end_date:
                filtered = filtered[filtered['date'] <= self.config.end_date]

            all_dates.update(filtered['date'].tolist())

        return sorted(all_dates)
