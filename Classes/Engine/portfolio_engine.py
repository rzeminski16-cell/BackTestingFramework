"""
Portfolio-level backtesting engine with capital contention management.
"""
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from ..Strategy.base_strategy import BaseStrategy
from ..Strategy.strategy_context import StrategyContext
from ..Models.signal import Signal, SignalType
from ..Models.trade_direction import TradeDirection
from ..Models.order import Order, OrderSide, OrderType
from ..Models.trade import Trade, reset_trade_counter
from ..Models.position import Position
from ..Config.config import PortfolioConfig
from ..Config.capital_contention import CapitalContentionMode
from .position_manager import PositionManager
from .trade_executor import TradeExecutor
from .backtest_result import BacktestResult
from .vulnerability_score import VulnerabilityScoreCalculator, VulnerabilityResult, SwapDecision
from ..Data.currency_converter import CurrencyConverter
from ..Data.security_registry import SecurityRegistry
from ..Data.historical_data_view import HistoricalDataView


@dataclass
class CapitalAllocationEvent:
    """Records capital allocation state when a buy signal is processed."""
    date: datetime
    symbol: str
    signal_type: str  # "EXECUTED", "REJECTED", "SWAPPED_IN", "SWAPPED_OUT"
    available_capital: float
    required_capital: float
    total_equity: float
    num_open_positions: int
    open_position_symbols: List[str]
    competing_signals: List[str]  # Other symbols with signals on this date
    outcome: str  # Detailed description of what happened
    vulnerability_scores: Optional[Dict[str, float]] = None  # If vulnerability mode
    trade_id: Optional[str] = None  # Trade ID if executed


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
    capital_allocation_events: List[CapitalAllocationEvent]  # All capital decisions

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
        self.capital_allocation_events: List[CapitalAllocationEvent] = []

        # Cache for vulnerability scores (reset per day to avoid duplicate calculations)
        self._current_day_vuln_scores: Optional[Dict[str, VulnerabilityResult]] = None

        # Track last known prices to handle missing data in equity calculations
        # This prevents false drawdowns when a security doesn't trade on a particular day
        self._last_known_prices: Dict[str, float] = {}

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
        self.capital_allocation_events = []
        self.trade_executor = TradeExecutor(self.config.commission)

        # Reset trade ID counter for this backtest
        reset_trade_counter()

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

        # Reset last known prices for this backtest run
        self._last_known_prices = {}

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
                    # Update last known price
                    self._last_known_prices[symbol] = current_prices[symbol]

            # Calculate total position value (converted to base currency)
            # Use last known price as fallback when current price is not available
            total_position_value = 0
            for symbol, pm in self.position_managers.items():
                if pm.has_position:
                    # Use current price if available, otherwise use last known price
                    price = current_prices.get(symbol) or self._last_known_prices.get(symbol)
                    if price:
                        pos_value = pm.get_position_value(price)
                        pos_value_base = self._convert_to_base_currency(pos_value, symbol, current_date)
                        total_position_value += pos_value_base

            total_equity = capital + total_position_value

            # Track vulnerability scores for the day (if using vulnerability mode)
            # PERFORMANCE: Cache scores to avoid duplicate calculations during signal processing
            self._current_day_vuln_scores = None  # Reset cache for new day
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
                    self._current_day_vuln_scores = day_scores  # Cache for reuse in signal processing
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
                # PERFORMANCE: Use HistoricalDataView instead of .copy() to avoid O(n²) memory operations
                # HistoricalDataView enforces look-ahead protection without expensive copying
                historical_data = HistoricalDataView(data, valid_end_index=bar_position)

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

                    # Check trailing stop (direction-aware)
                    new_stop = strategy.should_adjust_stop(context)
                    if new_stop is not None:
                        current_stop = pm.position.stop_loss
                        direction = pm.position.direction

                        if direction == TradeDirection.LONG:
                            # For LONG: stop can only move up
                            if current_stop is None or new_stop > current_stop:
                                pm.adjust_stop_loss(new_stop)
                        else:
                            # For SHORT: stop can only move down
                            if current_stop is None or new_stop < current_stop:
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
                    # Adjust stop loss (direction-aware)
                    if signal.new_stop_loss is not None:
                        current_stop = pm.position.stop_loss
                        direction = pm.position.direction

                        if direction == TradeDirection.LONG:
                            if current_stop is None or signal.new_stop_loss > current_stop:
                                pm.adjust_stop_loss(signal.new_stop_loss)
                        else:
                            if current_stop is None or signal.new_stop_loss < current_stop:
                                pm.adjust_stop_loss(signal.new_stop_loss)
                elif signal.type == SignalType.PYRAMID and pm.has_position:
                    # Pyramid - add to position with break-even stop
                    capital = self._pyramid_position(
                        symbol, current_date, current_price,
                        signal, strategy, context, capital, pm
                    )
                elif signal.type == SignalType.BUY and not pm.has_position:
                    # Collect BUY signals for capital contention processing
                    new_signals.append((symbol, signal, context, strategy))

            # Process BUY signals with capital contention
            capital = self._process_buy_signals(
                new_signals, capital, total_equity, current_prices, current_date
            )

            # Recalculate equity at end of day
            # Use last known price as fallback when current price is not available
            total_position_value = 0
            for symbol, pm in self.position_managers.items():
                if pm.has_position:
                    # Use current price if available, otherwise use last known price
                    price = current_prices.get(symbol) or self._last_known_prices.get(symbol)
                    if price:
                        pos_value = pm.get_position_value(price)
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
            capital_allocation_events=self.capital_allocation_events,
            config=self.config,
            strategy_name=strategy.get_name()
        )

    def _process_buy_signals(self, signals: List[tuple], capital: float,
                             total_equity: float, current_prices: Dict[str, float],
                             current_date: datetime) -> float:
        """
        Process BUY signals with capital contention logic.

        When multiple signals arrive and there isn't enough capital for all:
        - In DEFAULT mode: Process in order, reject when capital runs out
        - In VULNERABILITY mode: Compare all signals, prioritize by vulnerability scoring

        Args:
            signals: List of (symbol, signal, context, strategy) tuples
            capital: Current available capital
            total_equity: Total portfolio equity
            current_prices: Current prices for all securities
            current_date: Current date

        Returns:
            Updated capital after processing signals
        """
        if not signals:
            return capital

        # Get current open position info for tracking
        open_position_symbols = [
            sym for sym, pm in self.position_managers.items() if pm.has_position
        ]
        num_open_positions = len(open_position_symbols)

        # Calculate capital requirements for ALL signals first
        signal_requirements: List[Tuple[str, Signal, StrategyContext, BaseStrategy, float, float]] = []
        all_signal_symbols = [s[0] for s in signals]

        for symbol, signal, context, strategy in signals:
            current_price = current_prices[symbol]
            quantity = strategy.position_size(context, signal)
            if quantity <= 0:
                # Record rejection instead of silently skipping
                self._record_signal_rejection(
                    current_date, symbol, "BUY",
                    f"Position size calculation returned {quantity:.4f} (stop too tight or insufficient equity)",
                    capital, 0.0
                )
                continue
            fx_rate = self._get_fx_rate(symbol, current_date)
            required_capital = quantity * current_price * fx_rate
            signal_requirements.append((symbol, signal, context, strategy, required_capital, quantity))

        if not signal_requirements:
            return capital

        # Calculate total capital needed for all signals
        total_required = sum(req[4] for req in signal_requirements)

        # If we have enough capital for all signals, process all
        if total_required <= capital:
            for symbol, signal, context, strategy, required_capital, quantity in signal_requirements:
                pm = self.position_managers[symbol]
                current_price = current_prices[symbol]

                # Record capital allocation event
                other_signals = [s for s in all_signal_symbols if s != symbol]
                self._record_capital_event(
                    current_date, symbol, "EXECUTED",
                    capital, required_capital, total_equity,
                    num_open_positions, open_position_symbols, other_signals,
                    f"Sufficient capital for all {len(signal_requirements)} signals"
                )

                capital = self._open_position_with_tracking(
                    symbol, current_date, current_price, signal, strategy, context,
                    capital, pm, required_capital, num_open_positions, other_signals
                )
                num_open_positions += 1
                open_position_symbols.append(symbol)

            return capital

        # CAPITAL CONTENTION: Not enough capital for all signals
        # Check if using vulnerability score mode
        mode = self.config.capital_contention.mode

        if mode == CapitalContentionMode.VULNERABILITY_SCORE and self.vulnerability_calculator:
            capital = self._process_signals_with_vulnerability(
                signal_requirements, capital, total_equity, current_prices,
                current_date, open_position_symbols, all_signal_symbols
            )
        else:
            # DEFAULT MODE: Process signals in order until capital runs out
            capital = self._process_signals_default_mode(
                signal_requirements, capital, total_equity, current_prices,
                current_date, open_position_symbols, all_signal_symbols
            )

        return capital

    def _try_reduced_position(self, capital: float, required_capital: float,
                               quantity: float, current_price: float,
                               fx_rate: float, min_reduction: float = 0.5) -> Tuple[float, float, bool]:
        """
        Try to fit a position with reduced size if full size doesn't fit.

        Args:
            capital: Available capital
            required_capital: Capital required for full position
            quantity: Full position quantity
            current_price: Current price
            fx_rate: FX rate for currency conversion
            min_reduction: Minimum fraction of original position (default 0.5 = 50%)

        Returns:
            Tuple of (adjusted_quantity, adjusted_capital, was_reduced)
        """
        if required_capital <= capital:
            return quantity, required_capital, False

        # Calculate what fraction of the position we can afford
        affordable_fraction = capital / required_capital

        if affordable_fraction >= min_reduction:
            # Can afford at least min_reduction of the position
            adjusted_quantity = quantity * affordable_fraction
            adjusted_capital = adjusted_quantity * current_price * fx_rate
            return adjusted_quantity, adjusted_capital, True

        # Can't afford even the minimum reduced position
        return 0.0, 0.0, False

    def _process_signals_default_mode(self, signal_requirements: List[Tuple],
                                      capital: float, total_equity: float,
                                      current_prices: Dict[str, float],
                                      current_date: datetime,
                                      open_position_symbols: List[str],
                                      all_signal_symbols: List[str]) -> float:
        """
        Process signals in default mode (first-come, first-served).

        For each signal:
        1. Try full position size
        2. If not enough capital, try reduced position (down to 50%)
        3. If still not enough, reject the signal
        """
        num_open_positions = len(open_position_symbols)

        for symbol, signal, context, strategy, required_capital, quantity in signal_requirements:
            pm = self.position_managers[symbol]
            current_price = current_prices[symbol]
            fx_rate = self._get_fx_rate(symbol, current_date)
            other_signals = [s for s in all_signal_symbols if s != symbol]

            # Try to fit the position (full or reduced)
            adj_quantity, adj_capital, was_reduced = self._try_reduced_position(
                capital, required_capital, quantity, current_price, fx_rate
            )

            if adj_quantity > 0:
                # Can execute (either full or reduced position)
                outcome = "Executed (reduced position)" if was_reduced else "Executed (sufficient capital)"
                if was_reduced:
                    reduction_pct = (1 - adj_quantity / quantity) * 100
                    outcome = f"Executed at {100 - reduction_pct:.0f}% position size"

                self._record_capital_event(
                    current_date, symbol, "EXECUTED",
                    capital, adj_capital, total_equity,
                    num_open_positions, open_position_symbols.copy(), other_signals,
                    outcome
                )

                # Update signal with adjusted quantity for position opening
                capital = self._open_position_with_tracking(
                    symbol, current_date, current_price, signal, strategy, context,
                    capital, pm, adj_capital, num_open_positions, other_signals,
                    override_quantity=adj_quantity
                )
                num_open_positions += 1
                open_position_symbols.append(symbol)
            else:
                # Cannot fit even reduced position
                self._record_capital_event(
                    current_date, symbol, "REJECTED",
                    capital, required_capital, total_equity,
                    num_open_positions, open_position_symbols.copy(), other_signals,
                    f"Rejected: insufficient capital even for 50% position ({capital:.2f} < {required_capital * 0.5:.2f})"
                )
                self._record_signal_rejection(
                    current_date, symbol, "BUY",
                    f"Insufficient capital for even 50% position (available: {capital:.2f}, min required: {required_capital * 0.5:.2f})",
                    capital, required_capital
                )

        return capital

    def _process_signals_with_vulnerability(self, signal_requirements: List[Tuple],
                                            capital: float, total_equity: float,
                                            current_prices: Dict[str, float],
                                            current_date: datetime,
                                            open_position_symbols: List[str],
                                            all_signal_symbols: List[str]) -> float:
        """
        Process signals using vulnerability score for capital contention.

        For each signal:
        1. Try full position size
        2. If not enough capital, try reduced position (down to 50%)
        3. If still not enough, try vulnerability swap
        4. Otherwise reject the signal
        """
        num_open_positions = len(open_position_symbols)

        for symbol, signal, context, strategy, required_capital, quantity in signal_requirements:
            pm = self.position_managers[symbol]
            current_price = current_prices[symbol]
            fx_rate = self._get_fx_rate(symbol, current_date)
            other_signals = [s for s in all_signal_symbols if s != symbol]

            # Get vulnerability scores for context (used in multiple places below)
            # PERFORMANCE: Use cached scores from earlier in the day instead of recalculating
            vuln_scores = None
            if self._current_day_vuln_scores:
                vuln_scores = {s: r.score for s, r in self._current_day_vuln_scores.items()}

            # Step 1 & 2: Try full position, then reduced position
            adj_quantity, adj_capital, was_reduced = self._try_reduced_position(
                capital, required_capital, quantity, current_price, fx_rate
            )

            if adj_quantity > 0:
                # Can execute (either full or reduced position)
                outcome = "Executed (sufficient capital)"
                if was_reduced:
                    reduction_pct = (1 - adj_quantity / quantity) * 100
                    outcome = f"Executed at {100 - reduction_pct:.0f}% position size"

                self._record_capital_event(
                    current_date, symbol, "EXECUTED",
                    capital, adj_capital, total_equity,
                    num_open_positions, open_position_symbols.copy(), other_signals,
                    outcome,
                    vulnerability_scores=vuln_scores
                )

                capital = self._open_position_with_tracking(
                    symbol, current_date, current_price, signal, strategy, context,
                    capital, pm, adj_capital, num_open_positions, other_signals,
                    override_quantity=adj_quantity
                )
                num_open_positions += 1
                open_position_symbols.append(symbol)
            else:
                # Step 3: Not enough capital even for 50% position - try vulnerability swap
                capital = self._try_vulnerability_swap(
                    symbol, signal, context, strategy, capital, total_equity,
                    current_prices, current_date, required_capital, quantity,
                    open_position_symbols, other_signals
                )
                # Update tracking if swap happened
                if self.position_managers[symbol].has_position:
                    num_open_positions = sum(1 for pm in self.position_managers.values() if pm.has_position)
                    open_position_symbols = [
                        sym for sym, pm in self.position_managers.items() if pm.has_position
                    ]

        return capital

    def _try_vulnerability_swap(self, new_symbol: str, signal: Signal,
                                context: StrategyContext, strategy: BaseStrategy,
                                capital: float, total_equity: float,
                                current_prices: Dict[str, float],
                                current_date: datetime, required_capital: float,
                                quantity: float,
                                open_position_symbols: List[str],
                                other_signals: List[str]) -> float:
        """
        Try to swap a vulnerable position for a new signal.

        After a successful swap, the new position uses the full requested quantity
        since closing the old position frees up capital.
        """
        # Get open positions
        open_positions = {
            sym: self.position_managers[sym].get_position()
            for sym in open_position_symbols
            if self.position_managers[sym].has_position
        }

        if not open_positions:
            # No positions to swap
            self._record_capital_event(
                current_date, new_symbol, "REJECTED",
                capital, required_capital, total_equity,
                len(open_position_symbols), open_position_symbols.copy(), other_signals,
                "Rejected: insufficient capital and no positions to swap"
            )
            self._record_signal_rejection(
                current_date, new_symbol, "BUY",
                "Insufficient capital and no positions to swap",
                capital, required_capital
            )
            return capital

        # Get swap decision from vulnerability calculator
        swap_decision = self.vulnerability_calculator.should_swap(
            open_positions, current_prices, current_date, new_symbol
        )

        # Get vulnerability scores for reporting
        vuln_scores = {s: r.score for s, r in swap_decision.all_scores.items()} if swap_decision.all_scores else {}

        if swap_decision.should_swap:
            # Close the weak position
            weak_symbol = swap_decision.position_to_close
            weak_pm = self.position_managers[weak_symbol]
            weak_price = current_prices[weak_symbol]

            # Record the swap-out event
            self._record_capital_event(
                current_date, weak_symbol, "SWAPPED_OUT",
                capital, 0, total_equity,
                len(open_position_symbols), open_position_symbols.copy(), [new_symbol],
                f"Swapped out: vulnerability score {swap_decision.position_score:.1f} < threshold",
                vulnerability_scores=vuln_scores
            )

            # Record the swap
            self.vulnerability_swaps.append(VulnerabilitySwap(
                date=current_date,
                closed_symbol=weak_symbol,
                closed_score=swap_decision.position_score,
                new_symbol=new_symbol,
                all_scores=swap_decision.all_scores
            ))

            # Close the weak position
            capital = self._close_position(
                weak_symbol, current_date, weak_price,
                f"Vulnerability score swap (score: {swap_decision.position_score:.1f}) for {new_symbol}",
                capital, weak_pm
            )

            # Recalculate total equity after closing
            # Use last known price as fallback when current price is not available
            total_position_value = 0
            for sym, pm in self.position_managers.items():
                if pm.has_position:
                    price = current_prices.get(sym) or self._last_known_prices.get(sym)
                    if price:
                        pos_value = pm.get_position_value(price)
                        pos_value_base = self._convert_to_base_currency(pos_value, sym, current_date)
                        total_position_value += pos_value_base
            total_equity = capital + total_position_value

            # Update position tracking
            updated_open = [sym for sym, pm in self.position_managers.items() if pm.has_position]

            # Record the swap-in event
            self._record_capital_event(
                current_date, new_symbol, "SWAPPED_IN",
                capital, required_capital, total_equity,
                len(updated_open), updated_open.copy(), other_signals,
                f"Swapped in: replaced {weak_symbol} (score: {swap_decision.position_score:.1f})",
                vulnerability_scores=vuln_scores
            )

            # Open the new position
            new_pm = self.position_managers[new_symbol]
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

            # Open with the full requested quantity (capital freed from swap)
            capital = self._open_position_with_tracking(
                new_symbol, current_date, current_prices[new_symbol],
                signal, strategy, new_context, capital, new_pm,
                required_capital, len(updated_open), other_signals,
                override_quantity=quantity
            )
        else:
            # Cannot swap - record rejection with vulnerability info
            self._record_capital_event(
                current_date, new_symbol, "REJECTED",
                capital, required_capital, total_equity,
                len(open_position_symbols), open_position_symbols.copy(), other_signals,
                f"Rejected: {swap_decision.reason}",
                vulnerability_scores=vuln_scores
            )
            self._record_signal_rejection(
                current_date, new_symbol, "BUY",
                swap_decision.reason,
                capital, required_capital,
                swap_decision
            )

        return capital

    def _record_capital_event(self, date: datetime, symbol: str, event_type: str,
                              available_capital: float, required_capital: float,
                              total_equity: float, num_open_positions: int,
                              open_position_symbols: List[str],
                              competing_signals: List[str], outcome: str,
                              vulnerability_scores: Optional[Dict[str, float]] = None,
                              trade_id: Optional[str] = None):
        """Record a capital allocation event."""
        self.capital_allocation_events.append(CapitalAllocationEvent(
            date=date,
            symbol=symbol,
            signal_type=event_type,
            available_capital=available_capital,
            required_capital=required_capital,
            total_equity=total_equity,
            num_open_positions=num_open_positions,
            open_position_symbols=open_position_symbols.copy(),
            competing_signals=competing_signals.copy(),
            outcome=outcome,
            vulnerability_scores=vulnerability_scores,
            trade_id=trade_id
        ))

    def _open_position_with_tracking(self, symbol: str, date: datetime, price: float,
                                     signal: Signal, strategy: BaseStrategy,
                                     context: StrategyContext, capital: float,
                                     pm: PositionManager, required_capital: float,
                                     concurrent_positions: int,
                                     competing_signals: List[str],
                                     override_quantity: Optional[float] = None) -> float:
        """Open position with additional tracking info for the trade."""
        # Store tracking info in position manager for later trade creation
        pm.pending_trade_info = {
            'entry_capital_available': capital,
            'entry_capital_required': required_capital,
            'concurrent_positions': concurrent_positions,
            'competing_signals': competing_signals.copy()
        }
        return self._open_position(symbol, date, price, signal, strategy, context, capital, pm, override_quantity)

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
                       pm: PositionManager,
                       override_quantity: Optional[float] = None) -> float:
        """Open position with capital constraints."""
        # Apply slippage to BUY orders (pay more due to slippage)
        execution_price = price * (1 + self.config.slippage_percent / 100)

        fx_rate = self._get_fx_rate(symbol, date)

        # Use override_quantity if provided (e.g., for reduced positions)
        if override_quantity is not None:
            quantity = override_quantity
        else:
            quantity = strategy.position_size(context, signal)

        if quantity <= 0:
            return capital

        # Adjust quantity to account for slippage (reduce shares to maintain same capital allocation)
        quantity = quantity * (price / execution_price)

        # Create and execute order with slippage-adjusted price
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

        # Calculate entry equity (total equity before this position)
        total_position_value = 0
        for sym, other_pm in self.position_managers.items():
            if other_pm.has_position and sym != symbol:
                other_price = self._last_known_prices.get(sym, 0)
                if other_price:
                    pos_val = other_pm.get_position_value(other_price)
                    pos_val_base = self._convert_to_base_currency(pos_val, sym, date)
                    total_position_value += pos_val_base
        entry_equity = capital + total_position_value

        # Open position with actual execution price (including slippage)
        pm.open_position(
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

        capital -= total_cost_base
        return capital

    def _close_position(self, symbol: str, date: datetime, price: float,
                        reason: str, capital: float, pm: PositionManager) -> float:
        """Close position."""
        # Apply slippage to SELL orders (receive less due to slippage)
        execution_price = price * (1 - self.config.slippage_percent / 100)

        position = pm.get_position()
        quantity = position.current_quantity

        exit_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=execution_price,
            timestamp=date,
            reason=reason
        )

        exit_commission = self.trade_executor.execute_order(exit_order)
        proceeds = exit_order.total_value() - exit_commission
        proceeds_base = self._convert_to_base_currency(proceeds, symbol, date)

        capital += proceeds_base

        exit_fx_rate = self._get_fx_rate(symbol, date)

        # Calculate slippage cost in security currency
        slippage_pct = self.config.slippage_percent / 100
        entry_slippage_per_share = position.entry_price * slippage_pct / (1 + slippage_pct)
        entry_slippage = entry_slippage_per_share * position.initial_quantity
        exit_slippage_per_share = execution_price * slippage_pct / (1 - slippage_pct)
        exit_slippage = exit_slippage_per_share * quantity
        total_slippage_sec = entry_slippage + exit_slippage
        avg_fx_rate = (position.entry_fx_rate + exit_fx_rate) / 2
        total_slippage_base = total_slippage_sec * avg_fx_rate

        # Get capital allocation info from position manager if available
        trade_info = pm.pending_trade_info or {}

        self.trade_executor.create_trade(
            position=position,
            exit_date=date,
            exit_price=execution_price,
            exit_reason=reason,
            exit_commission=exit_commission,
            entry_fx_rate=position.entry_fx_rate,
            exit_fx_rate=exit_fx_rate,
            security_currency=position.security_currency,
            slippage_cost=total_slippage_base,
            entry_capital_available=trade_info.get('entry_capital_available', 0.0),
            entry_capital_required=trade_info.get('entry_capital_required', 0.0),
            concurrent_positions=trade_info.get('concurrent_positions', 0),
            competing_signals=trade_info.get('competing_signals', [])
        )

        # Clear pending trade info
        pm.pending_trade_info = None
        pm.close_position()
        return capital

    def _partial_exit(self, symbol: str, date: datetime, price: float,
                      fraction: float, reason: str, capital: float,
                      pm: PositionManager) -> float:
        """Partial exit."""
        # Apply slippage to SELL orders (receive less due to slippage)
        execution_price = price * (1 - self.config.slippage_percent / 100)

        position = pm.get_position()
        exit_quantity = position.current_quantity * fraction

        exit_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=exit_quantity,
            order_type=OrderType.MARKET,
            price=execution_price,
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
            price=execution_price,
            reason=reason,
            commission_paid=exit_commission
        )

        return capital

    def _pyramid_position(self, symbol: str, date: datetime, price: float,
                          signal: Signal, strategy: BaseStrategy,
                          context: StrategyContext, capital: float,
                          pm: PositionManager) -> float:
        """Add to existing position (pyramid) with break-even stop."""
        # Apply slippage to pyramid (same as BUY)
        execution_price = price * (1 + self.config.slippage_percent / 100)

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

        pyramid_commission = self.trade_executor.execute_order(pyramid_order)
        total_cost = pyramid_order.total_value() + pyramid_commission
        total_cost_base = self._convert_to_base_currency(total_cost, symbol, date)

        if total_cost_base > capital:
            return capital

        # Add pyramid to position (updates average price and sets break-even stop)
        pm.add_pyramid(
            pyramid_date=date,
            quantity=quantity,
            price=execution_price,
            commission=pyramid_commission,
            reason=signal.reason
        )

        capital -= total_cost_base
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
