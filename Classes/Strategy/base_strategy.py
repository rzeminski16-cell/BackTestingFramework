"""
Base strategy class for defining trading strategies.

Every strategy must follow this standardized structure:

REQUIRED COMPONENTS (must be implemented):
1. trade_direction: LONG or SHORT
2. required_columns(): List of raw data columns needed
3. generate_entry_signal(): Core entry rules
4. calculate_initial_stop_loss(): How initial stop loss is calculated
5. position_size(): How position size is calculated
6. generate_exit_signal(): Core exit rules (full exit)

OPTIONAL COMPONENTS (may be left blank):
1. fundamental_rules: Fundamental filter (default: AlwaysPassFundamentalRules)
2. should_adjust_stop(): Trailing stop loss rules
3. should_partial_exit(): Partial exit rules
4. should_pyramid(): Pyramiding rules (max 1 pyramid per trade)
5. max_position_duration: Max bars to hold position (checked in exit rules)

================================================================================
PARAMETER OPTIMIZATION RULES
================================================================================

When designing strategies for optimization, it is critical to understand which
parameters can be optimized and which cannot:

RAW DATA INDICATORS (NOT OPTIMIZABLE):
--------------------------------------
Any indicator that comes from the raw data CSV files CANNOT be optimized.
The period/lookback of these indicators is baked into the column name and
changing it would require regenerating the raw data files.

Examples of NON-OPTIMIZABLE parameters:
  - atr_14: The "14" period is fixed - cannot optimize to atr_10 or atr_20
  - ema_50: The "50" period is fixed - cannot optimize to ema_20 or ema_100
  - rsi_14: The "14" period is fixed
  - mfi_14: The "14" period is fixed
  - bb_upper_20: The "20" period is fixed

To use a different period, the raw data must be regenerated with that
indicator included. This is a data pipeline concern, not an optimization
concern.

CALCULATED PARAMETERS (OPTIMIZABLE):
------------------------------------
Parameters that are calculated within the strategy logic at runtime CAN be
optimized because they don't depend on pre-computed raw data columns.

Examples of OPTIMIZABLE parameters:
  - ATR multipliers (e.g., atr_stop_loss_multiple = 2.5)
  - Percentage thresholds (e.g., stop_loss_percent = 5.0)
  - Lookback windows for internal calculations (e.g., volume_ma_window = 20)
  - Risk percentages (e.g., risk_percent = 1.0)
  - Momentum thresholds (e.g., momentum_gain_pct = 5.0)
  - Grace periods (e.g., grace_period_bars = 5)
  - Score thresholds (e.g., min_score = 0.7)

STRATEGY DOCUMENTATION REQUIREMENT:
-----------------------------------
Every strategy implementation MUST clearly document in its class docstring:
1. Which parameters are RAW DATA INDICATORS (NOT OPTIMIZABLE)
2. Which parameters are OPTIMIZABLE

Example documentation format:

    RAW DATA INDICATORS (NOT OPTIMIZABLE - period fixed in data):
        - atr_14: Average True Range (14-period)
        - ema_50: Exponential Moving Average (50-period)
        - rsi_14: Relative Strength Index (14-period)

    OPTIMIZABLE PARAMETERS (can be tuned during optimization):
        - atr_multiplier (default: 2.5): ATR multiple for stop loss
        - risk_percent (default: 1.0): Portfolio risk per trade
        - min_score (default: 0.7): Minimum entry score threshold
================================================================================
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import pandas as pd

from ..Models.signal import Signal, SignalType
from ..Models.trade_direction import TradeDirection
from .strategy_context import StrategyContext
from .fundamental_rules import (
    BaseFundamentalRules,
    AlwaysPassFundamentalRules,
    FundamentalData,
    FundamentalCheckResult
)


class StrategyValidationError(Exception):
    """Raised when strategy validation fails."""
    pass


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - trade_direction: Property returning LONG or SHORT
    - required_columns(): List of column names needed from CSV (must come from raw data)
    - generate_entry_signal(): Generate entry signal (BUY)
    - calculate_initial_stop_loss(): Calculate initial stop loss price
    - position_size(): Calculate position size for entry
    - generate_exit_signal(): Generate exit signal (SELL)

    Strategies can optionally override:
    - fundamental_rules: Fundamental filter rules (default: always pass)
    - prepare_data(): Pre-calculate custom indicators (approved exceptions only)
    - should_adjust_stop(): Trailing stop logic
    - should_partial_exit(): Partial exit logic
    - should_pyramid(): Pyramiding logic (max 1 per trade)
    - max_position_duration: Max bars before forced exit

    IMPORTANT - RAW DATA REQUIREMENT:
    All indicators specified in required_columns() MUST exist in the raw data CSV.
    Custom calculations in prepare_data() are only allowed for strategy-specific
    indicators that are not standard (approved on a case-by-case basis).

    IMPORTANT - OPTIMIZATION RULES:
    When implementing a strategy, you must understand:

    1. RAW DATA INDICATORS (NOT OPTIMIZABLE):
       Indicators from CSV files like atr_14, ema_50, rsi_14 have their period
       baked into the column name. You CANNOT optimize these periods because
       changing them requires regenerating the raw data files.

    2. CALCULATED PARAMETERS (OPTIMIZABLE):
       Parameters used in strategy logic (multipliers, thresholds, percentages)
       CAN be optimized because they are computed at runtime.

    Every strategy subclass MUST document which parameters fall into each
    category in its class docstring. See the module docstring for the full
    documentation format requirement.
    """

    # Class-level validation flag - set to True to validate on init
    _validate_on_init: bool = True

    def __init__(self, **params):
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters

        Raises:
            StrategyValidationError: If required components are missing
        """
        self.params = params
        self._has_pyramided: Dict[str, bool] = {}  # Track pyramiding per symbol
        self._validate_parameters()

        # Validate strategy structure if enabled
        if self._validate_on_init:
            self._validate_strategy_structure()

    @property
    @abstractmethod
    def trade_direction(self) -> TradeDirection:
        """
        Trade direction for this strategy.

        REQUIRED: Must return TradeDirection.LONG or TradeDirection.SHORT

        Returns:
            TradeDirection enum value
        """
        pass

    @property
    def fundamental_rules(self) -> BaseFundamentalRules:
        """
        Fundamental rules for entry filtering.

        OPTIONAL: Override to provide fundamental filtering.
        Default: AlwaysPassFundamentalRules (no filtering)

        These rules are checked every time there is an entry signal.
        If they fail, the trade does not take place.

        Returns:
            BaseFundamentalRules instance
        """
        return AlwaysPassFundamentalRules()

    @property
    def max_position_duration(self) -> Optional[int]:
        """
        Maximum position duration in bars.

        OPTIONAL: Override to force exit after N bars.
        Default: None (no duration limit)

        Returns:
            Max bars to hold position, or None
        """
        return None

    @abstractmethod
    def required_columns(self) -> List[str]:
        """
        Return list of required column names from CSV data.

        REQUIRED: Must include 'date' and 'close' at minimum.
        All columns listed here MUST exist in the raw data CSV files.

        IMPORTANT: Indicators cannot be calculated if not in raw data.
        If a required indicator is missing, the backtest will fail with
        a clear error message.

        Returns:
            List of required column names (lowercase)

        Example:
            return ['date', 'close', 'atr_14', 'ema_50', 'rsi_14']
        """
        pass

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate custom strategy-specific indicators before backtesting.

        OPTIONAL: Only override for strategy-specific calculations that are
        approved exceptions (e.g., AlphaTrend signals). Standard indicators
        MUST come from raw data.

        CRITICAL - PREVENT LOOK-AHEAD BIAS:
        Only use CAUSAL operations that don't look ahead:
        - ALLOWED: .rolling(), .expanding(), .shift(n) where n >= 0
        - FORBIDDEN: .shift(-n) where n > 0, global .mean()/.std()

        Args:
            data: Raw OHLCV data with pre-calculated standard indicators

        Returns:
            Data with any custom strategy-specific indicators added

        Raises:
            ValueError: If required indicators are missing from raw data
        """
        # Validate required columns exist in raw data
        self._validate_raw_data_columns(data)

        # Store original columns for leak detection
        original_columns = set(data.columns)

        # Call implementation
        result = self._prepare_data_impl(data)

        # Check for look-ahead bias in new columns
        self._check_look_ahead_bias(result, original_columns)

        return result

    def _validate_raw_data_columns(self, data: pd.DataFrame) -> None:
        """
        Validate that all required columns exist in raw data.

        Raises:
            ValueError: If required columns are missing
        """
        required = set(self.required_columns())
        available = set(data.columns)
        missing = required - available

        if missing:
            raise ValueError(
                f"Missing required columns in raw data: {sorted(missing)}\n"
                f"These indicators must be present in the raw data CSV files.\n"
                f"Available columns: {sorted(available)}"
            )

    def _check_look_ahead_bias(self, result: pd.DataFrame, original_columns: Set[str]) -> None:
        """Check new columns for potential look-ahead bias."""
        if len(result) == 0:
            return

        new_columns = set(result.columns) - original_columns
        for col in new_columns:
            check_rows = min(10, max(1, int(len(result) * 0.05)))
            if result[col].iloc[-check_rows:].isna().any():
                print(f"\n⚠️  DATA LEAKAGE WARNING in prepare_data():")
                print(f"   Column '{col}' has NaN values at the END of the dataset.")
                print(f"   This often indicates .shift(-n) with negative offset (look-ahead bias)!")
                print(f"   Only use causal operations: .rolling(), .expanding(), .shift(n>=0)\n")

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Internal implementation of prepare_data().

        Override this method for custom indicator calculations.
        Default: Returns data unchanged.

        Args:
            data: Raw data with standard indicators

        Returns:
            Data with custom indicators added
        """
        return data

    @abstractmethod
    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate entry signal based on entry rules.

        REQUIRED: Core entry logic for the strategy.
        Called on every bar when not in a position.

        Should return Signal.buy() when entry conditions are met.
        The stop_loss from calculate_initial_stop_loss() will be used.

        Note: Fundamental rules are checked AFTER this method returns
        a BUY signal. If fundamentals fail, the trade is skipped.

        Args:
            context: Current market context

        Returns:
            Signal.buy() if entry conditions met, None otherwise

        Example:
            if context.current_price > context.get_indicator_value('ema_50'):
                return Signal.buy(
                    size=1.0,
                    stop_loss=self.calculate_initial_stop_loss(context),
                    reason="Price above EMA-50",
                    direction=self.trade_direction
                )
            return None
        """
        pass

    @abstractmethod
    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """
        Calculate initial stop loss price for entry.

        REQUIRED: Define how the initial stop loss is calculated.
        Called when generating entry signals.

        Args:
            context: Current market context

        Returns:
            Stop loss price (absolute value)

        Example (ATR-based):
            atr = context.get_indicator_value('atr_14')
            return context.current_price - (atr * 2.5)

        Example (percentage-based):
            return context.current_price * 0.95  # 5% stop
        """
        pass

    @abstractmethod
    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size (number of shares/units).

        REQUIRED: Define how position size is calculated.
        Called when a BUY signal is generated.

        Args:
            context: Current market context
            signal: BUY signal with stop_loss set

        Returns:
            Number of shares/units to buy (must be > 0)

        Example (risk-based):
            equity = context.total_equity
            risk_amount = equity * (self.risk_percent / 100)
            stop_distance = context.current_price - signal.stop_loss
            return risk_amount / (stop_distance * context.fx_rate)

        Example (capital fraction):
            capital_to_use = context.available_capital * signal.size
            return capital_to_use / context.current_price
        """
        pass

    @abstractmethod
    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """
        Generate exit signal based on exit rules.

        REQUIRED: Core exit logic for the strategy.
        Called on every bar when in a position (after stop loss checks).

        Should return Signal.sell() when exit conditions are met.

        Args:
            context: Current market context

        Returns:
            Signal.sell() if exit conditions met, None otherwise

        Example:
            # Max duration check
            if self.max_position_duration:
                bars_held = context.current_index - self._entry_bar[context.symbol]
                if bars_held >= self.max_position_duration:
                    return Signal.sell(reason="Max duration reached")

            # Technical exit
            if context.current_price < context.get_indicator_value('ema_50'):
                return Signal.sell(reason="Price below EMA-50")

            return None
        """
        pass

    def should_check_stop_loss(self, context: StrategyContext) -> bool:
        """
        Determine if stop loss should be checked on this bar.

        Default: Always check if position has a stop loss.
        Override for custom logic (e.g., only check after N bars).

        Args:
            context: Current market context

        Returns:
            True if stop loss should be checked
        """
        return context.has_position and context.position.stop_loss is not None

    def should_check_take_profit(self, context: StrategyContext) -> bool:
        """
        Determine if take profit should be checked on this bar.

        Default: Always check if position has a take profit.

        Args:
            context: Current market context

        Returns:
            True if take profit should be checked
        """
        return context.has_position and context.position.take_profit is not None

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """
        Check if stop loss should be adjusted (trailing stop logic).

        OPTIONAL: Override to implement trailing stops.
        Default: No adjustment (returns None).

        Note: For LONG positions, stop can only move UP.
              For SHORT positions, stop can only move DOWN.
        This is enforced by the engine.

        Args:
            context: Current market context

        Returns:
            New stop loss price, or None if no adjustment needed

        Example:
            # Trail stop using ATR
            atr = context.get_indicator_value('atr_14')
            new_stop = context.current_price - (atr * 2.0)
            if new_stop > context.position.stop_loss:
                return new_stop
            return None
        """
        return None

    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        """
        Check if partial exit should be taken.

        OPTIONAL: Override to implement partial profit-taking.
        Default: No partial exits (returns None).

        Args:
            context: Current market context

        Returns:
            Fraction of position to exit (0.0-1.0), or None

        Example:
            # Take 50% profit at 10% gain
            if context.get_position_pl_pct() > 10.0:
                return 0.5
            return None
        """
        return None

    def should_pyramid(self, context: StrategyContext) -> Optional[Signal]:
        """
        Check if position should be pyramided (added to).

        OPTIONAL: Override to implement pyramiding.
        Default: No pyramiding (returns None).

        IMPORTANT:
        - Only ONE pyramid is allowed per trade
        - When pyramiding occurs, stop loss automatically moves to break-even
          (considering increased position size and commission costs)
        - The engine tracks pyramiding state

        Args:
            context: Current market context

        Returns:
            Signal.pyramid() if should add to position, None otherwise

        Example:
            # Pyramid when price moves 5% in our favor
            if context.get_position_pl_pct() > 5.0:
                return Signal.pyramid(
                    size=0.5,  # Add 50% of original position
                    reason="Price momentum continuation"
                )
            return None
        """
        return None

    def check_fundamentals(self, context: StrategyContext) -> FundamentalCheckResult:
        """
        Check fundamental rules for entry.

        Called automatically when an entry signal is generated.
        Uses the fundamental_rules property to perform the check.

        Args:
            context: Current market context

        Returns:
            FundamentalCheckResult with passed=True/False
        """
        fundamentals = FundamentalData.from_bar(context.current_bar)
        return self.fundamental_rules.check_fundamentals(context, fundamentals)

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on current context.

        This is the main method called by the engine on every bar.
        It orchestrates the strategy logic flow:

        1. If no position: Check entry signal + fundamentals
        2. If in position: Check exit signal, pyramiding

        Args:
            context: Current market context

        Returns:
            Signal (BUY, SELL, PYRAMID, or HOLD)
        """
        symbol = context.symbol

        if not context.has_position:
            # Reset pyramiding tracker when not in position
            self._has_pyramided[symbol] = False

            # Check for entry signal
            entry_signal = self.generate_entry_signal(context)

            if entry_signal is not None and entry_signal.type == SignalType.BUY:
                # Check fundamental rules before allowing entry
                fundamental_result = self.check_fundamentals(context)

                if not fundamental_result.passed:
                    # Fundamentals failed - no trade
                    return Signal.hold(
                        reason=f"Fundamentals failed: {fundamental_result.reason}"
                    )

                # Ensure stop loss is set
                if entry_signal.stop_loss is None:
                    entry_signal = Signal.buy(
                        size=entry_signal.size,
                        stop_loss=self.calculate_initial_stop_loss(context),
                        take_profit=entry_signal.take_profit,
                        direction=self.trade_direction,
                        reason=entry_signal.reason
                    )

                return entry_signal

        else:
            # In position - check for exit first
            exit_signal = self.generate_exit_signal(context)

            if exit_signal is not None and exit_signal.type == SignalType.SELL:
                return exit_signal

            # Check for pyramiding (only if not already pyramided)
            if not self._has_pyramided.get(symbol, False):
                pyramid_signal = self.should_pyramid(context)
                if pyramid_signal is not None and pyramid_signal.type == SignalType.PYRAMID:
                    self._has_pyramided[symbol] = True
                    return pyramid_signal

        return Signal.hold()

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Override to add custom parameter validation.

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def _validate_strategy_structure(self) -> None:
        """
        Validate that strategy implements all required components.

        Raises:
            StrategyValidationError: If required components are missing
        """
        errors = []

        # Check trade_direction
        try:
            direction = self.trade_direction
            if not isinstance(direction, TradeDirection):
                errors.append(
                    f"trade_direction must return TradeDirection enum, "
                    f"got {type(direction).__name__}"
                )
        except NotImplementedError:
            errors.append("trade_direction property not implemented")

        # Check required_columns returns non-empty list with date and close
        try:
            columns = self.required_columns()
            if not columns:
                errors.append("required_columns() returned empty list")
            else:
                if 'date' not in columns:
                    errors.append("required_columns() must include 'date'")
                if 'close' not in columns:
                    errors.append("required_columns() must include 'close'")
        except NotImplementedError:
            errors.append("required_columns() not implemented")

        if errors:
            raise StrategyValidationError(
                f"Strategy validation failed for {self.__class__.__name__}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get strategy parameter value.

        Args:
            name: Parameter name
            default: Default value if parameter not set

        Returns:
            Parameter value
        """
        return self.params.get(name, default)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return self.params.copy()

    def get_name(self) -> str:
        """
        Get strategy name.

        Default: Class name.
        Override to provide custom name.

        Returns:
            Strategy name
        """
        return self.__class__.__name__

    def __str__(self) -> str:
        """String representation of strategy."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        direction = self.trade_direction.value
        return f"{self.get_name()}[{direction}]({param_str})"
