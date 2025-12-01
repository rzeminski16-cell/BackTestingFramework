"""
Base strategy class for defining trading strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from ..Models.signal import Signal
from .strategy_context import StrategyContext


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies must implement:
    - required_columns(): List of column names needed from CSV
    - generate_signal(): Generate trading signal based on current context

    Strategies can optionally override:
    - prepare_data(): Pre-calculate indicators before backtest (RECOMMENDED for performance)
    - position_size(): Calculate position size for entry
    - should_check_stop_loss(): Check if stop loss should be hit
    - should_check_take_profit(): Check if take profit should be hit
    - should_adjust_stop(): Check if stop loss should be adjusted
    - should_partial_exit(): Check if partial exit should be taken

    Strategy Parameters:
    - Pass parameters via __init__ and store as instance variables
    - This allows for easy parameter optimization

    Performance Optimization:
    - Override prepare_data() to pre-calculate all indicators ONCE before backtesting
    - This eliminates O(n²) complexity from repeated calculations during the bar loop
    - Use IndicatorEngine for vectorized indicator calculations
    """

    def __init__(self, **params):
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
        self._validate_parameters()

    @abstractmethod
    def required_columns(self) -> List[str]:
        """
        Return list of required column names from CSV data.

        Must include 'date' and 'close' at minimum.
        Include any indicators needed for the strategy.

        Returns:
            List of required column names (lowercase)

        Example:
            return ['date', 'close', 'sma_50', 'rsi_14']
        """
        pass

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate custom strategy-specific indicators before backtesting begins.

        STANDARD INDICATORS: All standard indicators (atr_14, ema_50, sma_200, rsi_14,
        cmf, par_sar, base_bb, upper_bb, lower_bb) are read from raw data with fixed
        settings and cannot be altered via parameters.

        CUSTOM INDICATORS: Override this method to calculate strategy-specific indicators
        once using vectorized operations. This is called ONCE before the backtest loop,
        providing 10-100x speedup for indicator-heavy strategies.

        ⚠️  CRITICAL - PREVENT LOOK-AHEAD BIAS:
        Only use CAUSAL operations that don't look ahead:
        - ✅ ALLOWED: .rolling(), .expanding(), .shift(n) where n >= 0, .pct_change()
        - ❌ FORBIDDEN: .shift(-n) where n > 0, global .mean()/.std() on full series

        The framework will automatically detect potential look-ahead bias by checking
        for suspicious NaN patterns in new columns.

        Default implementation: Returns data unchanged (assumes all indicators pre-exist).

        Args:
            data: Raw OHLCV data with pre-calculated standard indicators

        Returns:
            Data with any custom strategy-specific indicators added as columns

        Example (reading standard indicators from raw data):
            def required_columns(self) -> List[str]:
                return ['date', 'close', 'atr_14', 'ema_50', 'rsi_14']

            def generate_signal(self, context: StrategyContext) -> Signal:
                atr = context.get_indicator_value('atr_14')  # Read from raw data
                ema = context.get_indicator_value('ema_50')  # Read from raw data
                ...

        Example (calculating custom indicators - CORRECT):
            def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
                df = data.copy()
                # ✅ GOOD: Rolling operations (look backward only)
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['returns'] = df['close'].pct_change()  # Uses shift(1) internally
                return df

        Example (calculating custom indicators - INCORRECT):
            def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
                df = data.copy()
                # ❌ BAD: Looking ahead 5 bars!
                df['future_return'] = df['close'].shift(-5)
                # ❌ BAD: Using statistics from entire dataset!
                df['z_score'] = (df['close'] - df['close'].mean()) / df['close'].std()
                return df
        """
        # Store original columns to detect new ones
        original_columns = set(data.columns)

        # Call the actual implementation (subclasses should implement this logic)
        result = self._prepare_data_impl(data)

        # DATA LEAKAGE VALIDATION: Check for non-causal operations
        if len(result) > 0:
            new_columns = set(result.columns) - original_columns

            for col in new_columns:
                # Check last rows for NaN (backward shift with .shift(-n) creates trailing NaNs)
                # Check at least 5% of data or 10 rows, whichever is smaller
                check_rows = min(10, max(1, int(len(result) * 0.05)))

                if result[col].iloc[-check_rows:].isna().any():
                    print(f"\n⚠️  DATA LEAKAGE WARNING in prepare_data():")
                    print(f"   Column '{col}' has NaN values at the END of the dataset.")
                    print(f"   This often indicates .shift(-n) with negative offset (look-ahead bias)!")
                    print(f"   Please verify that '{col}' only uses causal operations:")
                    print(f"   - ✅ ALLOWED: .rolling(), .expanding(), .shift(n) where n >= 0")
                    print(f"   - ❌ FORBIDDEN: .shift(-n) where n > 0\n")

        return result

    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Internal implementation of prepare_data().

        Subclasses should typically NOT override prepare_data() directly.
        Instead, keep the default prepare_data() implementation and just
        return the data directly from this method or override prepare_data()
        and be aware of the validation that occurs.

        Args:
            data: Raw OHLCV data with pre-calculated standard indicators

        Returns:
            Data with any custom strategy-specific indicators added as columns
        """
        return data

    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on current context.

        This is the main strategy logic. Called on every bar.

        Args:
            context: Current market context (data, position, prices, etc.)

        Returns:
            Signal (BUY, SELL, PARTIAL_EXIT, ADJUST_STOP, or HOLD)

        Example:
            if not context.has_position:
                if context.current_price > context.get_indicator_value('sma_50'):
                    return Signal.buy(size=0.1, reason="Price above SMA")
            else:
                if context.current_price < context.get_indicator_value('sma_50'):
                    return Signal.sell(reason="Price below SMA")
            return Signal.hold()
        """
        pass

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """
        Calculate position size (number of shares/units) for a BUY signal.

        Default: Uses signal.size as fraction of available capital.

        Override this method for custom position sizing logic (e.g., ATR-based,
        volatility-adjusted, Kelly criterion, etc.)

        Args:
            context: Current market context
            signal: BUY signal with size parameter

        Returns:
            Number of shares/units to buy (must be > 0)

        Example:
            # Default implementation
            capital_to_use = context.available_capital * signal.size
            shares = capital_to_use / context.current_price
            return shares
        """
        if signal.size <= 0 or signal.size > 1.0:
            raise ValueError(f"Signal size must be between 0 and 1, got {signal.size}")

        capital_to_use = context.available_capital * signal.size
        shares = capital_to_use / context.current_price

        return shares

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

        Override for custom logic.

        Args:
            context: Current market context

        Returns:
            True if take profit should be checked
        """
        return context.has_position and context.position.take_profit is not None

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """
        Check if stop loss should be adjusted (trailing stop logic).

        Default: No adjustment (returns None).

        Override this method to implement:
        - Trailing stops based on price movement
        - Trailing stops based on indicator values
        - Conditional stop adjustments (e.g., move to breakeven after X% profit)

        Args:
            context: Current market context

        Returns:
            New stop loss price, or None if no adjustment needed

        Example:
            # Move stop to breakeven after 5% profit
            if context.get_position_pl_pct() > 5.0:
                return context.position.entry_price

            # Trail stop using EMA
            ema_value = context.get_indicator_value('ema_14')
            if ema_value and ema_value > context.position.stop_loss:
                return ema_value

            return None
        """
        return None

    def should_partial_exit(self, context: StrategyContext) -> Optional[float]:
        """
        Check if partial exit should be taken.

        Default: No partial exits (returns None).

        Override this method to implement partial profit-taking:
        - Scale out at specific profit levels
        - Scale out based on indicator signals
        - Time-based partial exits

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

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Override to add custom parameter validation.

        Raises:
            ValueError: If parameters are invalid
        """
        pass

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
        return f"{self.get_name()}({param_str})"
