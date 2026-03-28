"""
Comprehensive tests for Strategy framework:
- BaseStrategy validation
- StrategyContext data access
- Signal generation flow (generate_signal orchestration)
- Fundamental rules integration
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

from Classes.Strategy.base_strategy import BaseStrategy, StrategyValidationError
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal, SignalType
from Classes.Models.trade_direction import TradeDirection
from Classes.Models.position import Position


# =============================================================================
# Test Strategy Implementations
# =============================================================================

class ValidLongStrategy(BaseStrategy):
    """Minimal valid LONG strategy for testing."""
    _validate_on_init = True

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_price > 100:
            return Signal.buy(size=1.0, stop_loss=context.current_price * 0.95,
                              reason="Above 100", direction=self.trade_direction)
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        return context.current_price * 0.95

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_price < 95:
            return Signal.sell(reason="Below 95")
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return context.available_capital / context.current_price


class ValidShortStrategy(BaseStrategy):
    """Minimal valid SHORT strategy."""
    _validate_on_init = True

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.SHORT

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_price < 90:
            return Signal.buy(size=1.0, stop_loss=95, direction=self.trade_direction,
                              reason="Short below 90")
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        return context.current_price * 1.05

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return 100.0


class MissingDateStrategy(BaseStrategy):
    """Strategy that doesn't include 'date' in required_columns."""
    _validate_on_init = True

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['close']  # Missing 'date'

    def generate_entry_signal(self, context): return None
    def calculate_initial_stop_loss(self, context): return 0
    def generate_exit_signal(self, context): return None
    def position_size(self, context, signal): return 0


class MissingCloseStrategy(BaseStrategy):
    """Strategy that doesn't include 'close' in required_columns."""
    _validate_on_init = True

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date']  # Missing 'close'

    def generate_entry_signal(self, context): return None
    def calculate_initial_stop_loss(self, context): return 0
    def generate_exit_signal(self, context): return None
    def position_size(self, context, signal): return 0


class StrategyWithPyramid(BaseStrategy):
    """Strategy with pyramiding enabled."""
    _validate_on_init = False

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context):
        return Signal.buy(size=1.0, stop_loss=90, reason="entry")

    def calculate_initial_stop_loss(self, context):
        return 90.0

    def generate_exit_signal(self, context):
        return None

    def position_size(self, context, signal):
        return 100.0

    def should_pyramid(self, context):
        if context.current_price > 110:
            return Signal.pyramid(size=0.5, reason="momentum")
        return None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample price data."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0, 1, 50))
    return pd.DataFrame({
        'date': dates,
        'close': close,
        'open': close - 0.5,
        'high': close + 1,
        'low': close - 1,
        'volume': np.random.randint(1000, 10000, 50)
    })


@pytest.fixture
def context_no_position(sample_data):
    """Create context without position."""
    return StrategyContext(
        data=sample_data,
        current_index=25,
        current_price=float(sample_data.iloc[25]['close']),
        current_date=sample_data.iloc[25]['date'],
        position=None,
        available_capital=100000.0,
        total_equity=100000.0,
        symbol="TEST"
    )


@pytest.fixture
def context_with_position(sample_data):
    """Create context with open position."""
    pos = Position(
        symbol="TEST",
        entry_date=sample_data.iloc[10]['date'].to_pydatetime(),
        entry_price=100.0,
        initial_quantity=100.0,
        current_quantity=100.0,
        direction=TradeDirection.LONG,
        stop_loss=90.0,
    )
    return StrategyContext(
        data=sample_data,
        current_index=25,
        current_price=float(sample_data.iloc[25]['close']),
        current_date=sample_data.iloc[25]['date'],
        position=pos,
        available_capital=50000.0,
        total_equity=60000.0,
        symbol="TEST"
    )


# =============================================================================
# BaseStrategy Validation Tests
# =============================================================================

class TestBaseStrategyValidation:
    def test_valid_long_strategy(self):
        strategy = ValidLongStrategy()
        assert strategy.trade_direction == TradeDirection.LONG

    def test_valid_short_strategy(self):
        strategy = ValidShortStrategy()
        assert strategy.trade_direction == TradeDirection.SHORT

    def test_missing_date_rejected(self):
        with pytest.raises(StrategyValidationError, match="date"):
            MissingDateStrategy()

    def test_missing_close_rejected(self):
        with pytest.raises(StrategyValidationError, match="close"):
            MissingCloseStrategy()

    def test_get_name(self):
        strategy = ValidLongStrategy()
        assert strategy.get_name() == "ValidLongStrategy"

    def test_get_parameters(self):
        strategy = ValidLongStrategy(param_a=1, param_b="test")
        params = strategy.get_parameters()
        assert params == {'param_a': 1, 'param_b': 'test'}

    def test_get_parameter_default(self):
        strategy = ValidLongStrategy()
        assert strategy.get_parameter('nonexistent', 42) == 42

    def test_str_representation(self):
        strategy = ValidLongStrategy()
        s = str(strategy)
        assert "ValidLongStrategy" in s
        assert "LONG" in s


# =============================================================================
# StrategyContext Tests
# =============================================================================

class TestStrategyContext:
    def test_has_position_false(self, context_no_position):
        assert context_no_position.has_position is False

    def test_has_position_true(self, context_with_position):
        assert context_with_position.has_position is True

    def test_current_bar(self, context_no_position, sample_data):
        bar = context_no_position.current_bar
        assert bar['close'] == sample_data.iloc[25]['close']

    def test_previous_bar(self, context_no_position, sample_data):
        bar = context_no_position.previous_bar
        assert bar['close'] == sample_data.iloc[24]['close']

    def test_previous_bar_at_start(self, sample_data):
        ctx = StrategyContext(
            data=sample_data, current_index=0,
            current_price=float(sample_data.iloc[0]['close']),
            current_date=sample_data.iloc[0]['date'],
            position=None, available_capital=100000, total_equity=100000
        )
        assert ctx.previous_bar is None

    def test_get_bar_offset(self, context_no_position, sample_data):
        bar = context_no_position.get_bar(-2)
        assert bar['close'] == sample_data.iloc[23]['close']

    def test_get_bar_out_of_bounds(self, context_no_position):
        assert context_no_position.get_bar(-100) is None
        assert context_no_position.get_bar(100) is None

    def test_get_indicator_value(self, context_no_position, sample_data):
        val = context_no_position.get_indicator_value('close')
        assert val == sample_data.iloc[25]['close']

    def test_get_indicator_value_nonexistent(self, context_no_position):
        assert context_no_position.get_indicator_value('nonexistent_col') is None

    def test_get_position_pl_no_position(self, context_no_position):
        assert context_no_position.get_position_pl() == 0.0

    def test_get_position_pl_with_position(self, context_with_position):
        pl = context_with_position.get_position_pl()
        # Position entry at 100, current price varies
        assert isinstance(pl, float)

    def test_get_position_pl_pct_no_position(self, context_no_position):
        assert context_no_position.get_position_pl_pct() == 0.0


# =============================================================================
# Signal Generation Orchestration Tests
# =============================================================================

class TestGenerateSignal:
    def test_no_position_entry_signal(self, context_no_position):
        """When no position and entry conditions met, should get BUY."""
        strategy = ValidLongStrategy()
        # Override price to be > 100 to trigger entry
        if context_no_position.current_price > 100:
            signal = strategy.generate_signal(context_no_position)
            assert signal.type == SignalType.BUY
        else:
            signal = strategy.generate_signal(context_no_position)
            assert signal.type == SignalType.HOLD

    def test_no_position_no_entry(self, sample_data):
        """When no position and no entry condition, should HOLD."""
        strategy = ValidLongStrategy()
        # Set price to 50, which is below 100 threshold
        ctx = StrategyContext(
            data=sample_data, current_index=25,
            current_price=50.0,
            current_date=sample_data.iloc[25]['date'],
            position=None, available_capital=100000.0,
            total_equity=100000.0, symbol="TEST"
        )
        signal = strategy.generate_signal(ctx)
        assert signal.type == SignalType.HOLD

    def test_in_position_exit_signal(self, sample_data):
        """When in position and exit conditions met, should SELL."""
        strategy = ValidLongStrategy()
        pos = Position(
            symbol="TEST",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
        )
        # Price below 95 triggers exit
        ctx = StrategyContext(
            data=sample_data, current_index=25,
            current_price=90.0,
            current_date=sample_data.iloc[25]['date'],
            position=pos, available_capital=50000.0,
            total_equity=59000.0, symbol="TEST"
        )
        signal = strategy.generate_signal(ctx)
        assert signal.type == SignalType.SELL

    def test_in_position_hold(self, sample_data):
        """When in position but no exit condition, should HOLD."""
        strategy = ValidLongStrategy()
        pos = Position(
            symbol="TEST",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
        )
        # Price 105 is above 95 exit threshold
        ctx = StrategyContext(
            data=sample_data, current_index=25,
            current_price=105.0,
            current_date=sample_data.iloc[25]['date'],
            position=pos, available_capital=50000.0,
            total_equity=60500.0, symbol="TEST"
        )
        signal = strategy.generate_signal(ctx)
        assert signal.type == SignalType.HOLD

    def test_entry_signal_sets_stop_loss(self, sample_data):
        """Entry signal should have stop_loss set."""
        strategy = ValidLongStrategy()
        ctx = StrategyContext(
            data=sample_data, current_index=25,
            current_price=105.0,
            current_date=sample_data.iloc[25]['date'],
            position=None, available_capital=100000.0,
            total_equity=100000.0, symbol="TEST"
        )
        signal = strategy.generate_signal(ctx)
        if signal.type == SignalType.BUY:
            assert signal.stop_loss is not None
            assert signal.stop_loss < 105.0

    def test_pyramid_signal(self, sample_data):
        """Strategy with pyramiding should emit PYRAMID signal."""
        strategy = StrategyWithPyramid()
        pos = Position(
            symbol="TEST",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
        )
        ctx = StrategyContext(
            data=sample_data, current_index=25,
            current_price=115.0,  # Above 110 threshold
            current_date=sample_data.iloc[25]['date'],
            position=pos, available_capital=50000.0,
            total_equity=61500.0, symbol="TEST"
        )
        signal = strategy.generate_signal(ctx)
        assert signal.type == SignalType.PYRAMID

    def test_pyramid_only_once(self, sample_data):
        """Should only pyramid once per symbol."""
        strategy = StrategyWithPyramid()
        pos = Position(
            symbol="TEST",
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=90.0,
        )
        ctx = StrategyContext(
            data=sample_data, current_index=25,
            current_price=115.0,
            current_date=sample_data.iloc[25]['date'],
            position=pos, available_capital=50000.0,
            total_equity=61500.0, symbol="TEST"
        )
        # First pyramid
        signal1 = strategy.generate_signal(ctx)
        assert signal1.type == SignalType.PYRAMID
        # Second should be HOLD
        signal2 = strategy.generate_signal(ctx)
        assert signal2.type == SignalType.HOLD


# =============================================================================
# Default Optional Method Tests
# =============================================================================

class TestDefaultOptionalMethods:
    def test_should_adjust_stop_default(self, context_with_position):
        strategy = ValidLongStrategy()
        assert strategy.should_adjust_stop(context_with_position) is None

    def test_should_partial_exit_default(self, context_with_position):
        strategy = ValidLongStrategy()
        assert strategy.should_partial_exit(context_with_position) is None

    def test_should_pyramid_default(self, context_with_position):
        strategy = ValidLongStrategy()
        assert strategy.should_pyramid(context_with_position) is None

    def test_prepare_data_default(self, sample_data):
        strategy = ValidLongStrategy()
        result = strategy.prepare_data(sample_data)
        assert len(result) == len(sample_data)
