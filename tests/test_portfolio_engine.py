"""
Tests for PortfolioEngine and its data classes.

Tests verify:
- PortfolioBacktestResult, CapitalAllocationEvent, SignalRejection, VulnerabilitySwap dataclasses
- _try_reduced_position logic
- Multi-security backtesting (basic run, positions across symbols)
- Capital contention DEFAULT mode (signal rejection when capital exhausted)
- Capital contention VULNERABILITY_SCORE mode (swap of vulnerable positions)
- Equity curve tracks portfolio-level equity
- Signal rejection tracking
- Positions closed at end of backtest
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from Classes.Engine.portfolio_engine import (
    PortfolioEngine,
    PortfolioBacktestResult,
    CapitalAllocationEvent,
    SignalRejection,
    VulnerabilitySwap,
    RejectionPositionContext,
)
from Classes.Engine.backtest_result import BacktestResult
from Classes.Config.config import (
    PortfolioConfig, BacktestConfig, CommissionConfig, CommissionMode,
)
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig,
)
from Classes.Models.signal import Signal, SignalType
from Classes.Models.trade import Trade, reset_trade_counter
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext


# =============================================================================
# Deterministic Strategy for Portfolio Tests
# =============================================================================

class PortfolioDeterministicStrategy(BaseStrategy):
    """Strategy that buys/sells on specified bar indices, per-symbol aware."""

    _validate_on_init = False

    def __init__(self, buy_bars=None, sell_bars=None, stop_loss_price=None,
                 take_profit_price=None, position_size_shares=100.0):
        self.buy_bars = set(buy_bars or [])
        self.sell_bars = set(sell_bars or [])
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.position_size_shares = position_size_shares
        super().__init__()

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index in self.buy_bars:
            return Signal.buy(
                size=1.0,
                stop_loss=self.stop_loss_price,
                take_profit=self.take_profit_price,
                reason="Test buy",
                direction=self.trade_direction,
            )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        if self.stop_loss_price is not None:
            return self.stop_loss_price
        return context.current_price * 0.50  # Very loose stop to avoid triggers

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index in self.sell_bars:
            return Signal.sell(reason="Test sell")
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return self.position_size_shares


# =============================================================================
# Fixtures
# =============================================================================

def _make_price_data(symbol_offset=0.0, num_bars=30):
    """Create simple price data. offset shifts prices for different symbols."""
    dates = pd.date_range('2024-01-01', periods=num_bars, freq='D')
    prices = [100.0 + symbol_offset + i * 0.5 for i in range(num_bars)]
    return pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 1 for p in prices],
        'close': prices,
        'volume': [10000] * num_bars,
    })


def _make_portfolio_config(initial_capital=100000.0, mode=CapitalContentionMode.DEFAULT,
                           vuln_config=None):
    """Create a PortfolioConfig."""
    cc = CapitalContentionConfig(mode=mode, vulnerability_config=vuln_config)
    return PortfolioConfig(
        initial_capital=initial_capital,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
        capital_contention=cc,
    )


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    def test_capital_allocation_event(self):
        event = CapitalAllocationEvent(
            date=datetime(2024, 1, 5),
            symbol="AAPL",
            signal_type="EXECUTED",
            available_capital=50000.0,
            required_capital=10000.0,
            total_equity=100000.0,
            num_open_positions=1,
            open_position_symbols=["GOOG"],
            competing_signals=["MSFT"],
            outcome="Sufficient capital for all signals",
        )
        assert event.symbol == "AAPL"
        assert event.signal_type == "EXECUTED"
        assert event.vulnerability_scores is None

    def test_signal_rejection(self):
        rej = SignalRejection(
            date=datetime(2024, 1, 5),
            symbol="AAPL",
            signal_type="BUY",
            reason="Insufficient capital",
            available_capital=100.0,
            required_capital=50000.0,
        )
        assert rej.symbol == "AAPL"
        assert rej.vulnerability_decision is None

    def test_vulnerability_swap(self):
        swap = VulnerabilitySwap(
            date=datetime(2024, 1, 5),
            closed_symbol="GOOG",
            closed_score=25.0,
            new_symbol="AAPL",
        )
        assert swap.closed_symbol == "GOOG"
        assert swap.all_scores == {}

    def test_portfolio_backtest_result_fields(self):
        """PortfolioBacktestResult should store all tracking data."""
        config = _make_portfolio_config()
        result = PortfolioBacktestResult(
            symbol_results={},
            final_equity=100000.0,
            total_return=0.0,
            total_return_pct=0.0,
            portfolio_equity_curve=pd.DataFrame(),
            signal_rejections=[],
            vulnerability_swaps=[],
            vulnerability_history=[],
            capital_allocation_events=[],
            config=config,
            strategy_name="Test",
        )
        assert result.final_equity == 100000.0
        assert result.strategy_name == "Test"


# =============================================================================
# _try_reduced_position Tests
# =============================================================================

class TestTryReducedPosition:
    """Test the position reduction logic directly."""

    def setup_method(self):
        config = _make_portfolio_config()
        self.engine = PortfolioEngine(config)

    def test_full_position_fits(self):
        adj_qty, adj_cap, reduced = self.engine._try_reduced_position(
            capital=10000.0, required_capital=5000.0,
            quantity=100.0, current_price=50.0, fx_rate=1.0,
        )
        assert adj_qty == 100.0
        assert adj_cap == 5000.0
        assert reduced is False

    def test_reduced_position_above_minimum(self):
        """If we can afford 75% of position (above 50% min), it should be reduced."""
        adj_qty, adj_cap, reduced = self.engine._try_reduced_position(
            capital=7500.0, required_capital=10000.0,
            quantity=100.0, current_price=100.0, fx_rate=1.0,
        )
        assert reduced is True
        assert adj_qty == pytest.approx(75.0)
        assert adj_qty > 0

    def test_reduced_position_below_minimum_rejected(self):
        """If we can only afford 40% (below 50% min), reject."""
        adj_qty, adj_cap, reduced = self.engine._try_reduced_position(
            capital=4000.0, required_capital=10000.0,
            quantity=100.0, current_price=100.0, fx_rate=1.0,
        )
        assert adj_qty == 0.0
        assert adj_cap == 0.0
        assert reduced is False

    def test_exact_capital_match(self):
        """Exact match should use full position (not reduced)."""
        adj_qty, adj_cap, reduced = self.engine._try_reduced_position(
            capital=10000.0, required_capital=10000.0,
            quantity=100.0, current_price=100.0, fx_rate=1.0,
        )
        assert adj_qty == 100.0
        assert reduced is False

    def test_fx_rate_applied(self):
        """FX rate should be used in capital calculation."""
        adj_qty, adj_cap, reduced = self.engine._try_reduced_position(
            capital=15000.0, required_capital=20000.0,
            quantity=100.0, current_price=100.0, fx_rate=2.0,
            min_reduction=0.5,
        )
        assert reduced is True
        # affordable_fraction = 15000/20000 = 0.75
        assert adj_qty == pytest.approx(75.0)
        # adjusted_capital = 75 * 100 * 2.0 = 15000
        assert adj_cap == pytest.approx(15000.0)


# =============================================================================
# Portfolio Engine Integration Tests
# =============================================================================

class TestPortfolioBasic:
    """Basic portfolio engine integration tests."""

    def test_single_security_run(self):
        """Run portfolio engine with single security and one trade."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data = _make_price_data()
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0
        )

        result = engine.run({"AAPL": data}, strategy)
        assert isinstance(result, PortfolioBacktestResult)
        assert "AAPL" in result.symbol_results
        assert result.strategy_name == "PortfolioDeterministicStrategy"

    def test_two_securities_run(self):
        """Run with two securities, each getting one trade."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0),
            "GOOG": _make_price_data(symbol_offset=50.0),
        }
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0,
            position_size_shares=50,
        )

        result = engine.run(data_dict, strategy)
        assert "AAPL" in result.symbol_results
        assert "GOOG" in result.symbol_results

    def test_equity_curve_created(self):
        """Equity curve should be populated after run."""
        reset_trade_counter()
        config = _make_portfolio_config()
        engine = PortfolioEngine(config)

        data = _make_price_data()
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0
        )

        result = engine.run({"AAPL": data}, strategy)
        assert len(result.portfolio_equity_curve) > 0
        assert 'equity' in result.portfolio_equity_curve.columns
        assert 'date' in result.portfolio_equity_curve.columns

    def test_equity_starts_at_initial_capital(self):
        """First equity value should be the initial capital."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data = _make_price_data()
        # No trades - just hold
        strategy = PortfolioDeterministicStrategy(buy_bars=set(), sell_bars=set())

        result = engine.run({"AAPL": data}, strategy)
        first_equity = result.portfolio_equity_curve.iloc[0]['equity']
        assert first_equity == pytest.approx(100000.0)

    def test_no_trades_equity_flat(self):
        """With no signals, equity should remain at initial capital."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data = _make_price_data()
        strategy = PortfolioDeterministicStrategy(buy_bars=set(), sell_bars=set())

        result = engine.run({"AAPL": data}, strategy)
        assert result.total_return == pytest.approx(0.0)
        assert result.total_return_pct == pytest.approx(0.0)

    def test_positions_closed_at_end(self):
        """Positions open at end of data should be closed."""
        reset_trade_counter()
        config = _make_portfolio_config()
        engine = PortfolioEngine(config)

        data = _make_price_data(num_bars=20)
        # Buy at bar 3 but never sell - should be closed at end
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars=set(), stop_loss_price=50.0
        )

        result = engine.run({"AAPL": data}, strategy)
        # Should have at least one trade (force-closed at end)
        total_trades = sum(
            r.num_trades for r in result.symbol_results.values()
        )
        assert total_trades >= 1

    def test_progress_callback(self):
        """Progress callback should be invoked."""
        reset_trade_counter()
        config = _make_portfolio_config()
        engine = PortfolioEngine(config)

        calls = []
        def on_progress(current, total):
            calls.append((current, total))

        data = _make_price_data(num_bars=10)
        strategy = PortfolioDeterministicStrategy(buy_bars=set(), sell_bars=set())
        engine.run({"AAPL": data}, strategy, progress_callback=on_progress)
        assert len(calls) > 0


# =============================================================================
# Capital Contention DEFAULT Mode Tests
# =============================================================================

class TestCapitalContentionDefault:
    """Test DEFAULT mode capital contention."""

    def test_signal_rejection_tracked(self):
        """When capital is exhausted, rejected signals should be recorded."""
        reset_trade_counter()
        # Very small capital - can only afford one position
        config = _make_portfolio_config(initial_capital=5100.0)
        engine = PortfolioEngine(config)

        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0),
            "GOOG": _make_price_data(symbol_offset=0.0),
        }
        # Both want to buy at bar 3 for 100 shares * ~101.5 = ~10150 each
        # Only enough capital for one position (or neither at full size)
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0,
            position_size_shares=100,
        )

        result = engine.run(data_dict, strategy)
        # At least one signal should be rejected or reduced
        total_events = len(result.capital_allocation_events)
        assert total_events > 0

    def test_capital_allocation_events_recorded(self):
        """Capital allocation events should be tracked for executed signals."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data = _make_price_data()
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0
        )

        result = engine.run({"AAPL": data}, strategy)
        # Should have at least one capital allocation event for the buy signal
        assert len(result.capital_allocation_events) >= 1


# =============================================================================
# Capital Contention VULNERABILITY Mode Tests
# =============================================================================

class TestCapitalContentionVulnerability:
    """Test VULNERABILITY_SCORE mode capital contention."""

    def test_vulnerability_mode_initializes_calculator(self):
        """When vulnerability mode is used, calculator should be created."""
        vuln_config = VulnerabilityScoreConfig(min_trade_age_days=5)
        config = _make_portfolio_config(
            mode=CapitalContentionMode.VULNERABILITY_SCORE,
            vuln_config=vuln_config,
        )
        engine = PortfolioEngine(config)
        assert engine.vulnerability_calculator is not None

    def test_default_mode_no_calculator(self):
        """In DEFAULT mode, vulnerability calculator should be None."""
        config = _make_portfolio_config(mode=CapitalContentionMode.DEFAULT)
        engine = PortfolioEngine(config)
        assert engine.vulnerability_calculator is None

    def test_vulnerability_history_tracked(self):
        """Vulnerability scores should be recorded each day when positions are open."""
        reset_trade_counter()
        vuln_config = VulnerabilityScoreConfig(
            min_trade_age_days=2, target_monthly_growth=0.05, alpha=0.0, beta=0.0,
        )
        config = _make_portfolio_config(
            initial_capital=100000.0,
            mode=CapitalContentionMode.VULNERABILITY_SCORE,
            vuln_config=vuln_config,
        )
        engine = PortfolioEngine(config)

        data = _make_price_data(num_bars=20)
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={15}, stop_loss_price=50.0
        )

        result = engine.run({"AAPL": data}, strategy)
        # Should have vulnerability history entries for days when position is open
        assert len(result.vulnerability_history) > 0


# =============================================================================
# Rejection Position Context Tests
# =============================================================================

class TestRejectionPositionContext:
    """
    The rejection_position_tracking CSV needs one row per rejection that pairs
    the rejected signal with the weakest open position at that moment, plus
    that position's eventual outcome. These tests verify the engine populates
    that data correctly in both DEFAULT and end-of-backtest scenarios.
    """

    def test_dataclass_defaults(self):
        ctx = RejectionPositionContext(
            rejection_date=datetime(2024, 1, 5),
            rejected_symbol="AAPL",
            rejected_close_price=101.5,
            rejection_reason="Insufficient capital",
        )
        assert ctx.position_symbol is None
        assert ctx.position_final_pl_pct is None
        assert ctx.position_open_at_end is None

    def _build_contention_setup(self):
        """Two symbols, capital only enough for one full position at a time."""
        reset_trade_counter()
        # Each position needs ~100 * 101.5 = ~10150. Only one can fit.
        config = _make_portfolio_config(initial_capital=10500.0)
        engine = PortfolioEngine(config)
        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0, num_bars=15),
            "GOOG": _make_price_data(symbol_offset=0.0, num_bars=15),
        }
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0,
            position_size_shares=100,
        )
        return engine, data_dict, strategy

    def test_rejection_creates_context_with_open_position(self):
        """A rejection while a position is open should add a context row that
        names the open position as the weakest candidate."""
        engine, data_dict, strategy = self._build_contention_setup()
        result = engine.run(data_dict, strategy)

        assert len(result.rejection_position_contexts) > 0, (
            "Expected at least one rejection context to be recorded"
        )

        ctx = result.rejection_position_contexts[0]
        # The rejected signal's metadata should be present
        assert ctx.rejected_symbol in {"AAPL", "GOOG"}
        assert ctx.rejected_close_price is not None
        # Some other symbol was the open (weakest) position
        assert ctx.position_symbol is not None
        assert ctx.position_symbol != ctx.rejected_symbol
        # P/L % and duration captured at rejection time
        assert ctx.position_duration_days_at_rejection is not None
        assert ctx.position_pl_pct_at_rejection is not None

    def test_context_backfilled_after_position_closes(self):
        """Once the tracked position closes (sell signal at bar 10), the
        context should have its final outcome filled in."""
        engine, data_dict, strategy = self._build_contention_setup()
        result = engine.run(data_dict, strategy)

        # Find a context where the tracked position was named
        named = [c for c in result.rejection_position_contexts
                 if c.position_symbol is not None]
        assert named, "Expected contexts with a named open position"

        # The position will close at bar 10 (sell_bars={10}), so final fields
        # should be populated for all of those contexts.
        for ctx in named:
            assert ctx.position_final_duration_days is not None
            assert ctx.position_final_pl_pct is not None
            assert ctx.position_open_at_end is False, (
                f"Position closed naturally, expected open_at_end=False, "
                f"got {ctx.position_open_at_end}"
            )

    def test_open_at_end_when_position_never_closes(self):
        """If the position is still open at backtest end, the final fields
        should reflect the end-of-backtest close and open_at_end should be True."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=10500.0)
        engine = PortfolioEngine(config)
        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0, num_bars=15),
            "GOOG": _make_price_data(symbol_offset=0.0, num_bars=15),
        }
        # Buy at bar 3 but never sell - position stays open through end
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars=set(), stop_loss_price=50.0,
            position_size_shares=100,
        )
        result = engine.run(data_dict, strategy)

        named = [c for c in result.rejection_position_contexts
                 if c.position_symbol is not None]
        assert named, "Expected at least one rejection context with named position"

        # All tracked positions are force-closed at end of backtest
        for ctx in named:
            assert ctx.position_open_at_end is True
            assert ctx.position_final_duration_days is not None

    def test_default_mode_context_score_is_none(self):
        """In DEFAULT mode there is no vulnerability calculation, so the
        score field should be None."""
        engine, data_dict, strategy = self._build_contention_setup()
        result = engine.run(data_dict, strategy)

        named = [c for c in result.rejection_position_contexts
                 if c.position_symbol is not None]
        assert named
        for ctx in named:
            assert ctx.position_score_at_rejection is None

    def test_vulnerability_mode_context_includes_score(self):
        """In VULNERABILITY_SCORE mode the score field should be populated."""
        reset_trade_counter()
        vuln_config = VulnerabilityScoreConfig(
            min_trade_age_days=1, target_monthly_growth=0.05, alpha=0.0, beta=0.0,
        )
        config = _make_portfolio_config(
            initial_capital=10500.0,
            mode=CapitalContentionMode.VULNERABILITY_SCORE,
            vuln_config=vuln_config,
        )
        engine = PortfolioEngine(config)
        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0, num_bars=15),
            "GOOG": _make_price_data(symbol_offset=0.0, num_bars=15),
        }
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0,
            position_size_shares=100,
        )
        result = engine.run(data_dict, strategy)

        named = [c for c in result.rejection_position_contexts
                 if c.position_symbol is not None]
        assert named, "Expected rejection contexts in vulnerability mode"
        # At least one context should have a non-None score (the position is
        # past min_trade_age_days=1, so a score can be computed).
        scored = [c for c in named if c.position_score_at_rejection is not None]
        assert scored, "Expected at least one context with a vulnerability score"


# =============================================================================
# Full Isolation Mode Tests
# =============================================================================

class EquityRecordingStrategy(PortfolioDeterministicStrategy):
    """Records the total_equity seen each time position_size is called."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entry_equities = []

    def position_size(self, context, signal):
        self.entry_equities.append(context.total_equity)
        return self.position_size_shares


def _make_full_isolation_config(initial_capital=100000.0):
    """Create a PortfolioConfig with full isolation enabled."""
    return PortfolioConfig(
        initial_capital=initial_capital,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
        capital_contention=CapitalContentionConfig.default_mode(),
        full_isolation=True,
    )


class TestFullIsolation:
    """Test full isolation mode (every signal taken, fixed sizing equity)."""

    def test_all_signals_taken_when_capital_exhausted(self):
        """In full isolation, signals are never rejected for lack of capital."""
        reset_trade_counter()
        # Tiny capital that could not normally afford even one full position.
        config = _make_full_isolation_config(initial_capital=100.0)
        engine = PortfolioEngine(config)

        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0),
            "GOOG": _make_price_data(symbol_offset=0.0),
        }
        # Both buy at bar 3 for 100 shares * ~101.5 = ~10150 each (far above 100).
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0,
            position_size_shares=100,
        )

        result = engine.run(data_dict, strategy)

        # No signal should be rejected.
        assert len(result.signal_rejections) == 0
        # Every security should have actually traded.
        for symbol in data_dict:
            assert len(result.symbol_results[symbol].trades) == 1

    def test_default_mode_rejects_for_contrast(self):
        """Sanity check: same scenario rejects signals in default mode."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100.0)
        engine = PortfolioEngine(config)

        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0),
            "GOOG": _make_price_data(symbol_offset=0.0),
        }
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={10}, stop_loss_price=50.0,
            position_size_shares=100,
        )

        result = engine.run(data_dict, strategy)
        # Default mode cannot afford these positions, so signals are rejected.
        assert len(result.signal_rejections) > 0

    def test_sizing_equity_is_constant(self):
        """Position sizing always uses the fixed starting equity (no compounding)."""
        reset_trade_counter()
        config = _make_full_isolation_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        # Two sequential round-trip trades on a single rising security. In a
        # compounding run the second entry would see a larger equity.
        data = _make_price_data(num_bars=40)
        strategy = EquityRecordingStrategy(
            buy_bars={3, 25}, sell_bars={15, 38}, stop_loss_price=50.0,
            position_size_shares=100,
        )

        engine.run({"AAPL": data}, strategy)

        assert len(strategy.entry_equities) >= 2
        for equity in strategy.entry_equities:
            assert abs(equity - 100000.0) < 1e-6

    def test_sizing_equity_compounds_without_full_isolation(self):
        """Contrast: without full isolation the sizing equity changes over time."""
        reset_trade_counter()
        config = _make_portfolio_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data = _make_price_data(num_bars=40)
        strategy = EquityRecordingStrategy(
            buy_bars={3, 25}, sell_bars={15, 38}, stop_loss_price=50.0,
            position_size_shares=100,
        )

        engine.run({"AAPL": data}, strategy)

        assert len(strategy.entry_equities) >= 2
        # The second entry sees a different (compounded) equity.
        assert abs(strategy.entry_equities[-1] - strategy.entry_equities[0]) > 1e-6

    def test_equity_curve_reflects_summed_pl(self):
        """Final equity equals initial capital plus the sum of all trade P/L."""
        reset_trade_counter()
        config = _make_full_isolation_config(initial_capital=100000.0)
        engine = PortfolioEngine(config)

        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0, num_bars=40),
            "GOOG": _make_price_data(symbol_offset=20.0, num_bars=40),
        }
        strategy = PortfolioDeterministicStrategy(
            buy_bars={3}, sell_bars={20}, stop_loss_price=10.0,
            position_size_shares=100,
        )

        result = engine.run(data_dict, strategy)

        all_trades = []
        for sym_result in result.symbol_results.values():
            all_trades.extend(sym_result.trades)
        total_pl = sum(t.pl for t in all_trades)

        assert abs(result.final_equity - (100000.0 + total_pl)) < 1e-3
        assert len(result.signal_rejections) == 0


# =============================================================================
# Short Slippage Direction Tests
# =============================================================================

class ShortPortfolioStrategy(PortfolioDeterministicStrategy):
    """SHORT-only deterministic strategy whose stop sits above the close."""

    def __init__(self, stop_above_pct=0.0005, **kwargs):
        self.stop_above_pct = stop_above_pct
        super().__init__(**kwargs)

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.SHORT

    def generate_entry_signal(self, context: StrategyContext):
        if context.current_index in self.buy_bars:
            stop = context.current_price * (1 + self.stop_above_pct)
            return Signal.buy(
                size=1.0, stop_loss=stop, reason="Test short",
                direction=self.trade_direction,
            )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        return context.current_price * (1 + self.stop_above_pct)


def _make_short_config(full_isolation=False, slippage=0.1):
    """PortfolioConfig with non-zero slippage for short slippage tests."""
    return PortfolioConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=slippage,
        capital_contention=CapitalContentionConfig.default_mode(),
        full_isolation=full_isolation,
    )


class TestShortSlippageDirection:
    """Direction-aware slippage must not push a short entry across its stop."""

    def test_tight_short_stop_does_not_crash(self):
        reset_trade_counter()
        engine = PortfolioEngine(_make_short_config())
        data = _make_price_data(num_bars=20)
        strategy = ShortPortfolioStrategy(
            buy_bars={3}, sell_bars={10}, stop_above_pct=0.0005,
            position_size_shares=100,
        )
        # Previously raised ValueError: "Invalid SHORT stop loss ...".
        result = engine.run({"AAPL": data}, strategy)
        assert len(result.symbol_results["AAPL"].trades) == 1

    def test_full_isolation_short_does_not_crash(self):
        """The exact path the user hit: Full mode forces every short through."""
        reset_trade_counter()
        engine = PortfolioEngine(_make_short_config(full_isolation=True))
        data_dict = {
            "AAPL": _make_price_data(symbol_offset=0.0, num_bars=20),
            "GOOG": _make_price_data(symbol_offset=20.0, num_bars=20),
        }
        strategy = ShortPortfolioStrategy(
            buy_bars={3}, sell_bars={12}, stop_above_pct=0.0005,
            position_size_shares=100,
        )
        result = engine.run(data_dict, strategy)
        assert len(result.signal_rejections) == 0
        for sym in data_dict:
            assert len(result.symbol_results[sym].trades) == 1

    def test_short_entry_fills_below_close(self):
        reset_trade_counter()
        engine = PortfolioEngine(_make_short_config())
        data = _make_price_data(num_bars=20)
        close_at_entry = data.iloc[3]['close']
        strategy = ShortPortfolioStrategy(
            buy_bars={3}, sell_bars={10}, stop_above_pct=0.05,
            position_size_shares=100,
        )
        result = engine.run({"AAPL": data}, strategy)
        trades = result.symbol_results["AAPL"].trades
        assert len(trades) == 1
        assert trades[0].entry_price < close_at_entry
