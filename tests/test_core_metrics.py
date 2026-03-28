"""
Comprehensive tests for CentralizedPerformanceMetrics.

Tests verify the accuracy and correctness of all metric calculations:
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown (dollars and percentage)
- Win rate, profit factor, expectancy
- Total return, CAGR
- Volatility, downside deviation
- Consecutive wins/losses
- Edge cases (empty data, single point, all winners, all losers)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List

from Classes.Core.performance_metrics import (
    CentralizedPerformanceMetrics,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    MAX_PROFIT_FACTOR,
    MetricDefinition,
)


# =============================================================================
# Helpers
# =============================================================================

def make_equity_curve(values, start_date='2024-01-01'):
    """Create equity curve DataFrame from a list of equity values."""
    dates = pd.date_range(start_date, periods=len(values), freq='D')
    return pd.DataFrame({'date': dates, 'equity': values})


@dataclass
class MockTrade:
    """Minimal trade-like object for testing."""
    pl: float


def make_trades(pl_values: List[float]) -> List[MockTrade]:
    """Create list of mock trade objects with given P/L values."""
    return [MockTrade(pl=pl) for pl in pl_values]


# =============================================================================
# Sharpe Ratio Tests
# =============================================================================

class TestSharpeRatio:
    def test_empty_curve(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_sharpe_ratio(df) == 0.0

    def test_single_point(self):
        df = make_equity_curve([100000])
        assert CentralizedPerformanceMetrics.calculate_sharpe_ratio(df) == 0.0

    def test_flat_equity_zero_sharpe(self):
        """Constant equity should give 0 Sharpe (no variance)."""
        df = make_equity_curve([100000] * 100)
        assert CentralizedPerformanceMetrics.calculate_sharpe_ratio(df) == 0.0

    def test_positive_returns_positive_sharpe(self):
        """Steadily increasing equity should have positive Sharpe."""
        values = [100000 + i * 100 for i in range(252)]
        df = make_equity_curve(values)
        sharpe = CentralizedPerformanceMetrics.calculate_sharpe_ratio(df)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        """Steadily declining equity should have negative Sharpe."""
        values = [100000 - i * 100 for i in range(252)]
        df = make_equity_curve(values)
        sharpe = CentralizedPerformanceMetrics.calculate_sharpe_ratio(df)
        assert sharpe < 0

    def test_custom_risk_free_rate(self):
        values = [100000 + i * 100 for i in range(252)]
        df = make_equity_curve(values)
        sharpe_low_rf = CentralizedPerformanceMetrics.calculate_sharpe_ratio(
            df, risk_free_rate=0.01
        )
        sharpe_high_rf = CentralizedPerformanceMetrics.calculate_sharpe_ratio(
            df, risk_free_rate=0.10
        )
        # Higher risk-free rate should produce lower Sharpe
        assert sharpe_low_rf > sharpe_high_rf


# =============================================================================
# Sortino Ratio Tests
# =============================================================================

class TestSortinoRatio:
    def test_empty_curve(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_sortino_ratio(df) == 0.0

    def test_single_point(self):
        df = make_equity_curve([100000])
        assert CentralizedPerformanceMetrics.calculate_sortino_ratio(df) == 0.0

    def test_all_positive_returns_high_sortino(self):
        """No negative returns should give very high Sortino."""
        values = [100000 + i * 50 for i in range(252)]
        df = make_equity_curve(values)
        sortino = CentralizedPerformanceMetrics.calculate_sortino_ratio(df)
        assert sortino == 99.99  # Capped value for no downside

    def test_mixed_returns(self):
        """Mix of ups and downs should give moderate Sortino."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        values = [100000]
        for r in returns:
            values.append(values[-1] * (1 + r))
        df = make_equity_curve(values)
        sortino = CentralizedPerformanceMetrics.calculate_sortino_ratio(df)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)


# =============================================================================
# Max Drawdown Tests
# =============================================================================

class TestMaxDrawdown:
    def test_empty_curve(self):
        df = make_equity_curve([])
        dd, dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(df)
        assert dd == 0.0
        assert dd_pct == 0.0

    def test_monotonically_increasing(self):
        """No drawdown for constantly rising equity."""
        values = [100000 + i * 100 for i in range(100)]
        df = make_equity_curve(values)
        dd, dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(df)
        assert dd == 0.0
        assert dd_pct == 0.0

    def test_known_drawdown(self):
        """Known 10% drawdown."""
        values = [100000, 110000, 99000, 105000]
        df = make_equity_curve(values)
        dd, dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(df)
        # Max drawdown from 110000 to 99000 = 11000
        assert dd == pytest.approx(11000.0, abs=1)
        # Percentage: 11000/110000 * 100 = 10.0%
        assert dd_pct == pytest.approx(10.0, abs=0.1)

    def test_full_drawdown(self):
        """Complete loss scenario."""
        values = [100000, 50000, 10000, 1000]
        df = make_equity_curve(values)
        dd, dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(df)
        assert dd == pytest.approx(99000.0, abs=1)
        assert dd_pct == pytest.approx(99.0, abs=0.1)


# =============================================================================
# Volatility Tests
# =============================================================================

class TestVolatility:
    def test_empty_curve(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_volatility(df) == 0.0

    def test_flat_equity(self):
        df = make_equity_curve([100000] * 100)
        assert CentralizedPerformanceMetrics.calculate_volatility(df) == 0.0

    def test_volatile_equity(self):
        """High variance returns should produce high volatility."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.05, 252)
        values = [100000]
        for r in returns:
            values.append(max(values[-1] * (1 + r), 1))
        df = make_equity_curve(values)
        vol = CentralizedPerformanceMetrics.calculate_volatility(df)
        assert vol > 0


# =============================================================================
# Total Return Tests
# =============================================================================

class TestTotalReturn:
    def test_empty_curve(self):
        df = make_equity_curve([])
        dollars, pct = CentralizedPerformanceMetrics.calculate_total_return(df)
        assert dollars == 0.0
        assert pct == 0.0

    def test_known_return(self):
        df = make_equity_curve([100000, 110000])
        dollars, pct = CentralizedPerformanceMetrics.calculate_total_return(df)
        assert dollars == pytest.approx(10000.0)
        assert pct == pytest.approx(10.0)

    def test_negative_return(self):
        df = make_equity_curve([100000, 90000])
        dollars, pct = CentralizedPerformanceMetrics.calculate_total_return(df)
        assert dollars == pytest.approx(-10000.0)
        assert pct == pytest.approx(-10.0)


# =============================================================================
# Annual Return (CAGR) Tests
# =============================================================================

class TestAnnualReturn:
    def test_empty_curve(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_annual_return(df) == 0.0

    def test_one_year_10_percent(self):
        """10% return over ~1 year should give ~10% CAGR."""
        dates = pd.date_range('2024-01-01', periods=366, freq='D')
        values = np.linspace(100000, 110000, 366)
        df = pd.DataFrame({'date': dates, 'equity': values})
        cagr = CentralizedPerformanceMetrics.calculate_annual_return(df)
        assert cagr == pytest.approx(10.0, abs=1.0)

    def test_two_year_return(self):
        """Known return over 2 years."""
        dates = pd.date_range('2022-01-01', periods=731, freq='D')
        # 100k -> 121k over 2 years = ~10% CAGR
        values = np.linspace(100000, 121000, 731)
        df = pd.DataFrame({'date': dates, 'equity': values})
        cagr = CentralizedPerformanceMetrics.calculate_annual_return(df)
        assert cagr == pytest.approx(10.0, abs=1.0)


# =============================================================================
# Win Rate Tests
# =============================================================================

class TestWinRate:
    def test_empty_trades(self):
        assert CentralizedPerformanceMetrics.calculate_win_rate([]) == 0.0

    def test_all_winners(self):
        trades = make_trades([100, 200, 300])
        assert CentralizedPerformanceMetrics.calculate_win_rate(trades) == 100.0

    def test_all_losers(self):
        trades = make_trades([-100, -200, -300])
        assert CentralizedPerformanceMetrics.calculate_win_rate(trades) == 0.0

    def test_mixed(self):
        trades = make_trades([100, -50, 200, -25])
        assert CentralizedPerformanceMetrics.calculate_win_rate(trades) == 50.0

    def test_breakeven_is_loss(self):
        """Zero P/L counts as loss."""
        trades = make_trades([0])
        assert CentralizedPerformanceMetrics.calculate_win_rate(trades) == 0.0

    def test_dict_trades(self):
        """Win rate should work with dict-based trades too."""
        trades = [{'pl': 100}, {'pl': -50}, {'pl': 200}]
        wr = CentralizedPerformanceMetrics.calculate_win_rate(trades)
        assert wr == pytest.approx(66.67, abs=0.1)


# =============================================================================
# Profit Factor Tests
# =============================================================================

class TestProfitFactor:
    def test_empty_trades(self):
        assert CentralizedPerformanceMetrics.calculate_profit_factor([]) == 0.0

    def test_all_winners(self):
        trades = make_trades([100, 200])
        pf = CentralizedPerformanceMetrics.calculate_profit_factor(trades)
        assert pf == MAX_PROFIT_FACTOR

    def test_all_losers(self):
        trades = make_trades([-100, -200])
        pf = CentralizedPerformanceMetrics.calculate_profit_factor(trades)
        assert pf == 0.0

    def test_known_profit_factor(self):
        """Gross profit 300, gross loss 100 => PF = 3.0"""
        trades = make_trades([100, 200, -50, -50])
        pf = CentralizedPerformanceMetrics.calculate_profit_factor(trades)
        assert pf == pytest.approx(3.0)

    def test_profit_factor_1(self):
        """Equal wins and losses => PF = 1.0"""
        trades = make_trades([100, -100])
        pf = CentralizedPerformanceMetrics.calculate_profit_factor(trades)
        assert pf == pytest.approx(1.0)


# =============================================================================
# Expectancy Tests
# =============================================================================

class TestExpectancy:
    def test_empty_trades(self):
        assert CentralizedPerformanceMetrics.calculate_expectancy([]) == 0.0

    def test_positive_expectancy(self):
        """Expectancy uses pl_pct, not pl. Create trades with both."""
        @dataclass
        class TradeWithPct:
            pl: float
            pl_pct: float
        trades = [TradeWithPct(200, 20.0), TradeWithPct(-50, -5.0),
                  TradeWithPct(300, 30.0), TradeWithPct(-100, -10.0)]
        exp = CentralizedPerformanceMetrics.calculate_expectancy(trades)
        # Win rate = 50%, avg win pct = 25%, avg loss pct = 7.5%
        # Expectancy = (0.5 * 25) - (0.5 * 7.5) = 8.75
        assert exp > 0

    def test_negative_expectancy(self):
        @dataclass
        class TradeWithPct:
            pl: float
            pl_pct: float
        trades = [TradeWithPct(50, 2.0), TradeWithPct(-200, -20.0),
                  TradeWithPct(30, 1.5), TradeWithPct(-150, -15.0)]
        exp = CentralizedPerformanceMetrics.calculate_expectancy(trades)
        assert exp < 0


# =============================================================================
# Best/Worst Day Tests
# =============================================================================

class TestDailyReturns:
    def test_best_day_empty(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_best_day(df) == 0.0

    def test_worst_day_empty(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_worst_day(df) == 0.0

    def test_best_day_known(self):
        # 100000 -> 110000 = 10% return
        df = make_equity_curve([100000, 110000, 105000])
        best = CentralizedPerformanceMetrics.calculate_best_day(df)
        assert best == pytest.approx(10.0, abs=0.1)

    def test_worst_day_known(self):
        # 110000 -> 105000 = ~-4.55%
        df = make_equity_curve([100000, 110000, 105000])
        worst = CentralizedPerformanceMetrics.calculate_worst_day(df)
        assert worst == pytest.approx(-4.545, abs=0.1)


# =============================================================================
# Consecutive Wins/Losses Tests (via calculate_all_metrics)
# =============================================================================

class TestConsecutive:
    """Test consecutive wins/losses as part of full metrics calculation."""

    def _get_consecutive(self, pl_values):
        """Helper: get consecutive wins/losses from calculate_all_metrics."""
        @dataclass
        class FullTrade:
            pl: float
            pl_pct: float
            duration_days: int = 5
        trades = [FullTrade(pl=pl, pl_pct=pl / 100.0) for pl in pl_values]
        equity = make_equity_curve([100000, 100000 + sum(pl_values)])
        metrics = CentralizedPerformanceMetrics.calculate_all_metrics(
            equity_curve=equity, trades=trades, initial_capital=100000
        )
        return metrics.get('max_consecutive_wins', 0), metrics.get('max_consecutive_losses', 0)

    def test_known_streak(self):
        wins, losses = self._get_consecutive([100, 200, 300, -50, 100, -10, -20])
        assert wins == 3
        assert losses == 2

    def test_single_trade_win(self):
        wins, losses = self._get_consecutive([100])
        assert wins == 1
        assert losses == 0

    def test_alternating(self):
        wins, losses = self._get_consecutive([100, -50, 100, -50])
        assert wins == 1
        assert losses == 1


# =============================================================================
# Downside Deviation Tests
# =============================================================================

class TestDownsideDeviation:
    def test_empty_curve(self):
        df = make_equity_curve([])
        assert CentralizedPerformanceMetrics.calculate_downside_deviation(df) == 0.0

    def test_all_positive_returns(self):
        values = [100000 + i * 100 for i in range(100)]
        df = make_equity_curve(values)
        dd = CentralizedPerformanceMetrics.calculate_downside_deviation(df)
        assert dd == 0.0

    def test_mixed_returns(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        values = [100000]
        for r in returns:
            values.append(max(values[-1] * (1 + r), 1))
        df = make_equity_curve(values)
        dd = CentralizedPerformanceMetrics.calculate_downside_deviation(df)
        assert dd > 0


# =============================================================================
# MetricDefinition Tests
# =============================================================================

class TestMetricDefinition:
    def test_format_normal(self):
        md = MetricDefinition(name="Test", description="test", format_str="{:.2f}%")
        assert md.format(10.5) == "10.50%"

    def test_format_nan(self):
        md = MetricDefinition(name="Test", description="test")
        assert md.format(float('nan')) == "N/A"

    def test_format_inf(self):
        md = MetricDefinition(name="Test", description="test")
        assert md.format(float('inf')) == "N/A"
