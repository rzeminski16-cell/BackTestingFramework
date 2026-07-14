"""
Cross-layer metric consistency ("golden") tests.

The framework has one source of truth for metric formulas —
``Classes.Core.performance_metrics.CentralizedPerformanceMetrics`` — and a
report-facing adapter, ``Classes.Analysis.performance_metrics
.PerformanceMetrics``, whose output dict has a documented unit contract
(win_rate as a fraction, dollar-denominated win/loss aggregates). These tests
pin both: the adapter must agree with the centralized module after unit
conversion, and hand-computed golden values on a small fixture must match
both layers. If a formula is ever re-implemented locally and drifts, this
file fails.
"""
import numpy as np
import pandas as pd
import pytest

from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Core.performance_metrics import CentralizedPerformanceMetrics
from Classes.Engine.backtest_result import BacktestResult
from Classes.Models.trade import Trade


def _trade(entry, exit_, qty, entry_date, exit_date, stop=None):
    pl = (exit_ - entry) * qty
    t = Trade(
        symbol="TST",
        entry_date=pd.Timestamp(entry_date).to_pydatetime(),
        entry_price=entry,
        exit_date=pd.Timestamp(exit_date).to_pydatetime(),
        exit_price=exit_,
        quantity=qty,
        side="LONG",
        initial_stop_loss=stop,
        pl=pl,
        pl_pct=(exit_ - entry) / entry * 100,
        duration_days=(pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days,
    )
    return t


@pytest.fixture
def fixture_trades():
    return [
        _trade(100.0, 110.0, 10, "2024-01-02", "2024-01-12", stop=95.0),   # +100, +2R
        _trade(50.0, 45.0, 20, "2024-02-01", "2024-02-06", stop=48.0),     # -100, -2.5R
        _trade(200.0, 210.0, 5, "2024-03-01", "2024-03-21", stop=190.0),   # +50, +1R
        _trade(80.0, 76.0, 10, "2024-04-01", "2024-04-08", stop=76.0),     # -40, -1R
    ]


@pytest.fixture
def fixture_equity():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0006, 0.008, len(dates))
    equity = 100_000.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"date": dates, "equity": equity})


@pytest.fixture
def fixture_result(fixture_trades, fixture_equity):
    final = float(fixture_equity["equity"].iloc[-1])
    return BacktestResult(
        symbol="TST", strategy_name="Fixture",
        trades=fixture_trades, equity_curve=fixture_equity,
        final_equity=final, total_return=final - 100_000.0,
        total_return_pct=(final / 100_000.0 - 1.0) * 100.0,
    )


class TestGoldenValues:
    """Hand-computed values on the 4-trade fixture."""

    def test_win_rate(self, fixture_trades):
        # 2 winners of 4 trades
        assert CentralizedPerformanceMetrics.calculate_win_rate(fixture_trades) \
            == pytest.approx(50.0)

    def test_profit_factor(self, fixture_trades):
        # gross profit 150, gross loss 140
        assert CentralizedPerformanceMetrics.calculate_profit_factor(fixture_trades) \
            == pytest.approx(150.0 / 140.0)

    def test_r_multiples(self, fixture_trades):
        rs = CentralizedPerformanceMetrics.calculate_r_multiples(fixture_trades)
        assert rs == pytest.approx([2.0, -2.5, 1.0, -1.0])

    def test_avg_trade_duration(self, fixture_trades):
        assert CentralizedPerformanceMetrics.calculate_avg_trade_duration(fixture_trades) \
            == pytest.approx((10 + 5 + 20 + 7) / 4)

    def test_max_drawdown_simple(self):
        curve = pd.DataFrame({"equity": [100.0, 120.0, 90.0, 110.0, 130.0]})
        dd, dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(curve)
        assert dd == pytest.approx(30.0)
        assert dd_pct == pytest.approx(25.0)

    def test_cagr_two_years_exact(self):
        dates = pd.to_datetime(["2020-01-01", "2022-01-01"])
        curve = pd.DataFrame({"date": dates, "equity": [100_000.0, 121_000.0]})
        cagr = CentralizedPerformanceMetrics.calculate_annual_return(curve)
        # (1.21)^(1/2) - 1 = 10% (within calendar-day rounding of years)
        assert cagr == pytest.approx(10.0, abs=0.05)


class TestAdapterAgreesWithCore:
    """PerformanceMetrics.calculate_metrics must match the centralized module
    after the documented unit conversions."""

    def test_win_rate_fraction_vs_percent(self, fixture_result, fixture_trades):
        adapter = PerformanceMetrics.calculate_metrics(fixture_result)
        core = CentralizedPerformanceMetrics.calculate_win_rate(fixture_trades)
        assert adapter["win_rate"] == pytest.approx(core / 100.0)
        assert 0.0 <= adapter["win_rate"] <= 1.0, "documented contract: fraction"

    def test_equity_metrics_identical(self, fixture_result, fixture_equity):
        adapter = PerformanceMetrics.calculate_metrics(fixture_result)
        assert adapter["sharpe_ratio"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_sharpe_ratio(fixture_equity))
        assert adapter["sortino_ratio"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_sortino_ratio(fixture_equity))
        dd, dd_pct = CentralizedPerformanceMetrics.calculate_max_drawdown(fixture_equity)
        assert adapter["max_drawdown"] == pytest.approx(dd)
        assert adapter["max_drawdown_pct"] == pytest.approx(dd_pct)

    def test_trade_metrics_identical(self, fixture_result, fixture_trades):
        adapter = PerformanceMetrics.calculate_metrics(fixture_result)
        assert adapter["profit_factor"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_profit_factor(fixture_trades))
        assert adapter["avg_trade_duration"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_avg_trade_duration(fixture_trades))
        assert adapter["avg_r_multiple"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_avg_r_multiple(fixture_trades))
        assert adapter["r_expectancy"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_r_expectancy(fixture_trades))

    def test_dollar_aggregates_are_dollars(self, fixture_result):
        """avg_win/avg_loss/largest_* are base-currency dollars (documented)."""
        adapter = PerformanceMetrics.calculate_metrics(fixture_result)
        assert adapter["avg_win"] == pytest.approx((100.0 + 50.0) / 2)
        assert adapter["avg_loss"] == pytest.approx((-100.0 - 40.0) / 2)
        assert adapter["largest_win"] == pytest.approx(100.0)
        assert adapter["largest_loss"] == pytest.approx(-100.0)

    def test_calmar_is_cagr_over_max_dd(self, fixture_result, fixture_equity):
        adapter = PerformanceMetrics.calculate_metrics(fixture_result)
        cagr = CentralizedPerformanceMetrics.calculate_annual_return(fixture_equity)
        if adapter["max_drawdown_pct"] > 0:
            assert adapter["calmar_ratio"] == pytest.approx(
                cagr / adapter["max_drawdown_pct"])


class TestFromTradesUnitContract:
    def test_win_rate_is_percent_here(self, fixture_trades):
        m = PerformanceMetrics.calculate_from_trades(fixture_trades, 100_000.0)
        assert m["win_rate"] == pytest.approx(50.0), \
            "documented contract: calculate_from_trades returns percent"

    def test_profit_factor_matches_core(self, fixture_trades):
        m = PerformanceMetrics.calculate_from_trades(fixture_trades, 100_000.0)
        assert m["profit_factor"] == pytest.approx(
            CentralizedPerformanceMetrics.calculate_profit_factor(fixture_trades))
