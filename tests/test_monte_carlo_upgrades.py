"""
Tests for the P2 Monte Carlo upgrades: per-path annualized Sharpe/CAGR/
Calmar distributions (via SimulationConfig.periods_per_year) and the
daily-returns pool loader.
"""
import numpy as np
import pandas as pd
import pytest

from Classes.MonteCarloAnalysis.analyzer import SimulationAnalyzer
from Classes.MonteCarloAnalysis.config import SimulationConfig
from Classes.MonteCarloAnalysis.simulator import MonteCarloSimulator
from Classes.MonteCarloAnalysis.trade_log_loader import load_daily_returns


def _run(periods_per_year=None, num_trades=120, seed=7):
    cfg = SimulationConfig(
        num_simulations=400,
        num_trades=num_trades,
        initial_capital=100_000.0,
        risk_per_trade=1.0,
        random_seed=seed,
        periods_per_year=periods_per_year,
    )
    pool = np.random.default_rng(0).normal(0.0008, 0.01, 800)
    return SimulationAnalyzer(MonteCarloSimulator(cfg).run(pool))


class TestAnnualizedDistributions:
    def test_disabled_without_periods_per_year(self):
        analyzer = _run(periods_per_year=None)
        m = analyzer.metrics()
        assert m.median_annualized_sharpe is None
        assert m.median_calmar is None
        assert analyzer.cagr_distribution() is None
        assert analyzer.calmar_distribution() is None

    def test_enabled_with_periods_per_year(self):
        analyzer = _run(periods_per_year=252)
        m = analyzer.metrics()
        assert m.median_annualized_sharpe is not None
        assert m.p5_annualized_sharpe is not None
        assert m.median_annualized_cagr is not None
        assert m.median_calmar is not None
        # A +0.08%/day drift pool should annualize clearly positive.
        assert m.median_annualized_sharpe > 0
        assert m.p5_annualized_sharpe < m.median_annualized_sharpe

    def test_sharpe_annualization_scale(self):
        analyzer = _run(periods_per_year=252)
        raw = analyzer.sharpe_distribution(annualized=False)
        ann = analyzer.sharpe_distribution(annualized=True)
        np.testing.assert_allclose(ann, raw * np.sqrt(252), rtol=1e-12)

    def test_distribution_shapes(self):
        analyzer = _run(periods_per_year=252)
        n_sim = analyzer.equity_curves.shape[0]
        assert analyzer.sharpe_distribution().shape == (n_sim,)
        assert analyzer.cagr_distribution().shape == (n_sim,)
        assert analyzer.calmar_distribution().shape == (n_sim,)

    def test_cagr_known_growth(self):
        """Deterministic pool: +0.1% per step, 252 steps = one year."""
        cfg = SimulationConfig(num_simulations=10, num_trades=252,
                               initial_capital=100_000.0, risk_per_trade=1.0,
                               random_seed=1, periods_per_year=252)
        pool = np.full(50, 0.001)
        analyzer = SimulationAnalyzer(MonteCarloSimulator(cfg).run(pool))
        cagr = analyzer.cagr_distribution()
        expected = 1.001 ** 252 - 1
        np.testing.assert_allclose(cagr, expected, rtol=1e-9)

    def test_config_validation(self):
        assert SimulationConfig(periods_per_year=-5).validate()
        assert not SimulationConfig(periods_per_year=252).validate()

    def test_config_roundtrip(self):
        cfg = SimulationConfig(periods_per_year=52.0)
        assert SimulationConfig.from_dict(cfg.to_dict()).periods_per_year == 52.0


class TestDailyReturnsLoader:
    def _write(self, tmp_path, df, name="curve.csv"):
        p = tmp_path / name
        df.to_csv(p, index=False)
        return p

    def test_loads_equity_curve(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        equity = [100.0, 101.0, 100.0, 102.0, 103.02]
        p = self._write(tmp_path, pd.DataFrame({"date": dates, "equity": equity}))
        returns, warnings = load_daily_returns(p)
        assert len(returns) == 4
        assert returns[0] == pytest.approx(0.01)
        assert returns[1] == pytest.approx(-1.0 / 101.0)
        assert warnings == []

    def test_loads_close_series(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        p = self._write(tmp_path, pd.DataFrame(
            {"date": dates, "close": [50.0, 51.0, 49.0, 50.0]}))
        returns, _ = load_daily_returns(p)
        assert len(returns) == 3

    def test_sorts_by_date(self, tmp_path):
        # Rows shuffled on disk; returns must follow chronological order.
        df = pd.DataFrame({
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "equity": [104.0, 100.0, 102.0],
        })
        p = self._write(tmp_path, df)
        returns, _ = load_daily_returns(p)
        assert returns == pytest.approx([0.02, 104.0 / 102.0 - 1.0])

    def test_missing_value_column_warns(self, tmp_path):
        p = self._write(tmp_path, pd.DataFrame({"date": ["2024-01-01"], "foo": [1]}))
        returns, warnings = load_daily_returns(p)
        assert len(returns) == 0
        assert any("no usable value column" in w for w in warnings)

    def test_explicit_value_column(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        p = self._write(tmp_path, pd.DataFrame(
            {"date": dates, "nav": [100.0, 110.0, 121.0]}))
        returns, _ = load_daily_returns(p, value_column="nav")
        assert returns == pytest.approx([0.1, 0.1])

    def test_pool_feeds_simulator(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        rng = np.random.default_rng(5)
        equity = 100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
        p = self._write(tmp_path, pd.DataFrame({"date": dates, "equity": equity}))
        pool, _ = load_daily_returns(p)

        cfg = SimulationConfig(num_simulations=100, num_trades=252,
                               initial_capital=100_000.0, risk_per_trade=1.0,
                               random_seed=2, periods_per_year=252)
        analyzer = SimulationAnalyzer(MonteCarloSimulator(cfg).run(pool))
        m = analyzer.metrics()
        assert m.median_annualized_sharpe is not None
        assert m.num_trades == 252
