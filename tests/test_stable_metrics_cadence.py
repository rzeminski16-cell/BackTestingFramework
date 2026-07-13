"""
Cadence-invariance tests for the stable metrics (RAR% family).

BARS_PER_YEAR = 365 assumes one equity row per calendar day. Callers feed
curves at different cadences — trading-day bars from backtests, one point per
trade from the Rule Tester — so ``calculate_all`` must resample to a
calendar-daily grid before regressing. These tests pin that contract:
a curve with a known true annual growth rate must report (approximately) the
same RAR% no matter how it is sampled.

Regression coverage for the annualisation fix: previously a trading-day curve
(~252 rows/year) reported RAR% overstated by ~45% in log space, and a
per-trade curve was annualised as if the strategy traded 365 times per year.
"""
import numpy as np
import pandas as pd
import pytest

from Classes.Core.stable_metrics import StableMetricsCalculator
from Classes.Modelling.adjusted_rar import (
    AdjustedRARConfig, adjusted_rar_from_equity,
)

TRUE_ANNUAL_GROWTH = 0.10  # 10% per year
INITIAL = 100_000.0


def _calendar_curve(years=3):
    """Perfect exponential growth sampled every calendar day."""
    dates = pd.date_range("2020-01-01", periods=int(365 * years), freq="D")
    t = np.arange(len(dates))
    equity = INITIAL * (1.0 + TRUE_ANNUAL_GROWTH) ** (t / 365.0)
    return pd.DataFrame({"date": dates, "equity": equity})


def _trading_day_curve(years=3):
    """The same growth path sampled on business days only (~252 rows/year)."""
    full = _calendar_curve(years)
    mask = full["date"].dt.dayofweek < 5
    return full[mask].reset_index(drop=True)


def _per_trade_curve(years=3, every_n_days=9):
    """The same growth path sampled sparsely, like one point per trade exit."""
    full = _calendar_curve(years)
    return full.iloc[::every_n_days].reset_index(drop=True)


class TestRARCadenceInvariance:
    def test_calendar_daily_matches_true_growth(self):
        res = StableMetricsCalculator.calculate_all(_calendar_curve())
        assert res.rar_pct == pytest.approx(TRUE_ANNUAL_GROWTH * 100, abs=0.2)
        assert res.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_trading_day_curve_matches_true_growth(self):
        """Trading-day cadence must not inflate RAR% (was ~14.8% pre-fix)."""
        res = StableMetricsCalculator.calculate_all(_trading_day_curve())
        assert res.rar_pct == pytest.approx(TRUE_ANNUAL_GROWTH * 100, abs=0.2)

    def test_per_trade_curve_matches_true_growth(self):
        """Per-trade cadence must not annualise as 365 trades/year."""
        res = StableMetricsCalculator.calculate_all(_per_trade_curve())
        assert res.rar_pct == pytest.approx(TRUE_ANNUAL_GROWTH * 100, abs=0.2)

    def test_all_cadences_agree_with_each_other(self):
        values = [
            StableMetricsCalculator.calculate_all(curve).rar_pct
            for curve in (_calendar_curve(), _trading_day_curve(), _per_trade_curve())
        ]
        assert max(values) - min(values) < 0.3, (
            f"RAR% must be cadence-invariant, got {values}"
        )

    def test_consistent_with_modelling_adjusted_rar(self):
        """The backtest-side metric and the Modelling-stage metric must agree
        on the same underlying curve (both calendar-daily, 365 bars/year)."""
        curve = _trading_day_curve()
        stable = StableMetricsCalculator.calculate_all(curve)
        # The modelling helper expects an already calendar-daily curve; give it
        # the exact daily path.
        modelling = adjusted_rar_from_equity(
            _calendar_curve(),
            AdjustedRARConfig(bars_per_year=365, weight_by_r_squared=True,
                              clip_min=None, clip_max=None),
        )
        assert stable.rar_adjusted == pytest.approx(modelling, abs=0.3)

    def test_duplicate_dates_collapse_to_last(self):
        """Several trade exits on one day must not distort the regression."""
        curve = _per_trade_curve()
        dup = curve.iloc[[10]].copy()
        dup["equity"] = curve["equity"].iloc[10] * 0.5  # stale intraday point
        noisy = pd.concat([curve.iloc[:10], dup, curve.iloc[10:]]).reset_index(drop=True)
        res = StableMetricsCalculator.calculate_all(noisy)
        assert res.rar_pct == pytest.approx(TRUE_ANNUAL_GROWTH * 100, abs=0.5)

    def test_no_date_column_falls_back_to_row_cadence(self):
        """Without dates the previous behaviour is preserved (rows = days)."""
        curve = _calendar_curve().drop(columns=["date"])
        res = StableMetricsCalculator.calculate_all(curve)
        assert res.rar_pct == pytest.approx(TRUE_ANNUAL_GROWTH * 100, abs=0.2)


class TestDownstreamMetricsOnDailyGrid:
    def test_drawdown_durations_in_calendar_days(self):
        """A 30-row dip on a trading-day curve spans ~42 calendar days; the
        resampled curve should report the calendar duration either way."""
        curve = _trading_day_curve(years=2)
        equity = curve["equity"].values.copy()
        equity[200:230] *= 0.90  # 30 trading days ≈ 42 calendar days
        dipped = pd.DataFrame({"date": curve["date"], "equity": equity})
        res = StableMetricsCalculator.calculate_all(dipped)
        assert res.largest_drawdowns, "expected at least one drawdown"
        top = res.largest_drawdowns[0]
        assert 38 <= top.duration_days <= 48

    def test_robust_sharpe_positive_for_growth_curve(self):
        res = StableMetricsCalculator.calculate_all(_trading_day_curve())
        assert res.robust_sharpe_ratio > 0
