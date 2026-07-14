"""
Tests for the Probabilistic / Deflated Sharpe Ratio module and its
integration into the walk-forward optimizer's WindowResult.
"""
import math

import numpy as np
import pytest

from Classes.Core.deflated_sharpe import (
    deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio,
)


class TestProbabilisticSharpe:
    def test_zero_sharpe_is_coin_flip(self):
        assert probabilistic_sharpe_ratio(0.0, 0.0, n_obs=252) == pytest.approx(0.5)

    def test_strong_sharpe_long_sample_near_one(self):
        # Annual SR 2.0 over 4 years of daily data.
        sr_daily = 2.0 / math.sqrt(252)
        psr = probabilistic_sharpe_ratio(sr_daily, 0.0, n_obs=1008)
        assert psr > 0.99

    def test_more_observations_more_confidence(self):
        sr_daily = 1.0 / math.sqrt(252)
        psr_short = probabilistic_sharpe_ratio(sr_daily, 0.0, n_obs=126)
        psr_long = probabilistic_sharpe_ratio(sr_daily, 0.0, n_obs=1260)
        assert psr_long > psr_short

    def test_negative_skew_reduces_confidence(self):
        sr_daily = 1.5 / math.sqrt(252)
        base = probabilistic_sharpe_ratio(sr_daily, 0.0, n_obs=504,
                                          skew=0.0, kurtosis=3.0)
        skewed = probabilistic_sharpe_ratio(sr_daily, 0.0, n_obs=504,
                                            skew=-1.5, kurtosis=6.0)
        assert skewed < base

    def test_insufficient_observations(self):
        assert probabilistic_sharpe_ratio(1.0, 0.0, n_obs=1) == 0.0


class TestExpectedMaxSharpe:
    def test_grows_with_trials(self):
        v = 0.01
        e10 = expected_max_sharpe(10, v)
        e100 = expected_max_sharpe(100, v)
        e1000 = expected_max_sharpe(1000, v)
        assert 0 < e10 < e100 < e1000

    def test_no_selection_pressure_edge_cases(self):
        assert expected_max_sharpe(1, 0.5) == 0.0
        assert expected_max_sharpe(50, 0.0) == 0.0


class TestDeflatedSharpe:
    def test_lucky_noise_is_deflated(self):
        """The best of many noise strategies must NOT look significant."""
        rng = np.random.default_rng(0)
        n_obs = 252
        # 200 skill-less strategies: daily SR estimates ~ N(0, 1/sqrt(n)).
        trial_daily = rng.normal(0.0, 1.0 / math.sqrt(n_obs), 200)
        trials_annual = trial_daily * math.sqrt(252)
        winner_annual = float(trials_annual.max())

        out = deflated_sharpe_ratio(winner_annual, trials_annual, n_obs=n_obs)
        # Without deflation the winner looks great...
        assert out["psr_vs_zero"] > 0.9
        # ...but the DSR sees through the selection.
        assert out["dsr"] < 0.75

    def test_genuine_skill_survives_deflation(self):
        rng = np.random.default_rng(1)
        n_obs = 1260  # 5 years daily
        trial_daily = rng.normal(0.0, 1.0 / math.sqrt(n_obs), 50)
        trials_annual = list(trial_daily * math.sqrt(252)) + [2.5]
        out = deflated_sharpe_ratio(2.5, trials_annual, n_obs=n_obs)
        assert out["dsr"] > 0.95

    def test_more_trials_lower_dsr(self):
        n_obs = 504
        rng = np.random.default_rng(2)
        small = list(rng.normal(0, 0.4, 5))
        large = list(rng.normal(0, 0.4, 500))
        d_small = deflated_sharpe_ratio(1.2, small + [1.2], n_obs)["dsr"]
        d_large = deflated_sharpe_ratio(1.2, large + [1.2], n_obs)["dsr"]
        assert d_large < d_small

    def test_non_finite_trials_ignored(self):
        out = deflated_sharpe_ratio(
            1.0, [0.5, float("nan"), float("inf"), None, 0.7], n_obs=252)
        assert out["n_trials"] == 2

    def test_single_trial_degenerates_to_psr(self):
        out = deflated_sharpe_ratio(1.0, [1.0], n_obs=252)
        assert out["expected_max_sharpe_annual"] == 0.0
        assert out["dsr"] == pytest.approx(out["psr_vs_zero"])


class TestWindowResultIntegration:
    def test_window_result_has_dsr_fields(self):
        from Classes.Optimization.walk_forward_optimizer import WindowResult
        from datetime import datetime
        w = WindowResult(
            window_id=0,
            train_start=datetime(2020, 1, 1), train_end=datetime(2021, 1, 1),
            test_start=datetime(2021, 1, 2), test_end=datetime(2021, 6, 1),
            best_params={}, in_sample_sortino=1.0, in_sample_sharpe=1.0,
            in_sample_profit_factor=1.5, in_sample_max_drawdown_pct=10.0,
            in_sample_num_trades=20, in_sample_total_return_pct=15.0,
            in_sample_calmar=1.0, out_sample_sortino=0.8, out_sample_sharpe=0.7,
            out_sample_profit_factor=1.3, out_sample_max_drawdown_pct=12.0,
            out_sample_num_trades=8, out_sample_total_return_pct=5.0,
            out_sample_calmar=0.6, sortino_degradation_pct=20.0,
            sharpe_degradation_pct=30.0,
        )
        # Defaults exist for backwards compatibility...
        assert w.in_sample_dsr == 0.0
        assert w.n_trials == 0
        # ...and the fields are settable.
        w.in_sample_dsr = 0.9
        w.n_trials = 50
        assert w.in_sample_dsr == 0.9
