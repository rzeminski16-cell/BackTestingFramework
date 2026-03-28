"""
Tests for walk-forward optimizer data classes and result aggregation.

Tests verify:
- WalkForwardMode enum values
- OptimizationConstraints defaults
- WindowResult data integrity
- WalkForwardResults aggregation
- MultiSecurityResults.from_individual_results computation
  (parameter consistency, combined averages, best/worst securities)
"""
import pytest
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Tuple

from Classes.Optimization.walk_forward_optimizer import (
    WalkForwardMode,
    OptimizationConstraints,
    WindowResult,
    WalkForwardResults,
    MultiSecurityResults,
)


# =============================================================================
# WalkForwardMode Tests
# =============================================================================

class TestWalkForwardMode:
    def test_rolling_value(self):
        assert WalkForwardMode.ROLLING.value == "rolling"

    def test_anchored_value(self):
        assert WalkForwardMode.ANCHORED.value == "anchored"

    def test_from_string(self):
        assert WalkForwardMode("rolling") == WalkForwardMode.ROLLING
        assert WalkForwardMode("anchored") == WalkForwardMode.ANCHORED

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            WalkForwardMode("invalid")


# =============================================================================
# OptimizationConstraints Tests
# =============================================================================

class TestOptimizationConstraints:
    def test_defaults(self):
        c = OptimizationConstraints()
        assert c.min_profit_factor == 1.5
        assert c.max_drawdown_percent == 25.0
        assert c.min_trades_per_year == 1

    def test_custom_values(self):
        c = OptimizationConstraints(
            min_profit_factor=2.0,
            max_drawdown_percent=10.0,
            min_trades_per_year=5,
        )
        assert c.min_profit_factor == 2.0
        assert c.max_drawdown_percent == 10.0
        assert c.min_trades_per_year == 5


# =============================================================================
# WindowResult Tests
# =============================================================================

def _make_window_result(window_id=1, in_sortino=2.0, out_sortino=1.5,
                         in_sharpe=1.8, out_sharpe=1.2) -> WindowResult:
    """Create a WindowResult with sensible defaults."""
    sortino_deg = ((in_sortino - out_sortino) / abs(in_sortino) * 100) if in_sortino != 0 else 0
    sharpe_deg = ((in_sharpe - out_sharpe) / abs(in_sharpe) * 100) if in_sharpe != 0 else 0
    return WindowResult(
        window_id=window_id,
        train_start=datetime(2023, 1, 1),
        train_end=datetime(2023, 6, 30),
        test_start=datetime(2023, 7, 1),
        test_end=datetime(2023, 12, 31),
        best_params={"ma_period": 50, "position_size": 0.1},
        in_sample_sortino=in_sortino,
        in_sample_sharpe=in_sharpe,
        in_sample_profit_factor=2.5,
        in_sample_max_drawdown_pct=10.0,
        in_sample_num_trades=20,
        in_sample_total_return_pct=15.0,
        in_sample_calmar=1.5,
        out_sample_sortino=out_sortino,
        out_sample_sharpe=out_sharpe,
        out_sample_profit_factor=1.8,
        out_sample_max_drawdown_pct=12.0,
        out_sample_num_trades=10,
        out_sample_total_return_pct=8.0,
        out_sample_calmar=0.67,
        sortino_degradation_pct=sortino_deg,
        sharpe_degradation_pct=sharpe_deg,
    )


class TestWindowResult:
    def test_fields_preserved(self):
        wr = _make_window_result()
        assert wr.window_id == 1
        assert wr.best_params == {"ma_period": 50, "position_size": 0.1}
        assert wr.in_sample_sortino == 2.0
        assert wr.out_sample_sortino == 1.5

    def test_degradation_calculation(self):
        wr = _make_window_result(in_sortino=2.0, out_sortino=1.0)
        assert wr.sortino_degradation_pct == pytest.approx(50.0)

    def test_trades_default_empty(self):
        wr = _make_window_result()
        assert wr.in_sample_trades == []
        assert wr.out_sample_trades == []


# =============================================================================
# WalkForwardResults Tests
# =============================================================================

def _make_walk_forward_results(
    symbol="AAPL",
    avg_in_sortino=2.0, avg_out_sortino=1.5,
    avg_in_sharpe=1.8, avg_out_sharpe=1.2,
    most_common=None,
) -> WalkForwardResults:
    if most_common is None:
        most_common = {"ma_period": 50, "position_size": 0.1}
    return WalkForwardResults(
        strategy_name="TestStrategy",
        symbol=symbol,
        windows=[_make_window_result()],
        avg_in_sample_sortino=avg_in_sortino,
        avg_out_sample_sortino=avg_out_sortino,
        avg_sortino_degradation_pct=((avg_in_sortino - avg_out_sortino) / abs(avg_in_sortino) * 100) if avg_in_sortino != 0 else 0,
        avg_in_sample_sharpe=avg_in_sharpe,
        avg_out_sample_sharpe=avg_out_sharpe,
        avg_sharpe_degradation_pct=((avg_in_sharpe - avg_out_sharpe) / abs(avg_in_sharpe) * 100) if avg_in_sharpe != 0 else 0,
        parameter_ranges={"ma_period": (20, 100)},
        parameter_std={"ma_period": 15.0},
        most_common_params=most_common,
        windows_passed_constraints=1,
        total_windows=1,
    )


class TestWalkForwardResults:
    def test_fields(self):
        wfr = _make_walk_forward_results()
        assert wfr.strategy_name == "TestStrategy"
        assert wfr.symbol == "AAPL"
        assert len(wfr.windows) == 1

    def test_degradation_percentages(self):
        wfr = _make_walk_forward_results(avg_in_sortino=2.0, avg_out_sortino=1.0)
        assert wfr.avg_sortino_degradation_pct == pytest.approx(50.0)

    def test_windows_passed_constraints(self):
        wfr = _make_walk_forward_results()
        assert wfr.windows_passed_constraints <= wfr.total_windows


# =============================================================================
# MultiSecurityResults Tests
# =============================================================================

class TestMultiSecurityResults:
    def test_from_individual_results_basic(self):
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL", 2.0, 1.5, 1.8, 1.2,
                                                {"ma_period": 50, "size": 0.1}),
            "GOOG": _make_walk_forward_results("GOOG", 3.0, 2.0, 2.5, 1.8,
                                                {"ma_period": 60, "size": 0.2}),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.strategy_name == "TestStrategy"
        assert set(msr.securities) == {"AAPL", "GOOG"}

    def test_combined_averages(self):
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL", 2.0, 1.0, 1.8, 1.0),
            "GOOG": _make_walk_forward_results("GOOG", 4.0, 3.0, 2.2, 2.0),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.combined_avg_in_sample_sortino == pytest.approx(3.0)
        assert msr.combined_avg_out_sample_sortino == pytest.approx(2.0)
        assert msr.combined_avg_in_sample_sharpe == pytest.approx(2.0)
        assert msr.combined_avg_out_sample_sharpe == pytest.approx(1.5)

    def test_best_and_worst_security(self):
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL", 2.0, 0.5, 1.8, 1.0),
            "GOOG": _make_walk_forward_results("GOOG", 3.0, 2.5, 2.5, 2.0),
            "MSFT": _make_walk_forward_results("MSFT", 2.5, 1.5, 2.0, 1.5),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.best_security == "GOOG"  # highest OOS sortino
        assert msr.worst_security == "AAPL"  # lowest OOS sortino

    def test_securities_with_positive_oos(self):
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL", 2.0, 1.0, 1.8, 1.0),
            "GOOG": _make_walk_forward_results("GOOG", 3.0, -0.5, 2.5, -0.2),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.securities_with_positive_oos == 1

    def test_total_windows(self):
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL"),
            "GOOG": _make_walk_forward_results("GOOG"),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.total_windows_all_securities == 2  # 1 window each

    def test_parameter_consistency_identical_params(self):
        """When all securities have same params, consistency should be 100."""
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL", most_common={"ma_period": 50}),
            "GOOG": _make_walk_forward_results("GOOG", most_common={"ma_period": 50}),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.param_consistency_scores["ma_period"] == pytest.approx(100.0)
        assert msr.consistent_params["ma_period"] == pytest.approx(50.0)

    def test_parameter_consistency_different_params(self):
        """Widely different params should have low consistency."""
        results_dict = {
            "AAPL": _make_walk_forward_results("AAPL", most_common={"ma_period": 20}),
            "GOOG": _make_walk_forward_results("GOOG", most_common={"ma_period": 100}),
        }
        msr = MultiSecurityResults.from_individual_results("TestStrategy", results_dict)
        assert msr.param_consistency_scores["ma_period"] < 100.0

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="No results"):
            MultiSecurityResults.from_individual_results("TestStrategy", {})
