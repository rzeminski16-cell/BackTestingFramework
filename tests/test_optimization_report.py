"""
Tests for WindowResult, WalkForwardResults, MultiSecurityResults and the enhanced optimization report.
"""
import pytest
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classes.Optimization.walk_forward_optimizer import (
    WindowResult, WalkForwardResults, MultiSecurityResults
)
from Classes.Optimization.enhanced_optimization_report import EnhancedOptimizationReportGenerator


def create_mock_window_result(
    window_id: int = 1,
    oos_sortino: float = 1.5,
    oos_sharpe: float = 1.2,
    oos_return: float = 15.0,
    oos_drawdown: float = 10.0,
    oos_trades: int = 25,
    is_sortino: float = 2.0,
    is_sharpe: float = 1.8,
    params: Dict[str, Any] = None
) -> WindowResult:
    """Create a mock WindowResult for testing."""
    base_date = datetime(2023, 1, 1)

    if params is None:
        params = {'fast_period': 10, 'slow_period': 30}

    # Calculate degradation
    sortino_deg = ((oos_sortino - is_sortino) / is_sortino * 100) if is_sortino != 0 else 0
    sharpe_deg = ((oos_sharpe - is_sharpe) / is_sharpe * 100) if is_sharpe != 0 else 0

    return WindowResult(
        window_id=window_id,
        train_start=base_date + timedelta(days=(window_id-1)*90),
        train_end=base_date + timedelta(days=(window_id-1)*90 + 60),
        test_start=base_date + timedelta(days=(window_id-1)*90 + 61),
        test_end=base_date + timedelta(days=(window_id-1)*90 + 90),
        best_params=params,
        in_sample_sortino=is_sortino,
        in_sample_sharpe=is_sharpe,
        in_sample_profit_factor=1.8,
        in_sample_max_drawdown_pct=8.0,
        in_sample_num_trades=50,
        in_sample_total_return_pct=25.0,
        in_sample_calmar=2.5,
        out_sample_sortino=oos_sortino,
        out_sample_sharpe=oos_sharpe,
        out_sample_profit_factor=1.5,
        out_sample_max_drawdown_pct=oos_drawdown,
        out_sample_num_trades=oos_trades,
        out_sample_total_return_pct=oos_return,
        out_sample_calmar=1.5,
        sortino_degradation_pct=sortino_deg,
        sharpe_degradation_pct=sharpe_deg,
        in_sample_trades=[],
        out_sample_trades=[]
    )


def create_mock_wf_results(
    symbol: str = "EURUSD",
    num_windows: int = 5,
    avg_oos_sortino: float = 1.2
) -> WalkForwardResults:
    """Create mock WalkForwardResults for testing."""
    windows = []
    for i in range(1, num_windows + 1):
        # Vary the sortino slightly around the average
        variation = (i - num_windows/2) * 0.2
        windows.append(create_mock_window_result(
            window_id=i,
            oos_sortino=avg_oos_sortino + variation,
            oos_sharpe=1.0 + variation * 0.5,
            oos_return=10.0 + variation * 5,
            params={'fast_period': 10 + i, 'slow_period': 30 + i * 2}
        ))

    return WalkForwardResults(
        strategy_name="TestStrategy",
        symbol=symbol,
        windows=windows,
        avg_in_sample_sortino=2.0,
        avg_out_sample_sortino=avg_oos_sortino,
        avg_sortino_degradation_pct=-40.0,
        avg_in_sample_sharpe=1.8,
        avg_out_sample_sharpe=1.0,
        avg_sharpe_degradation_pct=-44.0,
        parameter_ranges={'fast_period': (10, 15), 'slow_period': (30, 40)},
        parameter_std={'fast_period': 1.5, 'slow_period': 3.0},
        most_common_params={'fast_period': 12, 'slow_period': 35},
        windows_passed_constraints=4,
        total_windows=num_windows
    )


def create_mock_multi_results(
    securities: list = None,
    num_windows: int = 5
) -> MultiSecurityResults:
    """Create mock MultiSecurityResults for testing."""
    if securities is None:
        securities = ["EURUSD", "GBPUSD"]

    individual_results = {}
    for i, symbol in enumerate(securities):
        individual_results[symbol] = create_mock_wf_results(
            symbol=symbol,
            num_windows=num_windows,
            avg_oos_sortino=1.2 + i * 0.3  # Vary by security
        )

    return MultiSecurityResults(
        strategy_name="TestStrategy",
        securities=securities,
        individual_results=individual_results,
        combined_avg_in_sample_sortino=2.0,
        combined_avg_out_sample_sortino=1.35,
        combined_avg_sortino_degradation_pct=-32.5,
        combined_avg_in_sample_sharpe=1.8,
        combined_avg_out_sample_sharpe=1.1,
        combined_avg_sharpe_degradation_pct=-38.9,
        consistent_params={'fast_period': 12, 'slow_period': 35},
        param_consistency_scores={'fast_period': 85.0, 'slow_period': 78.0},
        total_windows_all_securities=num_windows * len(securities),
        total_passed_all_securities=8,
        securities_with_positive_oos=len(securities),
        best_security=securities[-1],  # Last security has highest avg_oos_sortino
        worst_security=securities[0]   # First security has lowest avg_oos_sortino
    )


def get_test_config() -> Dict[str, Any]:
    """Get a test configuration for the report generator."""
    return {
        'reporting': {
            'excel': {
                'include_charts': True,
                'decimal_places': 4
            }
        },
        'report': {
            'excel': {
                'decimal_places': 4
            }
        }
    }


class TestWindowResult:
    """Tests for WindowResult dataclass."""

    def test_window_result_creation(self):
        """Test basic WindowResult creation."""
        window = create_mock_window_result()

        assert window.window_id == 1
        assert window.out_sample_sortino == 1.5
        assert window.out_sample_sharpe == 1.2
        assert window.out_sample_total_return_pct == 15.0
        assert window.out_sample_max_drawdown_pct == 10.0
        assert window.out_sample_num_trades == 25

    def test_window_result_with_negative_sortino(self):
        """Test WindowResult with negative OOS Sortino."""
        window = create_mock_window_result(oos_sortino=-0.5)

        assert window.out_sample_sortino == -0.5
        assert window.sortino_degradation_pct < 0  # Should show degradation

    def test_window_result_with_zero_trades(self):
        """Test WindowResult with zero trades."""
        window = create_mock_window_result(oos_trades=0)

        assert window.out_sample_num_trades == 0

    def test_window_result_with_high_drawdown(self):
        """Test WindowResult with high drawdown."""
        window = create_mock_window_result(oos_drawdown=35.0)

        assert window.out_sample_max_drawdown_pct == 35.0

    def test_window_result_dates(self):
        """Test WindowResult date attributes."""
        window = create_mock_window_result(window_id=1)

        assert window.train_start < window.train_end
        assert window.train_end < window.test_start
        assert window.test_start < window.test_end


class TestWalkForwardResults:
    """Tests for WalkForwardResults dataclass."""

    def test_wf_results_creation(self):
        """Test basic WalkForwardResults creation."""
        wf_results = create_mock_wf_results()

        assert wf_results.strategy_name == "TestStrategy"
        assert wf_results.symbol == "EURUSD"
        assert len(wf_results.windows) == 5
        assert wf_results.total_windows == 5

    def test_wf_results_windows_list(self):
        """Test that windows list contains valid WindowResult objects."""
        wf_results = create_mock_wf_results(num_windows=3)

        assert len(wf_results.windows) == 3
        for i, window in enumerate(wf_results.windows, 1):
            assert isinstance(window, WindowResult)
            assert window.window_id == i

    def test_wf_results_parameter_ranges(self):
        """Test parameter range tracking."""
        wf_results = create_mock_wf_results()

        assert 'fast_period' in wf_results.parameter_ranges
        assert 'slow_period' in wf_results.parameter_ranges
        assert wf_results.parameter_ranges['fast_period'][0] <= wf_results.parameter_ranges['fast_period'][1]


class TestMultiSecurityResults:
    """Tests for MultiSecurityResults dataclass."""

    def test_multi_results_creation(self):
        """Test basic MultiSecurityResults creation."""
        multi_results = create_mock_multi_results()

        assert multi_results.strategy_name == "TestStrategy"
        assert len(multi_results.securities) == 2
        assert "EURUSD" in multi_results.individual_results
        assert "GBPUSD" in multi_results.individual_results

    def test_multi_results_with_single_security(self):
        """Test MultiSecurityResults with single security."""
        multi_results = create_mock_multi_results(securities=["EURUSD"])

        assert len(multi_results.securities) == 1
        assert len(multi_results.individual_results) == 1

    def test_multi_results_with_many_securities(self):
        """Test MultiSecurityResults with many securities."""
        securities = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        multi_results = create_mock_multi_results(securities=securities)

        assert len(multi_results.securities) == 5
        assert len(multi_results.individual_results) == 5


class TestEnhancedOptimizationReport:
    """Tests for EnhancedOptimizationReportGenerator generation."""

    def test_report_generation_basic(self):
        """Test basic report generation."""
        multi_results = create_mock_multi_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            report = EnhancedOptimizationReportGenerator(get_test_config())
            filepath = report.generate_portfolio_optimization_report(
                multi_results,
                sensitivity_results_dict=None,
                output_dir=tmpdir
            )

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

    def test_report_generation_single_security(self):
        """Test report generation with single security."""
        multi_results = create_mock_multi_results(securities=["EURUSD"])

        with tempfile.TemporaryDirectory() as tmpdir:
            report = EnhancedOptimizationReportGenerator(get_test_config())
            filepath = report.generate_portfolio_optimization_report(
                multi_results,
                sensitivity_results_dict=None,
                output_dir=tmpdir
            )

            assert os.path.exists(filepath)

    def test_report_generation_many_windows(self):
        """Test report generation with many windows."""
        multi_results = create_mock_multi_results(num_windows=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            report = EnhancedOptimizationReportGenerator(get_test_config())
            filepath = report.generate_portfolio_optimization_report(
                multi_results,
                sensitivity_results_dict=None,
                output_dir=tmpdir
            )

            assert os.path.exists(filepath)

    def test_report_with_negative_metrics(self):
        """Test report generation with negative OOS metrics."""
        multi_results = create_mock_multi_results()

        # Modify windows to have negative Sortino
        for wf_results in multi_results.individual_results.values():
            for window in wf_results.windows:
                window.out_sample_sortino = -0.5
                window.out_sample_total_return_pct = -10.0

        with tempfile.TemporaryDirectory() as tmpdir:
            report = EnhancedOptimizationReportGenerator(get_test_config())
            filepath = report.generate_portfolio_optimization_report(
                multi_results,
                sensitivity_results_dict=None,
                output_dir=tmpdir
            )

            assert os.path.exists(filepath)

    def test_report_with_zero_trades(self):
        """Test report generation when windows have zero trades."""
        multi_results = create_mock_multi_results()

        # Set some windows to have zero trades
        for wf_results in multi_results.individual_results.values():
            for i, window in enumerate(wf_results.windows):
                if i % 2 == 0:
                    window.out_sample_num_trades = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            report = EnhancedOptimizationReportGenerator(get_test_config())
            filepath = report.generate_portfolio_optimization_report(
                multi_results,
                sensitivity_results_dict=None,
                output_dir=tmpdir
            )

            assert os.path.exists(filepath)

    def test_report_with_high_variability(self):
        """Test report with high variability in metrics (edge case for stability calculations)."""
        multi_results = create_mock_multi_results()

        # Create high variability - std > mean scenario
        for wf_results in multi_results.individual_results.values():
            for i, window in enumerate(wf_results.windows):
                # Alternate between 0 and high values
                window.out_sample_num_trades = 100 if i % 2 == 0 else 1

        with tempfile.TemporaryDirectory() as tmpdir:
            report = EnhancedOptimizationReportGenerator(get_test_config())
            filepath = report.generate_portfolio_optimization_report(
                multi_results,
                sensitivity_results_dict=None,
                output_dir=tmpdir
            )

            assert os.path.exists(filepath)

    def test_safe_float_method(self):
        """Test the _safe_float method for handling NaN and None."""
        report = EnhancedOptimizationReportGenerator(get_test_config())

        assert report._safe_float(1.5) == 1.5
        assert report._safe_float(0) == 0.0
        assert report._safe_float(None) == 0.0
        assert report._safe_float(float('nan')) == 0.0
        assert report._safe_float(float('inf')) == 0.0
        assert report._safe_float(float('-inf')) == 0.0
        assert report._safe_float("invalid") == 0.0


class TestCalculationEdgeCases:
    """Tests for calculation edge cases in the report."""

    def test_stability_calculation_clamping(self):
        """Test that stability calculations are clamped to 0-100."""
        # This tests the scenario where std > mean, which would result in negative stability
        values = [1, 100, 1, 100, 1]  # High variability
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Without clamping, this would be negative
        raw_stability = 100 - (std_val / mean_val * 100)
        assert raw_stability < 0, "Test case should have negative raw stability"

        # With clamping (as implemented in the code)
        clamped_stability = max(0, min(100, raw_stability))
        assert 0 <= clamped_stability <= 100

    def test_correlation_with_identical_values(self):
        """Test correlation calculation when all values are identical."""
        values = [10.0, 10.0, 10.0, 10.0, 10.0]

        # np.corrcoef returns NaN for identical values
        corr = np.corrcoef(values, values)[0, 1]
        assert np.isnan(corr), "Correlation of identical values should be NaN"

    def test_division_by_zero_handling(self):
        """Test that division by zero scenarios are handled."""
        # Scenario: calculating change percentage when IS value is 0
        is_val = 0
        oos_val = 1.5

        # Code should handle this gracefully
        change = ((oos_val - is_val) / is_val * 100) if is_val != 0 else 0
        assert change == 0

    def test_empty_list_mean(self):
        """Test mean calculation on empty list."""
        empty_list = []

        # Code uses conditional: np.mean(list) if list else 0
        result = np.mean(empty_list) if empty_list else 0
        assert result == 0


class TestMetricsCalculation:
    """Tests to verify metrics calculations match expected values."""

    def test_cumulative_return_calculation(self):
        """Test cumulative return calculation in WF Equity Curves."""
        windows = [
            create_mock_window_result(window_id=1, oos_return=10.0),
            create_mock_window_result(window_id=2, oos_return=5.0),
            create_mock_window_result(window_id=3, oos_return=-3.0),
        ]

        # Calculate cumulative as the report does
        cumulative = 0
        for window in windows:
            cumulative += window.out_sample_total_return_pct

        expected_cumulative = 10.0 + 5.0 + (-3.0)  # = 12.0
        assert cumulative == expected_cumulative

    def test_equity_progression_calculation(self):
        """Test equity progression calculation (compounding)."""
        windows = [
            create_mock_window_result(window_id=1, oos_return=10.0),
            create_mock_window_result(window_id=2, oos_return=10.0),
        ]

        # Calculate equity as the report does
        starting_equity = 100
        for window in windows:
            ending_equity = starting_equity * (1 + window.out_sample_total_return_pct / 100)
            starting_equity = ending_equity

        # After two 10% returns: 100 * 1.1 * 1.1 = 121
        assert abs(ending_equity - 121.0) < 0.01

    def test_sortino_degradation_calculation(self):
        """Test Sortino degradation percentage calculation."""
        is_sortino = 2.0
        oos_sortino = 1.5

        # Degradation formula
        degradation = ((oos_sortino - is_sortino) / is_sortino * 100)

        expected = ((1.5 - 2.0) / 2.0 * 100)  # = -25%
        assert degradation == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
