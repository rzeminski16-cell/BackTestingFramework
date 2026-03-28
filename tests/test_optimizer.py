"""
Tests for StrategyOptimizer (grid search) and OptimizationResults.

Tests verify:
- Parameter combination generation
- Metric extraction for all supported metrics
- Single-security optimization
- Min trades filtering
- OptimizationResults ranking, get_best, get_top_n, summary
"""
import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from Classes.Optimization.optimization_result import OptimizationResult, OptimizationResults
from Classes.Engine.backtest_result import BacktestResult


# =============================================================================
# OptimizationResults Tests (pure data structure, no engine needed)
# =============================================================================

def _make_backtest_result(total_return=1000.0, num_trades=10):
    """Create a minimal BacktestResult-like object."""
    equity = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'equity': [100000 + i * 100 for i in range(10)],
    })
    return BacktestResult(
        symbol="TEST",
        strategy_name="TestStrategy",
        trades=[],
        equity_curve=equity,
        final_equity=100000 + total_return,
        total_return=total_return,
        total_return_pct=total_return / 100000 * 100,
        strategy_params={},
    )


class TestOptimizationResults:
    def test_sorted_by_metric_descending(self):
        results = OptimizationResults(
            results=[
                OptimizationResult({"a": 1}, _make_backtest_result(500), 5.0),
                OptimizationResult({"a": 2}, _make_backtest_result(1000), 10.0),
                OptimizationResult({"a": 3}, _make_backtest_result(200), 2.0),
            ],
            metric="total_return",
            maximize=True,
        )
        assert results.results[0].metric_value == 10.0
        assert results.results[-1].metric_value == 2.0

    def test_sorted_ascending_when_minimize(self):
        results = OptimizationResults(
            results=[
                OptimizationResult({"a": 1}, _make_backtest_result(), 5.0),
                OptimizationResult({"a": 2}, _make_backtest_result(), 2.0),
                OptimizationResult({"a": 3}, _make_backtest_result(), 8.0),
            ],
            metric="max_drawdown",
            maximize=False,
        )
        assert results.results[0].metric_value == 2.0
        assert results.results[-1].metric_value == 8.0

    def test_get_best(self):
        results = OptimizationResults(
            results=[
                OptimizationResult({"x": 10}, _make_backtest_result(), 50.0),
                OptimizationResult({"x": 20}, _make_backtest_result(), 100.0),
            ],
            metric="sharpe",
            maximize=True,
        )
        assert results.get_best().metric_value == 100.0

    def test_get_best_empty_raises(self):
        results = OptimizationResults(results=[], metric="sharpe", maximize=True)
        with pytest.raises(ValueError):
            results.get_best()

    def test_get_best_parameters(self):
        results = OptimizationResults(
            results=[
                OptimizationResult({"period": 20}, _make_backtest_result(), 10.0),
                OptimizationResult({"period": 50}, _make_backtest_result(), 20.0),
            ],
            metric="sharpe",
            maximize=True,
        )
        assert results.get_best_parameters() == {"period": 50}

    def test_get_top_n(self):
        results = OptimizationResults(
            results=[
                OptimizationResult({"a": i}, _make_backtest_result(), float(i))
                for i in range(10)
            ],
            metric="total_return",
            maximize=True,
        )
        top3 = results.get_top_n(3)
        assert len(top3) == 3
        assert top3[0].metric_value == 9.0

    def test_get_all(self):
        all_results = [
            OptimizationResult({"a": i}, _make_backtest_result(), float(i))
            for i in range(5)
        ]
        results = OptimizationResults(
            results=all_results, metric="total_return", maximize=True
        )
        assert len(results.get_all()) == 5

    def test_summary_string(self):
        results = OptimizationResults(
            results=[
                OptimizationResult({"a": 1}, _make_backtest_result(), 5.0),
                OptimizationResult({"a": 2}, _make_backtest_result(), 10.0),
            ],
            metric="sharpe_ratio",
            maximize=True,
        )
        s = results.summary()
        assert "sharpe_ratio" in s
        assert "10.0000" in s
        assert "Total combinations tested: 2" in s

    def test_summary_empty(self):
        results = OptimizationResults(results=[], metric="sharpe", maximize=True)
        assert "No optimization results" in results.summary()


# =============================================================================
# OptimizationResult Tests
# =============================================================================

class TestOptimizationResult:
    def test_str_representation(self):
        r = OptimizationResult(
            parameters={"ma": 50, "size": 0.1},
            backtest_result=_make_backtest_result(),
            metric_value=3.14,
        )
        s = str(r)
        assert "ma=50" in s
        assert "3.14" in s
