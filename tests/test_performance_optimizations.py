"""
Tests for performance optimizations.

These tests verify that:
1. HistoricalDataView properly prevents look-ahead bias
2. Numba JIT-compiled functions produce identical results
3. Backtest results are identical before/after optimizations
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestHistoricalDataView(unittest.TestCase):
    """Tests for HistoricalDataView look-ahead protection."""

    def setUp(self):
        """Create test data."""
        from Classes.Data.historical_data_view import HistoricalDataView
        self.HistoricalDataView = HistoricalDataView

        # Create sample DataFrame with 100 rows
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        self.df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        })

    def test_length_limited_to_valid_index(self):
        """Verify len() returns valid_end_index + 1, not full DataFrame length."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)
        self.assertEqual(len(view), 51)  # 0 to 50 inclusive
        self.assertNotEqual(len(view), 100)

    def test_iloc_access_within_bounds(self):
        """Verify iloc access works within valid bounds."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)

        # Should work
        row = view.iloc[25]
        self.assertIsNotNone(row)
        self.assertEqual(row['date'], self.df.iloc[25]['date'])

        # Last valid row
        row = view.iloc[50]
        self.assertIsNotNone(row)

    def test_iloc_access_beyond_bounds_raises_error(self):
        """CRITICAL: Verify iloc access beyond valid_end_index raises IndexError."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)

        # Attempting to access future data must raise an error
        with self.assertRaises(IndexError) as context:
            _ = view.iloc[51]

        self.assertIn("beyond valid historical data", str(context.exception))

        # Also test further out
        with self.assertRaises(IndexError):
            _ = view.iloc[75]

    def test_slice_access_truncated(self):
        """Verify slice access is truncated at valid_end_index."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)

        # Slice that would go beyond should be truncated
        sliced = view.iloc[40:60]
        self.assertEqual(len(sliced), 11)  # 40 to 50 inclusive

    def test_negative_indexing(self):
        """Verify negative indexing works correctly."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)

        # -1 should be the last valid row (index 50)
        row = view.iloc[-1]
        self.assertEqual(row['date'], self.df.iloc[50]['date'])

    def test_column_access(self):
        """Verify column access is limited to valid range."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)

        close_series = view['close']
        self.assertEqual(len(close_series), 51)

    def test_shape_property(self):
        """Verify shape returns valid data dimensions."""
        view = self.HistoricalDataView(self.df, valid_end_index=50)

        self.assertEqual(view.shape, (51, 6))


class TestNumbaCalculations(unittest.TestCase):
    """Tests for Numba JIT-compiled calculations."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)  # For reproducibility
        self.n = 1000

        # Test data for AlphaTrend calculation
        self.up_band = np.random.uniform(95, 105, self.n)
        self.down_band = np.random.uniform(105, 115, self.n)
        self.momentum_bullish = np.random.choice([True, False], self.n)

        # Test data for signal filtering
        self.cross_up = np.zeros(self.n, dtype=bool)
        self.cross_down = np.zeros(self.n, dtype=bool)
        # Add some signals
        for i in range(0, self.n, 20):
            if i % 40 == 0:
                self.cross_up[i] = True
            else:
                self.cross_down[i] = True

    def test_alphatrend_numba_matches_python(self):
        """Verify Numba AlphaTrend produces identical results to Python version."""
        from strategies.alphatrend_strategy import _alphatrend_numba

        # Pure Python implementation (reference)
        def alphatrend_python(up_band, down_band, momentum_bullish):
            n = len(up_band)
            alphatrend = np.zeros(n)
            alphatrend[0] = up_band[0]

            for i in range(1, n):
                if momentum_bullish[i]:
                    if up_band[i] < alphatrend[i-1]:
                        alphatrend[i] = max(alphatrend[i-1], up_band[i])
                    else:
                        alphatrend[i] = up_band[i]
                else:
                    if down_band[i] > alphatrend[i-1]:
                        alphatrend[i] = min(alphatrend[i-1], down_band[i])
                    else:
                        alphatrend[i] = down_band[i]
            return alphatrend

        # Calculate with both implementations
        result_numba = _alphatrend_numba(
            self.up_band.astype(np.float64),
            self.down_band.astype(np.float64),
            self.momentum_bullish.astype(np.bool_)
        )
        result_python = alphatrend_python(
            self.up_band,
            self.down_band,
            self.momentum_bullish
        )

        # Verify identical results
        np.testing.assert_array_almost_equal(
            result_numba, result_python, decimal=10,
            err_msg="Numba AlphaTrend differs from Python implementation"
        )

    def test_filter_signals_numba_matches_python(self):
        """Verify Numba signal filtering produces identical results."""
        from strategies.alphatrend_strategy import _filter_signals_numba

        # Pure Python implementation (reference)
        def filter_signals_python(cross_up, cross_down):
            n = len(cross_up)
            filtered_buy = np.zeros(n, dtype=bool)
            filtered_sell = np.zeros(n, dtype=bool)
            last_signal = 0  # 0 = none, 1 = buy, 2 = sell

            for i in range(n):
                if cross_up[i] and last_signal != 1:
                    filtered_buy[i] = True
                    last_signal = 1
                elif cross_down[i] and last_signal != 2:
                    filtered_sell[i] = True
                    last_signal = 2

            return filtered_buy, filtered_sell

        # Calculate with both implementations
        buy_numba, sell_numba = _filter_signals_numba(
            self.cross_up.astype(np.bool_),
            self.cross_down.astype(np.bool_)
        )
        buy_python, sell_python = filter_signals_python(
            self.cross_up,
            self.cross_down
        )

        # Verify identical results
        np.testing.assert_array_equal(
            buy_numba, buy_python,
            err_msg="Numba buy signals differ from Python implementation"
        )
        np.testing.assert_array_equal(
            sell_numba, sell_python,
            err_msg="Numba sell signals differ from Python implementation"
        )


class TestBacktestIntegrity(unittest.TestCase):
    """Integration tests for backtest accuracy after optimizations."""

    def test_strategy_context_with_historical_view(self):
        """Verify StrategyContext works correctly with HistoricalDataView."""
        from Classes.Data.historical_data_view import HistoricalDataView
        from Classes.Strategy.strategy_context import StrategyContext

        # Create sample data
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        })

        # Create context with HistoricalDataView
        view = HistoricalDataView(df, valid_end_index=50)
        context = StrategyContext(
            data=view,
            current_index=50,
            current_price=df.iloc[50]['close'],
            current_date=pd.Timestamp(df.iloc[50]['date']),
            position=None,
            available_capital=100000.0,
            total_equity=100000.0,
            symbol='TEST',
            fx_rate=1.0
        )

        # Verify context properties work
        self.assertEqual(len(context.data), 51)
        current_bar = context.current_bar
        self.assertIsNotNone(current_bar)
        self.assertEqual(current_bar['date'], df.iloc[50]['date'])

        # Verify get_bar with offset works
        prev_bar = context.get_bar(-1)
        self.assertIsNotNone(prev_bar)
        self.assertEqual(prev_bar['date'], df.iloc[49]['date'])

    def test_look_ahead_protection_in_strategy(self):
        """Verify strategy cannot access future data through context."""
        from Classes.Data.historical_data_view import HistoricalDataView
        from Classes.Strategy.strategy_context import StrategyContext

        # Create sample data with identifiable future values
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        df = pd.DataFrame({
            'date': dates,
            'close': list(range(100))  # Close = index for easy verification
        })

        # Create view at index 50
        view = HistoricalDataView(df, valid_end_index=50)
        context = StrategyContext(
            data=view,
            current_index=50,
            current_price=50,
            current_date=pd.Timestamp(df.iloc[50]['date']),
            position=None,
            available_capital=100000.0,
            total_equity=100000.0,
            symbol='TEST',
            fx_rate=1.0
        )

        # Attempting to access future data via context.data.iloc should fail
        with self.assertRaises(IndexError):
            _ = context.data.iloc[51]

        # get_bar with positive offset beyond data should return None
        # (due to len(data) being limited)
        future_bar = context.get_bar(1)
        self.assertIsNone(future_bar)


if __name__ == '__main__':
    unittest.main()
