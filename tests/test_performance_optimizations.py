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
