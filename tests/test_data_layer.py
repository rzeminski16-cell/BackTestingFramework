"""
Comprehensive tests for the Data layer:
- DataLoader
- DataValidator
- MissingColumnError / MissingIndicatorError
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import os

from Classes.Data.data_loader import DataLoader, MissingColumnError
from Classes.Data.data_validator import DataValidator, MissingIndicatorError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a basic test CSV
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.randint(1000, 10000, 100),
        })
        df.to_csv(os.path.join(tmpdir, 'TEST.csv'), index=False)

        # Create a CSV with 'time' column instead of 'date'
        df2 = df.copy()
        df2 = df2.rename(columns={'date': 'time'})
        df2.to_csv(os.path.join(tmpdir, 'TEST_TIME.csv'), index=False)

        # Create a CSV with extra columns
        df3 = df.copy()
        df3['atr_14_atr'] = np.random.uniform(1, 5, 100)
        df3['ema_50_ema'] = np.random.uniform(95, 105, 100)
        df3.to_csv(os.path.join(tmpdir, 'TEST_INDICATORS.csv'), index=False)

        yield tmpdir


@pytest.fixture
def loader(temp_data_dir):
    return DataLoader(Path(temp_data_dir))


# =============================================================================
# DataLoader Tests
# =============================================================================

class TestDataLoader:
    def test_init_valid_directory(self, loader):
        assert loader.data_directory.exists()

    def test_init_invalid_directory(self):
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            DataLoader(Path("/nonexistent/directory"))

    def test_load_csv_basic(self, loader):
        df = loader.load_csv('TEST')
        assert len(df) == 100
        assert 'date' in df.columns
        assert 'close' in df.columns

    def test_load_csv_date_parsed(self, loader):
        df = loader.load_csv('TEST')
        assert pd.api.types.is_datetime64_any_dtype(df['date'])

    def test_load_csv_sorted_by_date(self, loader):
        df = loader.load_csv('TEST')
        assert df['date'].is_monotonic_increasing

    def test_load_csv_time_column_converted(self, loader):
        df = loader.load_csv('TEST_TIME')
        assert 'date' in df.columns
        assert 'time' not in df.columns

    def test_load_csv_nonexistent_file(self, loader):
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            loader.load_csv('NONEXISTENT')

    def test_load_csv_with_required_columns(self, loader):
        df = loader.load_csv('TEST', required_columns=['date', 'close'])
        assert len(df) > 0

    def test_load_csv_missing_required_columns(self, loader):
        with pytest.raises(MissingColumnError):
            loader.load_csv('TEST', required_columns=['date', 'close', 'nonexistent_col'])

    def test_load_csv_column_names_lowercase(self, temp_data_dir):
        """Verify column names are normalized to lowercase."""
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'Close': np.random.uniform(95, 105, 10),
            'Volume': np.random.randint(1000, 10000, 10),
        })
        df.to_csv(os.path.join(temp_data_dir, 'UPPER.csv'), index=False)
        loader = DataLoader(Path(temp_data_dir))
        loaded = loader.load_csv('UPPER')
        assert 'date' in loaded.columns
        assert 'close' in loaded.columns
        assert 'Date' not in loaded.columns

    def test_load_multiple(self, loader):
        data = loader.load_multiple(['TEST', 'TEST_INDICATORS'])
        assert 'TEST' in data
        assert 'TEST_INDICATORS' in data

    def test_load_multiple_strict_missing_columns(self, loader):
        with pytest.raises(MissingColumnError):
            loader.load_multiple(['TEST'], required_columns=['nonexistent'], strict=True)

    def test_load_multiple_non_strict_skips_errors(self, loader):
        data = loader.load_multiple(
            ['TEST', 'NONEXISTENT'],
            strict=False
        )
        assert 'TEST' in data
        assert 'NONEXISTENT' not in data

    def test_get_available_symbols(self, loader):
        symbols = loader.get_available_symbols()
        assert 'TEST' in symbols
        assert 'TEST_TIME' in symbols
        assert 'TEST_INDICATORS' in symbols

    def test_filter_by_date_range(self, loader):
        df = loader.load_csv('TEST')
        filtered = loader.filter_by_date_range(
            df,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 2, 15)
        )
        assert len(filtered) < len(df)
        assert filtered['date'].min() >= pd.Timestamp('2024-01-15')
        assert filtered['date'].max() <= pd.Timestamp('2024-02-15')

    def test_filter_by_date_range_no_date_column(self, loader):
        df = pd.DataFrame({'close': [1, 2, 3]})
        with pytest.raises(ValueError, match="must have 'date' column"):
            loader.filter_by_date_range(df)

    def test_filter_by_date_range_start_only(self, loader):
        df = loader.load_csv('TEST')
        filtered = loader.filter_by_date_range(df, start_date=datetime(2024, 3, 1))
        assert filtered['date'].min() >= pd.Timestamp('2024-03-01')

    def test_filter_by_date_range_end_only(self, loader):
        df = loader.load_csv('TEST')
        filtered = loader.filter_by_date_range(df, end_date=datetime(2024, 2, 1))
        assert filtered['date'].max() <= pd.Timestamp('2024-02-01')

    def test_normalize_column_name(self):
        assert DataLoader.normalize_column_name("  Close  ") == "close"
        assert DataLoader.normalize_column_name("VOLUME") == "volume"


# =============================================================================
# MissingColumnError Tests
# =============================================================================

class TestMissingColumnError:
    def test_error_message_contains_symbol(self):
        err = MissingColumnError(
            symbol="AAPL",
            missing_columns=['atr_14'],
            available_columns=['date', 'close']
        )
        assert "AAPL" in str(err)
        assert "atr_14" in str(err)

    def test_error_attributes(self):
        err = MissingColumnError(
            symbol="AAPL",
            missing_columns=['col_a', 'col_b'],
            available_columns=['date', 'close']
        )
        assert err.symbol == "AAPL"
        assert 'col_a' in err.missing_columns
        assert 'col_b' in err.missing_columns


# =============================================================================
# DataValidator Tests
# =============================================================================

class TestDataValidator:
    def _make_valid_df(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000 + i * 100 for i in range(10)],
        })

    def test_validate_required_columns_pass(self):
        df = self._make_valid_df()
        DataValidator.validate_required_columns(df, ['date', 'close'], 'TEST')

    def test_validate_required_columns_fail(self):
        df = self._make_valid_df()
        with pytest.raises(MissingIndicatorError):
            DataValidator.validate_required_columns(df, ['date', 'close', 'nonexistent'], 'TEST')

    def test_validate_indicators_exist(self):
        df = self._make_valid_df()
        result = DataValidator.validate_indicators_exist(df, ['close', 'nonexistent'])
        assert result['close'] is True
        assert result['nonexistent'] is False

    def test_get_missing_indicators(self):
        df = self._make_valid_df()
        missing = DataValidator.get_missing_indicators(df, ['close', 'xyz'])
        assert missing == ['xyz']

    def test_validate_date_column_pass(self):
        df = self._make_valid_df()
        DataValidator.validate_date_column(df, 'date')

    def test_validate_date_column_missing(self):
        df = pd.DataFrame({'close': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            DataValidator.validate_date_column(df, 'date')

    def test_validate_date_column_wrong_type(self):
        df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02']})
        with pytest.raises(ValueError, match="datetime type"):
            DataValidator.validate_date_column(df, 'date')

    def test_validate_numeric_columns(self):
        df = self._make_valid_df()
        DataValidator.validate_numeric_columns(df, ['close', 'volume'])

    def test_validate_numeric_columns_non_numeric(self):
        df = pd.DataFrame({'close': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match="numeric type"):
            DataValidator.validate_numeric_columns(df, ['close'])

    def test_validate_no_missing_values_pass(self):
        df = self._make_valid_df()
        DataValidator.validate_no_missing_values(df, ['close'])

    def test_validate_no_missing_values_fail(self):
        df = self._make_valid_df()
        df.loc[0, 'close'] = None
        with pytest.raises(ValueError, match="Missing values"):
            DataValidator.validate_no_missing_values(df, ['close'])

    def test_validate_positive_values_pass(self):
        df = self._make_valid_df()
        DataValidator.validate_positive_values(df, ['close'])

    def test_validate_positive_values_fail(self):
        df = self._make_valid_df()
        df.loc[0, 'close'] = -1
        with pytest.raises(ValueError, match="non-positive"):
            DataValidator.validate_positive_values(df, ['close'])

    def test_validate_data_sorted_pass(self):
        df = self._make_valid_df()
        DataValidator.validate_data_sorted(df, 'date')

    def test_validate_data_sorted_fail(self):
        df = self._make_valid_df()
        df = df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="sorted"):
            DataValidator.validate_data_sorted(df, 'date')

    def test_validate_file_exists(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("data")
        DataValidator.validate_file_exists(f)

    def test_validate_file_not_exists(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataValidator.validate_file_exists(tmp_path / "nope.csv")

    def test_validate_data_frame_full(self):
        df = self._make_valid_df()
        DataValidator.validate_data_frame(df, ['date', 'close'], symbol='TEST')

    def test_validate_data_frame_empty(self):
        df = pd.DataFrame(columns=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        with pytest.raises(ValueError):
            DataValidator.validate_data_frame(df, ['date', 'close'], symbol='TEST')

    def test_validate_data_frame_allow_missing(self):
        df = self._make_valid_df()
        df.loc[0, 'close'] = None
        # Should not raise when allow_missing_values=True
        DataValidator.validate_data_frame(
            df, ['date', 'close'], symbol='TEST', allow_missing_values=True
        )
