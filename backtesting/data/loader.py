"""
Data loading utilities for backtesting.

Supports loading data from:
- Yahoo Finance (via yfinance)
- CSV files
- Pandas DataFrames
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd


class DataLoader:
    """
    Utility class for loading market data.

    Example:
        loader = DataLoader()

        # Single security
        data = loader.from_yahoo("AAPL", start="2020-01-01", end="2023-12-31")

        # Portfolio of securities
        portfolio = loader.from_yahoo(
            ["AAPL", "GOOGL", "MSFT"],
            start="2020-01-01",
            end="2023-12-31"
        )

        # From CSV
        data = loader.from_csv("data/prices.csv")
    """

    @staticmethod
    def from_yahoo(
        symbols: Union[str, List[str]],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        interval: str = "1d",
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from Yahoo Finance.

        Args:
            symbols: Single symbol or list of symbols
            start: Start date (string or datetime)
            end: End date (string or datetime)
            interval: Data interval (1d, 1wk, 1mo, etc.)

        Returns:
            DataFrame for single symbol, dict of DataFrames for multiple
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        if isinstance(symbols, str):
            ticker = yf.Ticker(symbols)
            data = ticker.history(start=start, end=end, interval=interval)
            data.columns = [c.lower() for c in data.columns]
            return data

        # Multiple symbols
        result = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end, interval=interval)
            data.columns = [c.lower() for c in data.columns]
            result[symbol] = data

        return result

    @staticmethod
    def from_csv(
        filepath: str,
        date_column: Optional[str] = None,
        date_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to CSV file
            date_column: Name of the date column (auto-detected if None)
            date_format: Date format string (auto-detected if None)

        Returns:
            DataFrame with DatetimeIndex
        """
        df = pd.read_csv(filepath)

        # Find date column
        if date_column is None:
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols:
                date_column = date_cols[0]
            else:
                date_column = df.columns[0]

        # Parse dates
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])

        df.set_index(date_column, inplace=True)
        df.columns = [c.lower() for c in df.columns]

        return df

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        date_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Normalize a DataFrame for backtesting.

        Args:
            df: Input DataFrame
            date_column: Column to use as index (if not already DatetimeIndex)

        Returns:
            Normalized DataFrame with DatetimeIndex and lowercase columns
        """
        df = df.copy()

        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df.columns = [c.lower() for c in df.columns]

        return df

    @staticmethod
    def generate_sample_data(
        days: int = 252,
        start_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0005,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic price data for testing.

        Args:
            days: Number of trading days
            start_price: Starting price
            volatility: Daily volatility (standard deviation of returns)
            drift: Daily drift (mean return)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with OHLCV columns
        """
        import numpy as np

        if seed is not None:
            np.random.seed(seed)

        # Generate returns
        returns = np.random.normal(drift, volatility, days)

        # Generate prices
        close = start_price * np.exp(np.cumsum(returns))

        # Generate OHLV
        daily_range = volatility * close
        high = close + np.random.uniform(0, 1, days) * daily_range
        low = close - np.random.uniform(0, 1, days) * daily_range
        open_price = low + np.random.uniform(0, 1, days) * (high - low)

        # Ensure OHLC relationships
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))

        # Generate volume
        volume = np.random.randint(1000000, 10000000, days)

        # Create date index
        dates = pd.date_range(start="2020-01-01", periods=days, freq="B")

        return pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)
