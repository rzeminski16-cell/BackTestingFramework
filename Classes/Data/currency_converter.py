"""
Currency conversion module for handling multi-currency backtesting.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class CurrencyConverter:
    """
    Handles currency conversions for backtesting with time-series exchange rates.

    The base currency is always GBP (currency of account). Exchange rates should be
    provided as XXX/GBP pairs (e.g., USD/GBP = 0.79 means 1 USD = 0.79 GBP).
    """

    def __init__(self, base_currency: str = "GBP"):
        """
        Initialize currency converter.

        Args:
            base_currency: Base currency code (default: GBP)
        """
        self.base_currency = base_currency
        self.rates: Dict[str, pd.DataFrame] = {}

    def load_rates(self, currency_pair: str, file_path: Path) -> None:
        """
        Load exchange rates from a CSV file.

        Args:
            currency_pair: Currency pair code (e.g., 'USD/GBP', 'EUR/GBP')
            file_path: Path to CSV file with columns: date, rate (or OHLC format)

        Supported formats:
        - Simple: date, rate
        - OHLC: date, open, high, low, close (uses 'close' as rate)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Currency rate file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle different column formats
        if 'rate' not in df.columns:
            # Check for OHLC format - use 'close' as the rate
            if 'close' in df.columns:
                df['rate'] = df['close']
            else:
                raise ValueError(f"Missing 'rate' or 'close' column in {file_path}")

        # Validate date column
        if 'date' not in df.columns:
            raise ValueError(f"Missing 'date' column in {file_path}")

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df[['date', 'rate']].copy()  # Keep only needed columns
        df = df.set_index('date')

        # Store rates
        self.rates[currency_pair] = df

    def load_rates_directory(self, directory: Path) -> None:
        """
        Load all currency rate files from a directory.

        Supported filename formats:
        - XXXYYY.csv (e.g., USDGBP.csv, EURGBP.csv)
        - XXXYYY_suffix.csv (e.g., GBPUSD_weekly.csv, GBPEUR_daily.csv)

        Args:
            directory: Directory containing currency rate CSV files
        """
        if not directory.exists():
            raise FileNotFoundError(f"Currency rates directory not found: {directory}")

        for file_path in directory.glob("*.csv"):
            # Extract currency pair from filename
            filename = file_path.stem.upper()

            # Remove common suffixes like _weekly, _daily, _monthly
            for suffix in ['_WEEKLY', '_DAILY', '_MONTHLY', '_W', '_D', '_M']:
                if filename.endswith(suffix):
                    filename = filename[:-len(suffix)]
                    break

            # Now parse the 6-character currency pair
            if len(filename) >= 6:
                # Take first 6 characters as currency pair
                pair_str = filename[:6]
                from_currency = pair_str[:3]
                to_currency = pair_str[3:6]
                currency_pair = f"{from_currency}/{to_currency}"

                try:
                    self.load_rates(currency_pair, file_path)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

    def get_rate(self, from_currency: str, to_currency: str,
                 date: datetime) -> Optional[float]:
        """
        Get exchange rate for a specific date.

        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date for the exchange rate

        Returns:
            Exchange rate, or None if not available
        """
        # Same currency = 1.0
        if from_currency == to_currency:
            return 1.0

        # Convert to base currency if needed
        if to_currency != self.base_currency:
            raise ValueError(
                f"Target currency must be {self.base_currency}, got {to_currency}"
            )

        currency_pair = f"{from_currency}/{to_currency}"

        if currency_pair not in self.rates:
            return None

        df = self.rates[currency_pair]

        # Find the rate for this date or the most recent prior date
        # This handles weekends and holidays
        valid_dates = df.index[df.index <= date]
        if len(valid_dates) == 0:
            return None

        nearest_date = valid_dates[-1]
        return df.loc[nearest_date, 'rate']

    def convert(self, amount: float, from_currency: str,
                to_currency: str, date: datetime) -> Optional[float]:
        """
        Convert amount from one currency to another.

        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code (should be base currency)
            date: Date for the exchange rate

        Returns:
            Converted amount in target currency, or None if rate not available
        """
        rate = self.get_rate(from_currency, to_currency, date)
        if rate is None:
            return None

        return amount * rate

    def has_rate(self, currency: str) -> bool:
        """
        Check if conversion rates are available for a currency.

        Args:
            currency: Currency code

        Returns:
            True if rates are available or if currency is base currency
        """
        if currency == self.base_currency:
            return True

        currency_pair = f"{currency}/{self.base_currency}"
        return currency_pair in self.rates

    def get_available_currencies(self) -> list:
        """
        Get list of currencies with available conversion rates.

        Returns:
            List of currency codes
        """
        currencies = {self.base_currency}
        for pair in self.rates.keys():
            from_curr = pair.split('/')[0]
            currencies.add(from_curr)
        return sorted(list(currencies))
