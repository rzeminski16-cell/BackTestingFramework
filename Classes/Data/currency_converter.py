"""
Currency conversion module for handling multi-currency backtesting.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class MissingFXRateError(ValueError):
    """Raised when an FX rate required for conversion is not available."""
    pass


class CurrencyConverter:
    """
    Handles currency conversions for backtesting with time-series exchange rates.

    The base currency is always GBP (currency of account). Exchange rates can be
    supplied in either direction and are used automatically:

    - ``XXX/GBP`` (e.g. ``USD/GBP`` = 0.79 means 1 USD = 0.79 GBP), or
    - ``GBP/XXX`` (e.g. ``GBP/USD`` = 1.27 means 1 GBP = 1.27 USD).

    When only the inverse pair is loaded (e.g. you have ``GBP/USD`` data but need
    to convert USD into GBP) the converter inverts the rate (``1 / 1.27``)
    automatically, so the FX is applied correctly regardless of how the source
    series is quoted.
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

    def _rate_on_date(self, df: pd.DataFrame, date: datetime) -> Optional[float]:
        """
        Look up the rate for a date, using the nearest available value.

        Lookup order:

        1. The most recent entry on or before ``date``. This carries the last
           known rate forward across weekends, holidays, or any date that falls
           after the series ends.
        2. If ``date`` precedes the first entry (e.g. the file does not go back
           far enough - a 2005 backtest against a series that only starts in
           2007), fall back to the earliest entry on record - the next best
           value - so conversion still proceeds instead of failing.

        Only returns None when the series has no entries at all.

        Args:
            df: Rate DataFrame indexed by date with a 'rate' column
            date: Date for the exchange rate

        Returns:
            Rate value, or None only if ``df`` is empty
        """
        if df.empty:
            return None

        on_or_before = df.index[df.index <= date]
        if len(on_or_before) > 0:
            nearest_date = on_or_before[-1]
            return df.loc[nearest_date, 'rate']

        # ``date`` is before the first available entry: use the earliest rate
        # on record so conversion still proceeds with the next best value.
        return df.loc[df.index[0], 'rate']

    def get_rate(self, from_currency: str, to_currency: str,
                 date: datetime) -> Optional[float]:
        """
        Get exchange rate to convert ``from_currency`` into ``to_currency``.

        The rate can be sourced from a directly quoted pair (``from/to``) or from
        the inverse pair (``to/from``), in which case it is inverted. This means
        a ``GBP/USD`` series can convert USD into GBP and vice versa.

        When the requested ``date`` has no exact entry, the nearest available
        rate is used: the most recent entry on or before the date, or - when
        the date is earlier than the whole series - the earliest entry on
        record. A rate is therefore only unavailable when no suitable pair is
        loaded at all.

        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date for the exchange rate

        Returns:
            Exchange rate (units of ``to_currency`` per unit of ``from_currency``),
            or None if no suitable pair is loaded for this currency
        """
        # Same currency = 1.0
        if from_currency == to_currency:
            return 1.0

        # Direct pair: rate is already units of to_currency per from_currency.
        direct_pair = f"{from_currency}/{to_currency}"
        if direct_pair in self.rates:
            return self._rate_on_date(self.rates[direct_pair], date)

        # Inverse pair: e.g. we have GBP/USD but need USD->GBP. The stored rate
        # is units of from_currency per to_currency, so invert it.
        inverse_pair = f"{to_currency}/{from_currency}"
        if inverse_pair in self.rates:
            inverse_rate = self._rate_on_date(self.rates[inverse_pair], date)
            if inverse_rate is None or inverse_rate == 0:
                return None
            return 1.0 / inverse_rate

        return None

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

        A rate is considered available if either the direct pair
        (``currency/base``) or the inverse pair (``base/currency``) is loaded,
        since the converter can invert when needed.

        Args:
            currency: Currency code

        Returns:
            True if rates are available or if currency is base currency
        """
        if currency == self.base_currency:
            return True

        direct_pair = f"{currency}/{self.base_currency}"
        inverse_pair = f"{self.base_currency}/{currency}"
        return direct_pair in self.rates or inverse_pair in self.rates

    def get_available_currencies(self) -> list:
        """
        Get list of currencies with available conversion rates.

        Both sides of every loaded pair are reported, since the converter can
        invert a pair when only the opposite direction is supplied.

        Returns:
            List of currency codes
        """
        currencies = {self.base_currency}
        for pair in self.rates.keys():
            from_curr, to_curr = pair.split('/')
            currencies.add(from_curr)
            currencies.add(to_curr)
        return sorted(list(currencies))
