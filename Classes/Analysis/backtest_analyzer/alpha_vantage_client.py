"""
Alpha Vantage API client with caching and rate limiting.

This module provides a robust client for the Alpha Vantage API with:
- Automatic rate limiting based on subscription tier
- Response caching to minimize API calls
- Retry logic with exponential backoff
- Detailed logging for debugging
"""

import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import deque
import urllib.request
import urllib.error
import urllib.parse

from .alpha_vantage_config import AlphaVantageConfig


# Configure logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.

    Tracks API calls within a time window and blocks if limit exceeded.
    """

    def __init__(self, calls_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.window_seconds = 60
        self.call_times: deque = deque()

    def wait_if_needed(self):
        """Block if rate limit would be exceeded, then record the call."""
        now = time.time()

        # Remove old calls outside the window
        while self.call_times and self.call_times[0] < now - self.window_seconds:
            self.call_times.popleft()

        # Check if we need to wait
        if len(self.call_times) >= self.calls_per_minute:
            # Calculate how long to wait
            oldest_call = self.call_times[0]
            wait_time = oldest_call + self.window_seconds - now + 0.1  # Add small buffer

            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        # Record this call
        self.call_times.append(time.time())


class ResponseCache:
    """
    File-based cache for API responses.

    Caches responses to disk with configurable expiry.
    """

    def __init__(self, cache_dir: Path, expiry_days: int = 1):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cached responses
            expiry_days: How long to keep cached responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_days = expiry_days

    def _get_cache_key(self, function: str, params: Dict[str, str]) -> str:
        """Generate a unique cache key for the request."""
        # Sort params for consistent hashing
        param_str = json.dumps(params, sort_keys=True)
        key_str = f"{function}:{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, function: str, params: Dict[str, str]) -> Optional[Dict]:
        """
        Get cached response if available and not expired.

        Args:
            function: API function name
            params: Request parameters

        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._get_cache_key(function, params)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                cached = json.load(f)

            # Check expiry
            cached_time = datetime.fromisoformat(cached['_cached_at'])
            if datetime.now() - cached_time > timedelta(days=self.expiry_days):
                logger.debug(f"Cache expired for {function}")
                cache_path.unlink()  # Remove expired cache
                return None

            logger.debug(f"Cache hit for {function}")
            return cached['data']

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache file {cache_path}: {e}")
            cache_path.unlink()  # Remove invalid cache
            return None

    def set(self, function: str, params: Dict[str, str], data: Dict):
        """
        Cache a response.

        Args:
            function: API function name
            params: Request parameters
            data: Response data to cache
        """
        cache_key = self._get_cache_key(function, params)
        cache_path = self._get_cache_path(cache_key)

        cached = {
            '_cached_at': datetime.now().isoformat(),
            '_function': function,
            '_params': params,
            'data': data
        }

        with open(cache_path, 'w') as f:
            json.dump(cached, f, indent=2)

        logger.debug(f"Cached response for {function}")

    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cached responses.

        Args:
            older_than_days: Only clear items older than this. If None, clear all.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if older_than_days is not None:
                    with open(cache_file) as f:
                        cached = json.load(f)
                    cached_time = datetime.fromisoformat(cached['_cached_at'])
                    if datetime.now() - cached_time <= timedelta(days=older_than_days):
                        continue

                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error clearing cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cached responses")


class AlphaVantageClient:
    """
    Client for Alpha Vantage API with caching and rate limiting.

    Usage:
        config = AlphaVantageConfig.load()
        client = AlphaVantageClient(config)

        # Get company overview
        overview = client.get_overview("AAPL")

        # Get income statement
        income = client.get_income_statement("AAPL")
    """

    def __init__(self, config: AlphaVantageConfig):
        """
        Initialize the client.

        Args:
            config: Alpha Vantage configuration
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.cache = ResponseCache(config.cache_directory, config.cache_expiry_days)

        logger.info(f"Alpha Vantage client initialized (rate limit: {config.rate_limit_per_minute}/min)")

    def _make_request(self, function: str, params: Dict[str, str],
                      use_cache: bool = True) -> Dict[str, Any]:
        """
        Make an API request with caching and rate limiting.

        Args:
            function: API function name
            params: Additional parameters
            use_cache: Whether to use caching

        Returns:
            API response as dictionary

        Raises:
            Exception: If request fails after retries
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(function, params)
            if cached is not None:
                return cached

        # Build request URL
        request_params = {
            'function': function,
            'apikey': self.config.api_key,
            **params
        }
        url = f"{self.config.base_url}?{urllib.parse.urlencode(request_params)}"

        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Rate limit
                self.rate_limiter.wait_if_needed()

                # Make request
                logger.debug(f"API request: {function} {params}")

                request = urllib.request.Request(url)
                request.add_header('User-Agent', 'BacktestAnalyzer/1.0')

                with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                    data = json.loads(response.read().decode('utf-8'))

                # Check for API errors
                if 'Error Message' in data:
                    raise ValueError(f"API Error: {data['Error Message']}")

                if 'Note' in data and 'API call frequency' in data['Note']:
                    # Rate limit hit despite our tracking - wait and retry
                    logger.warning("API rate limit hit. Waiting 60 seconds...")
                    time.sleep(60)
                    continue

                # Cache successful response
                if use_cache:
                    self.cache.set(function, params, data)

                return data

            except urllib.error.URLError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except json.JSONDecodeError as e:
                last_error = e
                logger.error(f"Invalid JSON response: {e}")
                break

        raise Exception(f"Failed to fetch {function} after {self.config.max_retries} attempts: {last_error}")

    # =========================================================================
    # Fundamental Data Endpoints
    # =========================================================================

    def get_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview with key fundamental figures.

        Returns current values for:
        - EPS, PERatio, ForwardPE, PEGRatio
        - BookValue, PriceToBookRatio
        - MarketCapitalization, SharesOutstanding
        - OperatingMarginTTM, ProfitMargin
        - ReturnOnAssetsTTM, ReturnOnEquityTTM
        - DividendYield, DividendPerShare
        - Beta, AnalystTargetPrice
        - And more...

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Dictionary with company overview data
        """
        return self._make_request('OVERVIEW', {'symbol': symbol})

    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Get income statement data (annual and quarterly).

        Returns:
        - annualReports: List of annual income statements
        - quarterlyReports: List of quarterly income statements

        Each report includes:
        - fiscalDateEnding, reportedCurrency
        - totalRevenue, costOfRevenue, grossProfit
        - operatingIncome, operatingExpenses
        - netIncome, ebit, ebitda
        - And more...

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with income statement data
        """
        return self._make_request('INCOME_STATEMENT', {'symbol': symbol})

    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Get balance sheet data (annual and quarterly).

        Returns:
        - annualReports: List of annual balance sheets
        - quarterlyReports: List of quarterly balance sheets

        Each report includes:
        - fiscalDateEnding, reportedCurrency
        - totalAssets, totalLiabilities
        - totalShareholderEquity
        - shortTermDebt, longTermDebt
        - cashAndCashEquivalents
        - And more...

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with balance sheet data
        """
        return self._make_request('BALANCE_SHEET', {'symbol': symbol})

    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Get cash flow statement data (annual and quarterly).

        Returns:
        - annualReports: List of annual cash flow statements
        - quarterlyReports: List of quarterly cash flow statements

        Each report includes:
        - fiscalDateEnding, reportedCurrency
        - operatingCashflow
        - capitalExpenditures
        - dividendPayout
        - And more...

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with cash flow data
        """
        return self._make_request('CASH_FLOW', {'symbol': symbol})

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data (annual and quarterly).

        Returns:
        - annualEarnings: List of annual EPS
        - quarterlyEarnings: List of quarterly EPS with estimates

        Quarterly earnings include:
        - fiscalDateEnding, reportedDate
        - reportedEPS, estimatedEPS
        - surprise, surprisePercentage

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with earnings data
        """
        return self._make_request('EARNINGS', {'symbol': symbol})

    def get_daily_prices(self, symbol: str, outputsize: str = 'full') -> Dict[str, Any]:
        """
        Get daily adjusted price data.

        Args:
            symbol: Stock symbol
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)

        Returns:
            Dictionary with daily price data
        """
        return self._make_request(
            'TIME_SERIES_DAILY_ADJUSTED',
            {'symbol': symbol, 'outputsize': outputsize}
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_all_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all fundamental data for a symbol in one call.

        This makes multiple API calls but caches each response.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with all fundamental data:
            {
                'overview': {...},
                'income_statement': {...},
                'balance_sheet': {...},
                'cash_flow': {...},
                'earnings': {...}
            }
        """
        logger.info(f"Fetching all fundamental data for {symbol}...")

        return {
            'overview': self.get_overview(symbol),
            'income_statement': self.get_income_statement(symbol),
            'balance_sheet': self.get_balance_sheet(symbol),
            'cash_flow': self.get_cash_flow(symbol),
            'earnings': self.get_earnings(symbol),
        }

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached responses.

        Args:
            older_than_days: Only clear items older than this. If None, clear all.
        """
        self.cache.clear(older_than_days)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        cache_files = list(self.config.cache_directory.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'cache_directory': str(self.config.cache_directory),
            'num_cached_responses': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'expiry_days': self.config.cache_expiry_days,
        }
