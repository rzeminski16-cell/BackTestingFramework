"""
Alpha Vantage API Client with Rate Limiting, Caching, and Retry Logic.

Provides a robust interface to the Alpha Vantage API with:
- Rate limiting (configurable, default 75 requests/minute for premium)
- Response caching with configurable expiry
- Automatic retry with exponential backoff
- Comprehensive error handling
- Integration with the logging system
"""

import time
import json
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque

from .config import APIConfig, CacheConfig
from .logging_manager import SessionLogger, APILogEntry, ErrorLogEntry, LogLevel


class APIEndpoint(Enum):
    """Alpha Vantage API endpoints."""
    # Time Series
    TIME_SERIES_DAILY = "TIME_SERIES_DAILY"
    TIME_SERIES_DAILY_ADJUSTED = "TIME_SERIES_DAILY_ADJUSTED"
    TIME_SERIES_WEEKLY = "TIME_SERIES_WEEKLY"
    TIME_SERIES_WEEKLY_ADJUSTED = "TIME_SERIES_WEEKLY_ADJUSTED"

    # Technical Indicators
    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"
    DEMA = "DEMA"
    TEMA = "TEMA"
    TRIMA = "TRIMA"
    KAMA = "KAMA"
    MAMA = "MAMA"
    VWAP = "VWAP"
    T3 = "T3"
    MACD = "MACD"
    MACDEXT = "MACDEXT"
    STOCH = "STOCH"
    STOCHF = "STOCHF"
    RSI = "RSI"
    STOCHRSI = "STOCHRSI"
    WILLR = "WILLR"
    ADX = "ADX"
    ADXR = "ADXR"
    APO = "APO"
    PPO = "PPO"
    MOM = "MOM"
    BOP = "BOP"
    CCI = "CCI"
    CMO = "CMO"
    ROC = "ROC"
    ROCR = "ROCR"
    AROON = "AROON"
    AROONOSC = "AROONOSC"
    MFI = "MFI"
    TRIX = "TRIX"
    ULTOSC = "ULTOSC"
    DX = "DX"
    MINUS_DI = "MINUS_DI"
    PLUS_DI = "PLUS_DI"
    MINUS_DM = "MINUS_DM"
    PLUS_DM = "PLUS_DM"
    BBANDS = "BBANDS"
    MIDPOINT = "MIDPOINT"
    MIDPRICE = "MIDPRICE"
    SAR = "SAR"
    TRANGE = "TRANGE"
    ATR = "ATR"
    NATR = "NATR"
    AD = "AD"
    ADOSC = "ADOSC"
    OBV = "OBV"
    HT_TRENDLINE = "HT_TRENDLINE"
    HT_SINE = "HT_SINE"
    HT_TRENDMODE = "HT_TRENDMODE"
    HT_DCPERIOD = "HT_DCPERIOD"
    HT_DCPHASE = "HT_DCPHASE"
    HT_PHASOR = "HT_PHASOR"

    # Fundamentals
    OVERVIEW = "OVERVIEW"
    INCOME_STATEMENT = "INCOME_STATEMENT"
    BALANCE_SHEET = "BALANCE_SHEET"
    CASH_FLOW = "CASH_FLOW"
    EARNINGS = "EARNINGS"

    # Insider Transactions
    INSIDER_TRANSACTIONS = "INSIDER_TRANSACTIONS"

    # Forex
    FX_WEEKLY = "FX_WEEKLY"
    FX_DAILY = "FX_DAILY"

    # Options
    HISTORICAL_OPTIONS = "HISTORICAL_OPTIONS"


@dataclass
class APIResponse:
    """Wrapper for API response data."""
    success: bool
    data: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    from_cache: bool = False
    response_time: float = 0.0
    retry_count: int = 0


class RateLimiter:
    """
    Token bucket rate limiter.

    Ensures API calls don't exceed the rate limit.
    """

    def __init__(self, calls_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self._call_times: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Acquire permission to make an API call.

        Returns:
            Wait time in seconds before call can be made
        """
        with self._lock:
            now = time.time()

            # Remove calls older than 1 minute
            while self._call_times and now - self._call_times[0] > 60:
                self._call_times.popleft()

            # Check if we've hit the rate limit
            if len(self._call_times) >= self.calls_per_minute:
                # Wait until the oldest call expires
                wait_time = 60 - (now - self._call_times[0]) + 0.1
                return max(0, wait_time)

            # Record this call
            self._call_times.append(now)
            return 0

    def wait_and_acquire(self):
        """Wait if necessary and acquire permission."""
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)
            # Re-acquire after waiting
            self.acquire()


class CacheManager:
    """
    Manages caching of API responses.

    Stores responses on disk with configurable expiry.
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        # Sort params for consistent hashing
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True)
        key_str = f"{endpoint}:{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available and not expired.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Cached data or None if not available/expired
        """
        if not self.config.enabled:
            return None

        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        with self._lock:
            if not cache_path.exists():
                return None

            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)

                # Check expiry
                cached_time = datetime.fromisoformat(cached['timestamp'])
                expiry = cached_time + timedelta(hours=self.config.cache_expiry_hours)

                if datetime.now() > expiry:
                    # Cache expired
                    cache_path.unlink()
                    return None

                return cached['data']

            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupted cache file
                cache_path.unlink(missing_ok=True)
                return None

    def set(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]):
        """
        Store response in cache.

        Args:
            endpoint: API endpoint
            params: Request parameters
            data: Response data to cache
        """
        if not self.config.enabled:
            return

        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        with self._lock:
            cache_entry = {
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'params': params,
                'data': data
            }

            with open(cache_path, 'w') as f:
                json.dump(cache_entry, f)

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

    def get_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        return total_bytes / (1024 * 1024)

    def cleanup_expired(self):
        """Remove expired cache entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)

                    cached_time = datetime.fromisoformat(cached['timestamp'])
                    expiry = cached_time + timedelta(hours=self.config.cache_expiry_hours)

                    if datetime.now() > expiry:
                        cache_file.unlink()

                except (json.JSONDecodeError, KeyError, ValueError):
                    cache_file.unlink(missing_ok=True)


class AlphaVantageClient:
    """
    Alpha Vantage API client with rate limiting, caching, and retry logic.

    Provides methods for all data collection endpoints required by the framework.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_config: APIConfig,
        cache_config: CacheConfig,
        logger: Optional[SessionLogger] = None
    ):
        """
        Initialize the API client.

        Args:
            api_config: API configuration
            cache_config: Cache configuration
            logger: Optional session logger for logging API activity
        """
        self.api_config = api_config
        self.cache_config = cache_config
        self.logger = logger

        self.rate_limiter = RateLimiter(api_config.rate_limit_per_minute)
        self.cache = CacheManager(cache_config)

        # Progress callback for UI updates
        self._progress_callback: Optional[Callable[[str], None]] = None

        # Cancellation flag
        self._cancelled = False

    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def cancel(self):
        """Cancel ongoing operations."""
        self._cancelled = True

    def reset_cancel(self):
        """Reset cancellation flag."""
        self._cancelled = False

    def _notify_progress(self, message: str):
        """Notify progress callback if set."""
        if self._progress_callback:
            self._progress_callback(message)

    def validate_api_key(self) -> tuple:
        """
        Validate the API key by making a test request.

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            response = self._make_request(
                APIEndpoint.TIME_SERIES_DAILY,
                {"symbol": "IBM", "outputsize": "compact"},
                use_cache=False
            )

            if response.success:
                return True, "API key validated successfully"

            if "Invalid API call" in (response.error_message or ""):
                return False, "Invalid API key"

            return False, response.error_message or "Unknown error"

        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def _make_request(
        self,
        endpoint: APIEndpoint,
        params: Dict[str, Any],
        use_cache: bool = True
    ) -> APIResponse:
        """
        Make an API request with rate limiting, caching, and retry logic.

        Args:
            endpoint: API endpoint
            params: Request parameters
            use_cache: Whether to use caching

        Returns:
            APIResponse with result data
        """
        if self._cancelled:
            return APIResponse(
                success=False,
                data=None,
                error_message="Operation cancelled"
            )

        start_time = time.time()
        endpoint_str = endpoint.value

        # Add function and API key to params
        full_params = {
            "function": endpoint_str,
            "apikey": self.api_config.api_key,
            **params
        }

        # Check cache first
        if use_cache:
            cached_data = self.cache.get(endpoint_str, params)
            if cached_data:
                elapsed = time.time() - start_time

                if self.logger:
                    self.logger.log_api_call(APILogEntry(
                        timestamp=datetime.now(),
                        method="GET",
                        endpoint=endpoint_str,
                        symbol=params.get("symbol"),
                        status_code=200,
                        response_size_bytes=len(json.dumps(cached_data)),
                        cache_hit=True,
                        duration_seconds=elapsed
                    ))

                return APIResponse(
                    success=True,
                    data=cached_data,
                    from_cache=True,
                    response_time=elapsed
                )

        # Make the actual request with retry logic
        retry_count = 0
        last_error = None

        while retry_count <= self.api_config.max_retries:
            if self._cancelled:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message="Operation cancelled"
                )

            try:
                # Wait for rate limiter
                self.rate_limiter.wait_and_acquire()

                # Make request
                response = requests.get(
                    self.BASE_URL,
                    params=full_params,
                    timeout=self.api_config.timeout_seconds
                )

                elapsed = time.time() - start_time

                # Check for rate limit response
                if response.status_code == 429:
                    wait_time = self._calculate_backoff(retry_count)
                    self._notify_progress(f"Rate limit hit, waiting {wait_time:.0f}s...")

                    if self.logger:
                        self.logger.log_warning(
                            "API Rate Limit",
                            f"Rate limit exceeded, retrying in {wait_time:.0f}s",
                            endpoint=endpoint_str,
                            symbol=params.get("symbol")
                        )

                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                # Check for errors
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    retry_count += 1
                    time.sleep(self._calculate_backoff(retry_count))
                    continue

                # Parse response
                data = response.json()

                # Check for API error messages
                if "Error Message" in data:
                    return APIResponse(
                        success=False,
                        data=None,
                        error_message=data["Error Message"],
                        status_code=response.status_code,
                        response_time=elapsed
                    )

                if "Note" in data and "rate limit" in data["Note"].lower():
                    wait_time = self._calculate_backoff(retry_count)
                    self._notify_progress(f"Rate limit warning, waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                # Success - cache and return
                if use_cache:
                    self.cache.set(endpoint_str, params, data)

                if self.logger:
                    self.logger.log_api_call(APILogEntry(
                        timestamp=datetime.now(),
                        method="GET",
                        endpoint=endpoint_str,
                        symbol=params.get("symbol"),
                        status_code=response.status_code,
                        response_size_bytes=len(response.content),
                        cache_hit=False,
                        duration_seconds=elapsed,
                        retry_count=retry_count
                    ))

                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    response_time=elapsed,
                    retry_count=retry_count
                )

            except requests.Timeout:
                last_error = f"Request timeout after {self.api_config.timeout_seconds}s"
                retry_count += 1

                if self.logger:
                    self.logger.log_warning(
                        "API Timeout",
                        last_error,
                        endpoint=endpoint_str,
                        symbol=params.get("symbol"),
                        recovery_action=f"Retry {retry_count}/{self.api_config.max_retries}"
                    )

                time.sleep(self._calculate_backoff(retry_count))

            except requests.RequestException as e:
                last_error = f"Request error: {str(e)}"
                retry_count += 1

                if self.logger:
                    self.logger.log_warning(
                        "API Request Error",
                        last_error,
                        endpoint=endpoint_str,
                        symbol=params.get("symbol"),
                        recovery_action=f"Retry {retry_count}/{self.api_config.max_retries}"
                    )

                time.sleep(self._calculate_backoff(retry_count))

        # All retries exhausted
        elapsed = time.time() - start_time

        if self.logger:
            self.logger.log_error(ErrorLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.ERROR,
                error_type="API Request Failed",
                message=last_error or "Unknown error after max retries",
                endpoint=endpoint_str,
                symbol=params.get("symbol"),
                impact="Data not collected for this request"
            ))

        return APIResponse(
            success=False,
            data=None,
            error_message=last_error or "Max retries exceeded",
            response_time=elapsed,
            retry_count=retry_count
        )

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff time."""
        return min(
            self.api_config.retry_backoff_base ** retry_count,
            60  # Max 60 seconds
        )

    # === Daily Time Series ===

    def get_daily_prices(
        self,
        symbol: str,
        adjusted: bool = True,
        outputsize: str = "full"
    ) -> APIResponse:
        """
        Get daily OHLCV data.

        Args:
            symbol: Stock ticker symbol
            adjusted: Whether to use adjusted prices
            outputsize: "compact" (100 days) or "full" (20+ years)

        Returns:
            APIResponse with time series data
        """
        endpoint = APIEndpoint.TIME_SERIES_DAILY_ADJUSTED if adjusted else APIEndpoint.TIME_SERIES_DAILY
        params = {
            "symbol": symbol,
            "outputsize": outputsize
        }
        return self._make_request(endpoint, params)

    # === Weekly Time Series ===

    def get_weekly_prices(
        self,
        symbol: str,
        adjusted: bool = True
    ) -> APIResponse:
        """
        Get weekly OHLCV data.

        Args:
            symbol: Stock ticker symbol
            adjusted: Whether to use adjusted prices

        Returns:
            APIResponse with time series data
        """
        endpoint = APIEndpoint.TIME_SERIES_WEEKLY_ADJUSTED if adjusted else APIEndpoint.TIME_SERIES_WEEKLY
        params = {"symbol": symbol}
        return self._make_request(endpoint, params)

    # === Technical Indicators ===

    def get_indicator(
        self,
        indicator: str,
        symbol: str,
        interval: str = "daily",
        time_period: Optional[int] = None,
        series_type: str = "close",
        **kwargs
    ) -> APIResponse:
        """
        Get technical indicator data.

        Args:
            indicator: Indicator name (e.g., "SMA", "RSI", "MACD")
            symbol: Stock ticker symbol
            interval: Time interval (daily, weekly, monthly)
            time_period: Period for the indicator (if applicable)
            series_type: Price series to use (close, open, high, low)
            **kwargs: Additional indicator-specific parameters

        Returns:
            APIResponse with indicator data
        """
        try:
            endpoint = APIEndpoint[indicator]
        except KeyError:
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Unknown indicator: {indicator}"
            )

        params = {
            "symbol": symbol,
            "interval": interval,
            "series_type": series_type,
            **kwargs
        }

        if time_period is not None:
            params["time_period"] = time_period

        return self._make_request(endpoint, params)

    # === Fundamental Data ===

    def get_company_overview(self, symbol: str) -> APIResponse:
        """Get company overview with fundamental metrics."""
        return self._make_request(
            APIEndpoint.OVERVIEW,
            {"symbol": symbol}
        )

    def get_income_statement(self, symbol: str) -> APIResponse:
        """Get income statement data."""
        return self._make_request(
            APIEndpoint.INCOME_STATEMENT,
            {"symbol": symbol}
        )

    def get_balance_sheet(self, symbol: str) -> APIResponse:
        """Get balance sheet data."""
        return self._make_request(
            APIEndpoint.BALANCE_SHEET,
            {"symbol": symbol}
        )

    def get_cash_flow(self, symbol: str) -> APIResponse:
        """Get cash flow statement data."""
        return self._make_request(
            APIEndpoint.CASH_FLOW,
            {"symbol": symbol}
        )

    def get_earnings(self, symbol: str) -> APIResponse:
        """Get earnings data."""
        return self._make_request(
            APIEndpoint.EARNINGS,
            {"symbol": symbol}
        )

    # === Insider Transactions ===

    def get_insider_transactions(self, symbol: str) -> APIResponse:
        """Get insider trading transactions."""
        return self._make_request(
            APIEndpoint.INSIDER_TRANSACTIONS,
            {"symbol": symbol}
        )

    # === Forex ===

    def get_forex_weekly(
        self,
        from_currency: str,
        to_currency: str
    ) -> APIResponse:
        """
        Get weekly forex data.

        Args:
            from_currency: Base currency (e.g., "GBP")
            to_currency: Quote currency (e.g., "USD")

        Returns:
            APIResponse with forex data
        """
        return self._make_request(
            APIEndpoint.FX_WEEKLY,
            {
                "from_symbol": from_currency,
                "to_symbol": to_currency
            }
        )

    def get_forex_daily(
        self,
        from_currency: str,
        to_currency: str,
        outputsize: str = "full"
    ) -> APIResponse:
        """Get daily forex data."""
        return self._make_request(
            APIEndpoint.FX_DAILY,
            {
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "outputsize": outputsize
            }
        )

    # === Options ===

    def get_historical_options(
        self,
        symbol: str,
        date: Optional[str] = None
    ) -> APIResponse:
        """
        Get historical options data.

        Args:
            symbol: Stock ticker symbol
            date: Optional specific date (YYYY-MM-DD format)

        Returns:
            APIResponse with options chain data
        """
        params = {"symbol": symbol}
        if date:
            params["date"] = date

        return self._make_request(
            APIEndpoint.HISTORICAL_OPTIONS,
            params
        )

    # === Batch Operations ===

    def get_daily_with_indicators(
        self,
        symbol: str,
        indicators: List[Dict[str, Any]],
        adjusted: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, APIResponse]:
        """
        Get daily prices along with multiple indicators.

        Args:
            symbol: Stock ticker symbol
            indicators: List of indicator configs with name and parameters
            adjusted: Whether to use adjusted prices
            progress_callback: Callback for progress updates (message, current, total)

        Returns:
            Dict mapping data type to APIResponse
        """
        results = {}
        total = 1 + len(indicators)
        current = 0

        # Get price data first
        if progress_callback:
            progress_callback(f"Fetching {symbol} daily prices...", current, total)

        results['prices'] = self.get_daily_prices(symbol, adjusted)
        current += 1

        if self._cancelled:
            return results

        # Get each indicator
        for ind_config in indicators:
            if self._cancelled:
                break

            ind_name = ind_config.get('name', ind_config.get('indicator'))
            period = ind_config.get('period') or ind_config.get('time_period')

            if progress_callback:
                period_str = f"({period})" if period else ""
                progress_callback(f"Fetching {symbol} {ind_name}{period_str}...", current, total)

            # Build kwargs from config
            kwargs = {k: v for k, v in ind_config.items() if k not in ['name', 'indicator']}

            results[f"{ind_name}_{period}" if period else ind_name] = self.get_indicator(
                indicator=ind_name,
                symbol=symbol,
                time_period=period,
                **{k: v for k, v in kwargs.items() if k != 'period' and k != 'time_period'}
            )
            current += 1

        return results

    def clear_cache(self):
        """Clear all cached API responses."""
        self.cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.cache_config.enabled,
            "size_mb": self.cache.get_size_mb(),
            "max_size_mb": self.cache_config.max_cache_size_mb,
            "expiry_hours": self.cache_config.cache_expiry_hours,
            "cache_dir": str(self.cache_config.cache_dir)
        }
