"""
Alpha Vantage API client wrapper for backward compatibility.

This module wraps the main DataCollection AlphaVantageClient to provide
a simplified interface for the fundamental data fetcher.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from Classes.DataCollection.alpha_vantage_client import (
    AlphaVantageClient as DataCollectionClient,
    APIResponse,
)
from Classes.DataCollection.config import APIConfig, CacheConfig

from .alpha_vantage_config import AlphaVantageConfig


logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    Wrapper around DataCollection's AlphaVantageClient for backward compatibility.

    Provides a simplified interface that returns Dict instead of APIResponse.
    """

    def __init__(self, config: AlphaVantageConfig):
        """
        Initialize the client.

        Args:
            config: Alpha Vantage configuration
        """
        self.config = config

        # Convert to DataCollection config types
        api_config = APIConfig(
            api_key=config.api_key,
            rate_limit_per_minute=config.rate_limit_per_minute,
            timeout_seconds=config.request_timeout,
            max_retries=config.max_retries,
            retry_backoff_base=config.retry_delay,
        )

        cache_config = CacheConfig(
            cache_dir=str(config.cache_directory),
            expiry_hours=config.cache_expiry_days * 24,
        )

        # Create the underlying client
        self._client = DataCollectionClient(api_config, cache_config)

        logger.info(f"Alpha Vantage client initialized (rate limit: {config.rate_limit_per_minute}/min)")

    def _extract_data(self, response: APIResponse) -> Dict[str, Any]:
        """Extract data from APIResponse, raising on error."""
        if not response.success:
            raise Exception(response.error_message or "API request failed")
        return response.data or {}

    def get_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company overview with key fundamental figures."""
        response = self._client.get_company_overview(symbol)
        return self._extract_data(response)

    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Get income statement data (annual and quarterly)."""
        response = self._client.get_income_statement(symbol)
        return self._extract_data(response)

    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """Get balance sheet data (annual and quarterly)."""
        response = self._client.get_balance_sheet(symbol)
        return self._extract_data(response)

    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Get cash flow statement data (annual and quarterly)."""
        response = self._client.get_cash_flow(symbol)
        return self._extract_data(response)

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get earnings data (annual and quarterly)."""
        response = self._client.get_earnings(symbol)
        return self._extract_data(response)

    def get_daily_prices(self, symbol: str, outputsize: str = 'full') -> Dict[str, Any]:
        """Get daily adjusted price data."""
        response = self._client.get_daily_prices(symbol, adjusted=True, outputsize=outputsize)
        return self._extract_data(response)

    def get_all_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all fundamental data for a symbol in one call.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with all fundamental data
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
        """Clear cached responses."""
        self._client.cache.clear(older_than_days)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return self._client.get_cache_info()
