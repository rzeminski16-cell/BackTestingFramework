"""
Alpha Vantage API configuration.

This module handles configuration for the Alpha Vantage premium data service.
Create a config file at the specified path or set environment variables.

Configuration File Format (JSON):
{
    "api_key": "YOUR_ALPHA_VANTAGE_API_KEY",
    "rate_limit_per_minute": 75,
    "cache_directory": "cache/alpha_vantage",
    "cache_expiry_days": 1
}

Environment Variables (alternative to config file):
    ALPHA_VANTAGE_API_KEY: Your API key
    ALPHA_VANTAGE_RATE_LIMIT: Calls per minute (default: 75 for premium)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Default config file locations (checked in order)
DEFAULT_CONFIG_PATHS = [
    Path("alpha_vantage_config.json"),
    Path("config/alpha_vantage_config.json"),
    Path.home() / ".alpha_vantage_config.json",
]


@dataclass
class AlphaVantageConfig:
    """
    Configuration for Alpha Vantage API access.

    Attributes
    ----------
    api_key : str
        Your Alpha Vantage API key (required)

    rate_limit_per_minute : int
        API calls allowed per minute. Premium tiers:
        - 75 calls/min (lowest premium tier)
        - 150 calls/min
        - 300 calls/min
        - 600 calls/min
        - 1200 calls/min (highest tier)
        Default: 75

    cache_directory : Path
        Directory for caching API responses
        Default: cache/alpha_vantage

    cache_expiry_days : int
        How long to cache responses (in days)
        Default: 1 (financial data updates daily)

    request_timeout : int
        Timeout for API requests in seconds
        Default: 30

    max_retries : int
        Maximum retries for failed requests
        Default: 3

    retry_delay : float
        Initial delay between retries (seconds), doubles each retry
        Default: 2.0
    """

    api_key: str
    rate_limit_per_minute: int = 75
    cache_directory: Path = field(default_factory=lambda: Path("cache/alpha_vantage"))
    cache_expiry_days: int = 1
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

    # Base URL for Alpha Vantage API
    base_url: str = "https://www.alphavantage.co/query"

    def __post_init__(self):
        """Validate configuration and convert types."""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")

        if isinstance(self.cache_directory, str):
            self.cache_directory = Path(self.cache_directory)

        # Create cache directory if it doesn't exist
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'api_key': self.api_key,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'cache_directory': str(self.cache_directory),
            'cache_expiry_days': self.cache_expiry_days,
            'request_timeout': self.request_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
        }

    def save(self, file_path: Path):
        """Save configuration to a JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'AlphaVantageConfig':
        """Create config from dictionary."""
        return cls(
            api_key=data['api_key'],
            rate_limit_per_minute=data.get('rate_limit_per_minute', 75),
            cache_directory=Path(data.get('cache_directory', 'cache/alpha_vantage')),
            cache_expiry_days=data.get('cache_expiry_days', 1),
            request_timeout=data.get('request_timeout', 30),
            max_retries=data.get('max_retries', 3),
            retry_delay=data.get('retry_delay', 2.0),
        )

    @classmethod
    def from_file(cls, file_path: Optional[Path] = None) -> 'AlphaVantageConfig':
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to config file. If None, searches default locations.

        Returns:
            AlphaVantageConfig instance

        Raises:
            FileNotFoundError: If no config file found
            ValueError: If config file is invalid
        """
        if file_path is not None:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")

            with open(file_path) as f:
                data = json.load(f)
            return cls.from_dict(data)

        # Search default locations
        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                return cls.from_dict(data)

        raise FileNotFoundError(
            f"No Alpha Vantage config file found. Searched:\n"
            f"  {chr(10).join(str(p) for p in DEFAULT_CONFIG_PATHS)}\n"
            f"Create a config file or use from_environment()."
        )

    @classmethod
    def from_environment(cls) -> 'AlphaVantageConfig':
        """
        Load configuration from environment variables.

        Environment Variables:
            ALPHA_VANTAGE_API_KEY: API key (required)
            ALPHA_VANTAGE_RATE_LIMIT: Calls per minute
            ALPHA_VANTAGE_CACHE_DIR: Cache directory path
            ALPHA_VANTAGE_CACHE_EXPIRY: Cache expiry in days

        Returns:
            AlphaVantageConfig instance

        Raises:
            ValueError: If API key not set
        """
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY environment variable not set. "
                "Set it or use a config file."
            )

        return cls(
            api_key=api_key,
            rate_limit_per_minute=int(os.environ.get('ALPHA_VANTAGE_RATE_LIMIT', '75')),
            cache_directory=Path(os.environ.get('ALPHA_VANTAGE_CACHE_DIR', 'cache/alpha_vantage')),
            cache_expiry_days=int(os.environ.get('ALPHA_VANTAGE_CACHE_EXPIRY', '1')),
        )

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> 'AlphaVantageConfig':
        """
        Load configuration from file or environment.

        Tries in order:
        1. Specified file path
        2. Default config file locations
        3. Environment variables

        Args:
            file_path: Optional path to config file

        Returns:
            AlphaVantageConfig instance
        """
        # Try file first
        try:
            return cls.from_file(file_path)
        except FileNotFoundError:
            pass

        # Fall back to environment
        try:
            return cls.from_environment()
        except ValueError:
            pass

        raise ValueError(
            "Could not load Alpha Vantage configuration.\n"
            "Either create a config file or set ALPHA_VANTAGE_API_KEY environment variable.\n"
            f"Config file locations searched: {[str(p) for p in DEFAULT_CONFIG_PATHS]}"
        )


def create_sample_config(output_path: Path = Path("alpha_vantage_config.json")):
    """
    Create a sample configuration file.

    Args:
        output_path: Where to save the sample config
    """
    sample = {
        "api_key": "YOUR_API_KEY_HERE",
        "rate_limit_per_minute": 75,
        "cache_directory": "cache/alpha_vantage",
        "cache_expiry_days": 1,
        "request_timeout": 30,
        "max_retries": 3,
        "retry_delay": 2.0,
        "_comment": "Replace YOUR_API_KEY_HERE with your Alpha Vantage premium API key"
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(sample, f, indent=2)

    print(f"Sample config created at: {output_path}")
    print("Edit this file and replace YOUR_API_KEY_HERE with your API key.")
