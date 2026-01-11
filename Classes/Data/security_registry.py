"""
Security registry for tracking available securities and metadata.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class SecurityMetadata:
    """Metadata for a security."""
    symbol: str
    type: str = ""  # stock, etf, crypto, etc.
    sector: str = ""
    market_cap: str = ""  # large_cap, mid_cap, small_cap
    description: str = ""
    currency: str = ""  # USD, GBP, EUR, etc. - empty means unknown


class SecurityRegistry:
    """
    Registry of available securities with metadata.

    The base currency of the account is always GBP. Securities with different
    currencies will require exchange rate conversion during backtesting.
    """

    BASE_CURRENCY = "GBP"

    def __init__(self, metadata_file: Optional[Path] = None):
        """
        Initialize security registry.

        Args:
            metadata_file: Optional path to JSON file with security metadata
        """
        self.securities: Dict[str, SecurityMetadata] = {}
        self._warned_symbols: Set[str] = set()  # Track which symbols we've warned about

        if metadata_file and metadata_file.exists():
            self.load_metadata(metadata_file)

    def load_metadata(self, metadata_file: Path) -> None:
        """
        Load security metadata from JSON file.

        Args:
            metadata_file: Path to JSON metadata file
        """
        with open(metadata_file, 'r') as f:
            data = json.load(f)

        for symbol, info in data.items():
            self.securities[symbol] = SecurityMetadata(
                symbol=symbol,
                type=info.get('type', ''),
                sector=info.get('sector', ''),
                market_cap=info.get('market_cap', ''),
                description=info.get('description', ''),
                currency=info.get('currency', '')
            )

    def register(self, symbol: str, type: str = "",
                sector: str = "", market_cap: str = "",
                description: str = "", currency: str = "") -> None:
        """
        Register a security.

        Args:
            symbol: Security symbol
            type: Security type (stock, etf, crypto)
            sector: Sector
            market_cap: Market cap category (large_cap, mid_cap, small_cap)
            description: Description
            currency: Currency code (e.g., USD, EUR, GBP)
        """
        self.securities[symbol] = SecurityMetadata(
            symbol=symbol,
            type=type,
            sector=sector,
            market_cap=market_cap,
            description=description,
            currency=currency
        )

    def get_metadata(self, symbol: str, warn_on_missing_currency: bool = True) -> Optional[SecurityMetadata]:
        """
        Get metadata for a symbol.

        Args:
            symbol: Security symbol
            warn_on_missing_currency: If True, warn when currency is unknown

        Returns:
            SecurityMetadata or None if not found
        """
        metadata = self.securities.get(symbol)

        if metadata is None:
            # Symbol not in registry at all
            if symbol not in self._warned_symbols:
                warnings.warn(
                    f"Security '{symbol}' not found in security_metadata.json. "
                    f"Add it to config/security_metadata.json with currency info for accurate backtesting.",
                    UserWarning
                )
                self._warned_symbols.add(symbol)
            return None

        if warn_on_missing_currency and not metadata.currency:
            if symbol not in self._warned_symbols:
                warnings.warn(
                    f"Security '{symbol}' has no currency defined in security_metadata.json. "
                    f"Assuming USD. Update config/security_metadata.json for accurate currency conversion.",
                    UserWarning
                )
                self._warned_symbols.add(symbol)

        return metadata

    def get_currency(self, symbol: str, default: str = "USD") -> str:
        """
        Get currency for a symbol.

        Args:
            symbol: Security symbol
            default: Default currency if not found or not specified

        Returns:
            Currency code
        """
        metadata = self.get_metadata(symbol, warn_on_missing_currency=True)
        if metadata and metadata.currency:
            return metadata.currency
        return default

    def validate_securities(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Validate a list of securities and return any issues.

        Args:
            symbols: List of security symbols to validate

        Returns:
            Dict with keys 'missing_currency', 'missing_type', 'not_found'
        """
        issues = {
            'missing_currency': [],
            'missing_type': [],
            'not_found': []
        }

        for symbol in symbols:
            metadata = self.securities.get(symbol)
            if metadata is None:
                issues['not_found'].append(symbol)
            else:
                if not metadata.currency:
                    issues['missing_currency'].append(symbol)
                if not metadata.type:
                    issues['missing_type'].append(symbol)

        return issues

    def print_validation_warnings(self, symbols: List[str]) -> None:
        """
        Print validation warnings for a list of securities.

        Args:
            symbols: List of security symbols to validate
        """
        issues = self.validate_securities(symbols)

        if issues['not_found']:
            print(f"WARNING: {len(issues['not_found'])} securities not found in security_metadata.json:")
            for sym in issues['not_found'][:10]:
                print(f"  - {sym}")
            if len(issues['not_found']) > 10:
                print(f"  ... and {len(issues['not_found']) - 10} more")

        if issues['missing_currency']:
            print(f"WARNING: {len(issues['missing_currency'])} securities have no currency defined (assuming USD):")
            for sym in issues['missing_currency'][:10]:
                print(f"  - {sym}")
            if len(issues['missing_currency']) > 10:
                print(f"  ... and {len(issues['missing_currency']) - 10} more")

    def get_symbols_by_type(self, security_type: str) -> List[str]:
        """
        Get all symbols of a specific type.

        Args:
            security_type: Type of security (e.g., 'stock', 'crypto')

        Returns:
            List of symbols
        """
        return [
            meta.symbol for meta in self.securities.values()
            if meta.type.lower() == security_type.lower()
        ]

    def get_symbols_by_sector(self, sector: str) -> List[str]:
        """
        Get all symbols in a specific sector.

        Args:
            sector: Sector name

        Returns:
            List of symbols
        """
        return [
            meta.symbol for meta in self.securities.values()
            if meta.sector.lower() == sector.lower()
        ]

    def get_symbols_by_market_cap(self, market_cap: str) -> List[str]:
        """
        Get all symbols with a specific market cap category.

        Args:
            market_cap: Market cap category (large_cap, mid_cap, small_cap)

        Returns:
            List of symbols
        """
        return [
            meta.symbol for meta in self.securities.values()
            if meta.market_cap and meta.market_cap.lower() == market_cap.lower()
        ]

    def get_all_symbols(self) -> List[str]:
        """
        Get all registered symbols.

        Returns:
            List of all symbols
        """
        return list(self.securities.keys())

    def get_all_types(self) -> List[str]:
        """
        Get all unique security types.

        Returns:
            List of security types
        """
        return list(set(meta.type for meta in self.securities.values() if meta.type))

    def get_all_sectors(self) -> List[str]:
        """
        Get all unique sectors.

        Returns:
            List of sectors
        """
        return list(set(meta.sector for meta in self.securities.values() if meta.sector))

    def get_all_market_caps(self) -> List[str]:
        """
        Get all unique market cap categories.

        Returns:
            List of market cap categories
        """
        return list(set(meta.market_cap for meta in self.securities.values() if meta.market_cap))
