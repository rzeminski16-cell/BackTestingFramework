"""
Security registry for tracking available securities and metadata.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SecurityMetadata:
    """Metadata for a security."""
    symbol: str
    type: str = "unknown"  # stock, crypto, commodity, etc.
    sector: str = "unknown"
    description: str = ""
    currency: str = "GBP"  # Default to GBP (base currency)


class SecurityRegistry:
    """
    Registry of available securities with metadata.
    """

    def __init__(self, metadata_file: Optional[Path] = None):
        """
        Initialize security registry.

        Args:
            metadata_file: Optional path to JSON file with security metadata
        """
        self.securities: Dict[str, SecurityMetadata] = {}

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
                type=info.get('type', 'unknown'),
                sector=info.get('sector', 'unknown'),
                description=info.get('description', ''),
                currency=info.get('currency', 'GBP')
            )

    def register(self, symbol: str, type: str = "unknown",
                sector: str = "unknown", description: str = "",
                currency: str = "GBP") -> None:
        """
        Register a security.

        Args:
            symbol: Security symbol
            type: Security type
            sector: Sector
            description: Description
            currency: Currency code (e.g., USD, EUR, GBP)
        """
        self.securities[symbol] = SecurityMetadata(
            symbol=symbol,
            type=type,
            sector=sector,
            description=description,
            currency=currency
        )

    def get_metadata(self, symbol: str) -> Optional[SecurityMetadata]:
        """
        Get metadata for a symbol.

        Args:
            symbol: Security symbol

        Returns:
            SecurityMetadata or None if not found
        """
        return self.securities.get(symbol)

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
        return list(set(meta.type for meta in self.securities.values()))

    def get_all_sectors(self) -> List[str]:
        """
        Get all unique sectors.

        Returns:
            List of sectors
        """
        return list(set(meta.sector for meta in self.securities.values()))
