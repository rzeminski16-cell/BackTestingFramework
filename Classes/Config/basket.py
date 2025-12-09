"""
Basket/Portfolio configuration for grouping securities.

A basket is a saved collection of securities that can be backtested
together as a portfolio.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .capital_contention import CapitalContentionConfig, CapitalContentionMode


@dataclass
class Basket:
    """
    A basket/portfolio of securities for combined backtesting.

    Attributes:
        name: Unique name for the basket
        securities: List of security symbols in the basket
        description: Optional description of the basket
        created_date: When the basket was created
        modified_date: When the basket was last modified
        default_capital_contention: Default capital contention mode for this basket
        metadata: Additional metadata (e.g., sector focus, risk level)
    """
    name: str
    securities: List[str]
    description: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)
    default_capital_contention: Optional[CapitalContentionConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate basket configuration."""
        if not self.name or not self.name.strip():
            raise ValueError("Basket name cannot be empty")
        if not self.securities:
            raise ValueError("Basket must contain at least one security")
        # Remove duplicates while preserving order
        seen = set()
        unique_securities = []
        for sec in self.securities:
            if sec not in seen:
                seen.add(sec)
                unique_securities.append(sec)
        self.securities = unique_securities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'name': self.name,
            'securities': self.securities,
            'description': self.description,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'metadata': self.metadata
        }
        if self.default_capital_contention:
            result['default_capital_contention'] = self.default_capital_contention.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Basket':
        """Create from dictionary."""
        # Parse dates
        created_date = datetime.now()
        if 'created_date' in data:
            try:
                created_date = datetime.fromisoformat(data['created_date'])
            except (ValueError, TypeError):
                pass

        modified_date = datetime.now()
        if 'modified_date' in data:
            try:
                modified_date = datetime.fromisoformat(data['modified_date'])
            except (ValueError, TypeError):
                pass

        # Parse capital contention config
        capital_contention = None
        if 'default_capital_contention' in data:
            capital_contention = CapitalContentionConfig.from_dict(
                data['default_capital_contention']
            )

        return cls(
            name=data['name'],
            securities=data['securities'],
            description=data.get('description', ''),
            created_date=created_date,
            modified_date=modified_date,
            default_capital_contention=capital_contention,
            metadata=data.get('metadata', {})
        )

    def add_security(self, symbol: str) -> None:
        """Add a security to the basket."""
        if symbol not in self.securities:
            self.securities.append(symbol)
            self.modified_date = datetime.now()

    def remove_security(self, symbol: str) -> None:
        """Remove a security from the basket."""
        if symbol in self.securities:
            self.securities.remove(symbol)
            self.modified_date = datetime.now()

    def __len__(self) -> int:
        """Return number of securities in basket."""
        return len(self.securities)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in basket."""
        return symbol in self.securities

    def __str__(self) -> str:
        """String representation."""
        return f"Basket({self.name}: {len(self.securities)} securities)"


class BasketManager:
    """
    Manager for creating, saving, loading, and deleting baskets.

    Baskets are stored as JSON files in a configurable directory.
    """

    DEFAULT_DIRECTORY = "config/baskets"

    def __init__(self, directory: Optional[str] = None):
        """
        Initialize the basket manager.

        Args:
            directory: Directory to store basket files (default: config/baskets)
        """
        self.directory = Path(directory or self.DEFAULT_DIRECTORY)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Basket] = {}

    def _get_basket_path(self, name: str) -> Path:
        """Get the file path for a basket."""
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
        return self.directory / f"{safe_name}.json"

    def save(self, basket: Basket) -> Path:
        """
        Save a basket to file.

        Args:
            basket: Basket to save

        Returns:
            Path to saved file
        """
        basket.modified_date = datetime.now()
        filepath = self._get_basket_path(basket.name)

        with open(filepath, 'w') as f:
            json.dump(basket.to_dict(), f, indent=2)

        self._cache[basket.name] = basket
        return filepath

    def load(self, name: str) -> Optional[Basket]:
        """
        Load a basket by name.

        Args:
            name: Basket name

        Returns:
            Basket if found, None otherwise
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        filepath = self._get_basket_path(name)
        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            basket = Basket.from_dict(data)
            self._cache[name] = basket
            return basket
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading basket '{name}': {e}")
            return None

    def delete(self, name: str) -> bool:
        """
        Delete a basket.

        Args:
            name: Basket name

        Returns:
            True if deleted, False if not found
        """
        filepath = self._get_basket_path(name)
        if filepath.exists():
            filepath.unlink()
            if name in self._cache:
                del self._cache[name]
            return True
        return False

    def list_baskets(self) -> List[str]:
        """
        List all available basket names.

        Returns:
            List of basket names
        """
        baskets = []
        for filepath in self.directory.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                baskets.append(data.get('name', filepath.stem))
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(baskets)

    def get_all_baskets(self) -> List[Basket]:
        """
        Load and return all baskets.

        Returns:
            List of all baskets
        """
        baskets = []
        for name in self.list_baskets():
            basket = self.load(name)
            if basket:
                baskets.append(basket)
        return baskets

    def exists(self, name: str) -> bool:
        """
        Check if a basket exists.

        Args:
            name: Basket name

        Returns:
            True if basket exists
        """
        return self._get_basket_path(name).exists()

    def create(self, name: str, securities: List[str],
               description: str = "",
               capital_contention: Optional[CapitalContentionConfig] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Basket:
        """
        Create and save a new basket.

        Args:
            name: Basket name
            securities: List of security symbols
            description: Optional description
            capital_contention: Optional default capital contention config
            metadata: Optional metadata

        Returns:
            Created basket
        """
        basket = Basket(
            name=name,
            securities=securities,
            description=description,
            default_capital_contention=capital_contention,
            metadata=metadata or {}
        )
        self.save(basket)
        return basket

    def update(self, name: str, securities: Optional[List[str]] = None,
               description: Optional[str] = None,
               capital_contention: Optional[CapitalContentionConfig] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[Basket]:
        """
        Update an existing basket.

        Args:
            name: Basket name
            securities: New securities list (if None, keep existing)
            description: New description (if None, keep existing)
            capital_contention: New capital contention config (if None, keep existing)
            metadata: New metadata (if None, keep existing)

        Returns:
            Updated basket, or None if not found
        """
        basket = self.load(name)
        if not basket:
            return None

        if securities is not None:
            basket.securities = securities
        if description is not None:
            basket.description = description
        if capital_contention is not None:
            basket.default_capital_contention = capital_contention
        if metadata is not None:
            basket.metadata = metadata

        self.save(basket)
        return basket

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
