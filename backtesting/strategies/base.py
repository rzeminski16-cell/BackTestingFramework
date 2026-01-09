"""Base strategy class with parameter framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from backtesting.core.portfolio import Portfolio


class ParameterType(Enum):
    """Parameter type enumeration."""
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


@dataclass
class Parameter:
    """
    Defines a strategy parameter with metadata for optimization.

    Args:
        name: Parameter name (must match strategy attribute)
        param_type: Type of the parameter
        default: Default/control value
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        step: Step size for range generation (for numeric types)
        choices: List of valid choices (for categorical type)
        description: Human-readable description
    """
    name: str
    param_type: ParameterType
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""

    def __post_init__(self):
        """Validate parameter configuration."""
        if self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Categorical parameter '{self.name}' requires choices")
            if self.default not in self.choices:
                raise ValueError(f"Default value must be in choices for '{self.name}'")
        elif self.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.min_value is not None and self.max_value is not None:
                if self.min_value > self.max_value:
                    raise ValueError(f"min_value > max_value for '{self.name}'")
                if self.default < self.min_value or self.default > self.max_value:
                    raise ValueError(f"Default value out of range for '{self.name}'")

    def generate_values(
        self,
        num_values: Optional[int] = None,
        custom_range: Optional[List[Any]] = None,
    ) -> List[Any]:
        """
        Generate a list of values to test for this parameter.

        Args:
            num_values: Number of values to generate (for numeric types)
            custom_range: Custom list of values to use instead

        Returns:
            List of parameter values to test
        """
        if custom_range is not None:
            return custom_range

        if self.param_type == ParameterType.CATEGORICAL:
            return list(self.choices)

        if self.param_type == ParameterType.BOOLEAN:
            return [True, False]

        if self.min_value is None or self.max_value is None:
            raise ValueError(f"Cannot generate range without min/max for '{self.name}'")

        if self.step is not None:
            # Use step to generate values
            values = []
            current = self.min_value
            while current <= self.max_value:
                if self.param_type == ParameterType.INTEGER:
                    values.append(int(round(current)))
                else:
                    values.append(current)
                current += self.step
            return values

        if num_values is not None:
            # Generate evenly spaced values
            if self.param_type == ParameterType.INTEGER:
                values = []
                step = (self.max_value - self.min_value) / (num_values - 1)
                for i in range(num_values):
                    values.append(int(round(self.min_value + i * step)))
                return sorted(set(values))  # Remove duplicates for integers
            else:
                import numpy as np
                return list(np.linspace(self.min_value, self.max_value, num_values))

        # Default: generate reasonable number of values
        if self.param_type == ParameterType.INTEGER:
            range_size = int(self.max_value - self.min_value) + 1
            if range_size <= 20:
                return list(range(int(self.min_value), int(self.max_value) + 1))
            else:
                return self.generate_values(num_values=10)
        else:
            return self.generate_values(num_values=10)


class Strategy(ABC):
    """
    Base class for trading strategies.

    Subclasses must:
    1. Define parameters using the `parameters` class attribute
    2. Implement `initialize()` for setup
    3. Implement `generate_signals()` for signal generation

    Example:
        class SMACrossover(Strategy):
            name = "SMA Crossover"

            parameters = [
                Parameter("fast_period", ParameterType.INTEGER, default=10,
                         min_value=5, max_value=50, step=5),
                Parameter("slow_period", ParameterType.INTEGER, default=30,
                         min_value=20, max_value=100, step=10),
            ]

            def __init__(self, fast_period=10, slow_period=30):
                super().__init__()
                self.fast_period = fast_period
                self.slow_period = slow_period
    """

    name: str = "Base Strategy"
    parameters: List[Parameter] = []

    def __init__(self, **kwargs):
        """
        Initialize strategy with parameters.

        Parameters can be passed as keyword arguments or will use defaults.
        """
        self._param_values = {}

        # Set parameters from definitions
        for param in self.parameters:
            value = kwargs.get(param.name, param.default)
            setattr(self, param.name, value)
            self._param_values[param.name] = value

        # Store any extra kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
                self._param_values[key] = value

    @classmethod
    def get_parameter_definitions(cls) -> Dict[str, Parameter]:
        """Get dictionary of parameter definitions."""
        return {p.name: p for p in cls.parameters}

    def get_parameter_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return {p.name: getattr(self, p.name) for p in self.parameters}

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value."""
        param_def = self.get_parameter_definitions().get(name)
        if param_def is None:
            raise ValueError(f"Unknown parameter: {name}")

        # Type validation
        if param_def.param_type == ParameterType.INTEGER:
            value = int(value)
        elif param_def.param_type == ParameterType.FLOAT:
            value = float(value)
        elif param_def.param_type == ParameterType.BOOLEAN:
            value = bool(value)
        elif param_def.param_type == ParameterType.CATEGORICAL:
            if value not in param_def.choices:
                raise ValueError(f"Invalid choice '{value}' for parameter '{name}'")

        setattr(self, name, value)
        self._param_values[name] = value

    def set_parameters(self, **kwargs) -> None:
        """Set multiple parameters at once."""
        for name, value in kwargs.items():
            self.set_parameter(name, value)

    @classmethod
    def create_with_parameters(cls, **kwargs) -> "Strategy":
        """Create a new instance with specified parameters."""
        return cls(**kwargs)

    @classmethod
    def get_control_values(cls) -> Dict[str, Any]:
        """Get default/control values for all parameters."""
        return {p.name: p.default for p in cls.parameters}

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize the strategy with data.

        Called once before the backtest starts. Override to precompute
        indicators or perform setup.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        current_date: datetime,
    ) -> List[Dict]:
        """
        Generate trading signals for the current bar.

        Args:
            data: Dictionary of symbol -> DataFrame with data up to current_date
            portfolio: Current portfolio state
            current_date: Current date/time being processed

        Returns:
            List of signal dictionaries with keys:
                - action: "buy", "sell", "short", "cover"
                - symbol: Symbol to trade (default: "default")
                - size: Position size (optional, uses size_pct if not provided)
                - size_pct: Fraction of portfolio (default: 0.95)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
        """
        pass

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_parameter_values().items())
        return f"{self.name}({params})"
