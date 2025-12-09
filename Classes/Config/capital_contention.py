"""
Capital contention configuration for portfolio backtesting.

This module defines the capital contention modes and vulnerability score parameters
that control how the portfolio engine handles new signals when capital is limited.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class CapitalContentionMode(Enum):
    """Capital contention resolution mode."""
    DEFAULT = "default"  # Ignore new signals when no capital available
    VULNERABILITY_SCORE = "vulnerability_score"  # Evaluate and potentially swap weak positions


@dataclass
class VulnerabilityScoreConfig:
    """
    Configuration for the Vulnerability Score capital contention method.

    This method evaluates existing positions when a new signal arrives but
    no capital is available. If an existing position is deemed "vulnerable"
    (low score), it can be closed to make room for the new position.

    Attributes:
        immunity_days: Number of days a new trade is protected from being swapped out.
                       During this period, the trade has a score of 100 (sacred).
        min_profit_threshold: Minimum profit percentage (as decimal, e.g., 0.02 = 2%)
                              above which a trade is considered "performing" vs "stagnant".
        decay_rate_fast: Points lost per day for stagnant trades (P/L < min_profit_threshold).
                         Higher value = more aggressive culling of underperformers.
        decay_rate_slow: Points lost per day for performing trades (P/L >= min_profit_threshold).
                         Lower value = let winners run longer before becoming vulnerable.
        swap_threshold: Score below which a position becomes vulnerable to being swapped.
                        Only positions with score < swap_threshold can be closed for new signals.
        base_score: Starting score for vulnerability calculation (typically 100).
    """
    immunity_days: int = 7
    min_profit_threshold: float = 0.02  # 2%
    decay_rate_fast: float = 5.0  # Points per day for stagnant trades
    decay_rate_slow: float = 1.0  # Points per day for performing trades
    swap_threshold: float = 50.0  # Score below which position can be swapped
    base_score: float = 100.0

    def __post_init__(self):
        """Validate vulnerability score configuration."""
        if self.immunity_days < 0:
            raise ValueError("Immunity days must be non-negative")
        if self.min_profit_threshold < 0:
            raise ValueError("Minimum profit threshold must be non-negative")
        if self.decay_rate_fast < 0:
            raise ValueError("Fast decay rate must be non-negative")
        if self.decay_rate_slow < 0:
            raise ValueError("Slow decay rate must be non-negative")
        if self.swap_threshold < 0 or self.swap_threshold > self.base_score:
            raise ValueError(f"Swap threshold must be between 0 and {self.base_score}")
        if self.base_score <= 0:
            raise ValueError("Base score must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'immunity_days': self.immunity_days,
            'min_profit_threshold': self.min_profit_threshold,
            'decay_rate_fast': self.decay_rate_fast,
            'decay_rate_slow': self.decay_rate_slow,
            'swap_threshold': self.swap_threshold,
            'base_score': self.base_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VulnerabilityScoreConfig':
        """Create from dictionary."""
        return cls(
            immunity_days=data.get('immunity_days', 7),
            min_profit_threshold=data.get('min_profit_threshold', 0.02),
            decay_rate_fast=data.get('decay_rate_fast', 5.0),
            decay_rate_slow=data.get('decay_rate_slow', 1.0),
            swap_threshold=data.get('swap_threshold', 50.0),
            base_score=data.get('base_score', 100.0)
        )


@dataclass
class CapitalContentionConfig:
    """
    Configuration for capital contention resolution in portfolio backtesting.

    When a new BUY signal arrives but insufficient capital is available,
    this configuration determines how to handle the situation.

    Attributes:
        mode: The capital contention resolution mode
        vulnerability_config: Configuration for vulnerability score method (if mode is VULNERABILITY_SCORE)
    """
    mode: CapitalContentionMode = CapitalContentionMode.DEFAULT
    vulnerability_config: VulnerabilityScoreConfig = field(default_factory=VulnerabilityScoreConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mode': self.mode.value,
            'vulnerability_config': self.vulnerability_config.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapitalContentionConfig':
        """Create from dictionary."""
        mode_str = data.get('mode', 'default')
        mode = CapitalContentionMode(mode_str)

        vuln_data = data.get('vulnerability_config', {})
        vulnerability_config = VulnerabilityScoreConfig.from_dict(vuln_data)

        return cls(mode=mode, vulnerability_config=vulnerability_config)

    @classmethod
    def default_mode(cls) -> 'CapitalContentionConfig':
        """Create configuration with default mode (ignore signals when no capital)."""
        return cls(mode=CapitalContentionMode.DEFAULT)

    @classmethod
    def vulnerability_score_mode(cls, **kwargs) -> 'CapitalContentionConfig':
        """
        Create configuration with vulnerability score mode.

        Args:
            **kwargs: Parameters to pass to VulnerabilityScoreConfig
        """
        return cls(
            mode=CapitalContentionMode.VULNERABILITY_SCORE,
            vulnerability_config=VulnerabilityScoreConfig(**kwargs)
        )


# Parameter definitions for GUI and optimization
VULNERABILITY_SCORE_PARAM_DEFINITIONS = {
    'immunity_days': {
        'type': 'int',
        'min': 1,
        'max': 30,
        'default': 7,
        'description': 'Days a new trade is protected from being swapped'
    },
    'min_profit_threshold': {
        'type': 'float',
        'min': 0.0,
        'max': 0.20,
        'default': 0.02,
        'step': 0.005,
        'description': 'Profit % below which trade is considered stagnant'
    },
    'decay_rate_fast': {
        'type': 'float',
        'min': 1.0,
        'max': 20.0,
        'default': 5.0,
        'step': 0.5,
        'description': 'Points lost per day for stagnant trades'
    },
    'decay_rate_slow': {
        'type': 'float',
        'min': 0.1,
        'max': 5.0,
        'default': 1.0,
        'step': 0.1,
        'description': 'Points lost per day for performing trades'
    },
    'swap_threshold': {
        'type': 'float',
        'min': 10.0,
        'max': 90.0,
        'default': 50.0,
        'step': 5.0,
        'description': 'Score below which position can be swapped'
    }
}
