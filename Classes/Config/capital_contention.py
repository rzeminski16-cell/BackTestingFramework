"""
Capital contention configuration for portfolio backtesting.

This module defines the capital contention modes and vulnerability score parameters
that control how the portfolio engine handles new signals when capital is limited.

The vulnerability score is the percentage distance of the current reference price
below a compound-growth target price (modulated by realized performance and recent
short-term pullbacks). Trades priced at or above target are immune.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class CapitalContentionMode(Enum):
    """Capital contention resolution mode."""
    DEFAULT = "default"  # Ignore new signals when no capital available
    VULNERABILITY_SCORE = "vulnerability_score"  # Evaluate and potentially swap weak positions


@dataclass
class VulnerabilityScoreConfig:
    """
    Configuration for the target-price vulnerability score.

    The target price at age t (trading days since entry) is:

        P_target(t) = P_entry * (1 + g_d)^t
                      * ((1 + r_long(t)) / (1 + g_d))^(-alpha)
                      * (1 + beta * min(r14, 0))

    where
        g_d       = (1 + target_monthly_growth)^(1/30) - 1
        r_long(t) = (P_t / P_entry)^(1/t) - 1
        r14       = average daily return over the last `pullback_window_days` days

    The vulnerability score is the % distance of the current reference price
    below the target (0 if at/above target - trade is immune to swapping).
    Trades younger than `min_trade_age_days` are also immune.

    Attributes:
        min_trade_age_days: Age in days below which a position is immune (rule not applied).
        target_monthly_growth: Target monthly growth as decimal (0.05 = 5%).
        alpha: Sensitivity to realized long-run performance (higher = more target movement).
        beta: Sensitivity to negative 14-day returns (higher = more leniency on pullbacks).
        avg_window_days: Window length for reference price averaging (P_entry / P_t).
        pullback_window_days: Window length for computing r14 (mean daily return).
    """
    min_trade_age_days: int = 100
    target_monthly_growth: float = 0.05
    alpha: float = 1.0
    beta: float = 1.0
    avg_window_days: int = 7
    pullback_window_days: int = 14

    def __post_init__(self):
        """Validate configuration."""
        if self.min_trade_age_days < 0:
            raise ValueError("min_trade_age_days must be non-negative")
        if self.target_monthly_growth <= -1:
            raise ValueError("target_monthly_growth must be greater than -1")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.avg_window_days < 1:
            raise ValueError("avg_window_days must be at least 1")
        if self.pullback_window_days < 1:
            raise ValueError("pullback_window_days must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'min_trade_age_days': self.min_trade_age_days,
            'target_monthly_growth': self.target_monthly_growth,
            'alpha': self.alpha,
            'beta': self.beta,
            'avg_window_days': self.avg_window_days,
            'pullback_window_days': self.pullback_window_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VulnerabilityScoreConfig':
        """Create from dictionary, ignoring unknown keys (for migration)."""
        defaults = cls()
        return cls(
            min_trade_age_days=int(data.get('min_trade_age_days', defaults.min_trade_age_days)),
            target_monthly_growth=float(data.get('target_monthly_growth', defaults.target_monthly_growth)),
            alpha=float(data.get('alpha', defaults.alpha)),
            beta=float(data.get('beta', defaults.beta)),
            avg_window_days=int(data.get('avg_window_days', defaults.avg_window_days)),
            pullback_window_days=int(data.get('pullback_window_days', defaults.pullback_window_days)),
        )


@dataclass
class CapitalContentionConfig:
    """
    Configuration for capital contention resolution.

    Attributes:
        mode: The capital contention resolution mode.
        vulnerability_config: Parameters for the VULNERABILITY_SCORE mode.
    """
    mode: CapitalContentionMode = CapitalContentionMode.DEFAULT
    vulnerability_config: VulnerabilityScoreConfig = None

    def __post_init__(self):
        if self.vulnerability_config is None:
            self.vulnerability_config = VulnerabilityScoreConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mode': self.mode.value,
            'vulnerability_config': self.vulnerability_config.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapitalContentionConfig':
        """Create from dictionary."""
        mode = CapitalContentionMode(data.get('mode', 'default'))
        vuln_data = data.get('vulnerability_config', {})
        return cls(mode=mode, vulnerability_config=VulnerabilityScoreConfig.from_dict(vuln_data))

    @classmethod
    def default_mode(cls) -> 'CapitalContentionConfig':
        """Create configuration with default mode (ignore signals when no capital)."""
        return cls(mode=CapitalContentionMode.DEFAULT)

    @classmethod
    def vulnerability_score_mode(cls, **kwargs) -> 'CapitalContentionConfig':
        """Create configuration with vulnerability score mode."""
        return cls(
            mode=CapitalContentionMode.VULNERABILITY_SCORE,
            vulnerability_config=VulnerabilityScoreConfig(**kwargs)
        )


# Parameter definitions for GUI and optimization
VULNERABILITY_SCORE_PARAM_DEFINITIONS = {
    'min_trade_age_days': {
        'type': 'int',
        'min': 0,
        'max': 365,
        'default': 100,
        'step': 1,
        'description': 'Age (trading days) below which a position is immune',
        'hint': 'Trades younger than this are never swapped for capital contention.',
    },
    'target_monthly_growth': {
        'type': 'float',
        'min': 0.0,
        'max': 0.50,
        'default': 0.05,
        'step': 0.005,
        'description': 'Target monthly growth rate (decimal, 0.05 = 5%)',
        'hint': 'Converted to a daily compounded rate for the target path.',
    },
    'alpha': {
        'type': 'float',
        'min': 0.0,
        'max': 5.0,
        'default': 1.0,
        'step': 0.1,
        'description': 'Sensitivity to realized long-run performance',
        'hint': 'Higher alpha tightens the bar for laggards and loosens it for strong winners.',
    },
    'beta': {
        'type': 'float',
        'min': 0.0,
        'max': 20.0,
        'default': 1.0,
        'step': 0.5,
        'description': 'Sensitivity to negative 14-day average daily returns',
        'hint': 'Higher beta relaxes the target during short-term drawdowns.',
    },
    'avg_window_days': {
        'type': 'int',
        'min': 1,
        'max': 30,
        'default': 7,
        'step': 1,
        'description': 'Window size for reference price averaging',
        'hint': 'Number of bars averaged for P_entry and P_t.',
    },
    'pullback_window_days': {
        'type': 'int',
        'min': 1,
        'max': 60,
        'default': 14,
        'step': 1,
        'description': 'Window size for computing 14-day avg daily return (r14)',
        'hint': 'Number of daily returns averaged for the pullback factor.',
    },
}
