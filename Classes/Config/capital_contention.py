"""
Capital contention configuration for portfolio backtesting.

This module defines the capital contention modes and vulnerability score parameters
that control how the portfolio engine handles new signals when capital is limited.

Enhanced with feature-based vulnerability scoring from VulnerabilityScorer module.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List


class CapitalContentionMode(Enum):
    """Capital contention resolution mode."""
    DEFAULT = "default"  # Ignore new signals when no capital available
    VULNERABILITY_SCORE = "vulnerability_score"  # Evaluate and potentially swap weak positions
    ENHANCED_VULNERABILITY = "enhanced_vulnerability"  # Use enhanced feature-based scoring


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


@dataclass
class FeatureWeightConfig:
    """
    Configuration for a single feature in enhanced vulnerability scoring.

    This allows fine-grained control over how each feature contributes
    to the vulnerability score calculation.
    """
    enabled: bool = True
    weight: float = 1.0
    decay_point: int = 14
    fast_decay_rate: float = 5.0
    slow_decay_rate: float = 1.0
    stagnation_threshold: float = 2.0
    normalize_min: Optional[float] = None
    normalize_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'weight': self.weight,
            'decay_point': self.decay_point,
            'fast_decay_rate': self.fast_decay_rate,
            'slow_decay_rate': self.slow_decay_rate,
            'stagnation_threshold': self.stagnation_threshold,
            'normalize_min': self.normalize_min,
            'normalize_max': self.normalize_max
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureWeightConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EnhancedVulnerabilityConfig:
    """
    Enhanced vulnerability score configuration with feature-based scoring.

    This configuration provides full control over the vulnerability scoring
    algorithm, including:
    - Multiple configurable features (days_held, current_pl_pct, momentum, etc.)
    - Per-feature weights and parameters
    - Integration with the VulnerabilityScorer analysis module

    Attributes:
        name: Human-readable name for this configuration
        description: Description of the configuration
        immunity_days: Days a new trade is protected from being swapped
        base_score: Starting score (typically 100)
        swap_threshold: Score below which position can be swapped
        features: Dictionary mapping feature names to their configurations
        tiebreaker_order: Feature priority for breaking ties
    """
    name: str = "Enhanced Default"
    description: str = "Feature-based vulnerability scoring"
    immunity_days: int = 7
    base_score: float = 100.0
    swap_threshold: float = 50.0
    features: Dict[str, FeatureWeightConfig] = field(default_factory=dict)
    tiebreaker_order: List[str] = field(default_factory=lambda: ['current_pl_pct', 'days_held'])

    def __post_init__(self):
        """Initialize default features if not provided."""
        if not self.features:
            self.features = self._get_default_features()

        # Validate
        if self.immunity_days < 0:
            raise ValueError("Immunity days must be non-negative")
        if self.base_score <= 0:
            raise ValueError("Base score must be positive")
        if self.swap_threshold < 0 or self.swap_threshold > self.base_score:
            raise ValueError(f"Swap threshold must be between 0 and {self.base_score}")

    @staticmethod
    def _get_default_features() -> Dict[str, FeatureWeightConfig]:
        """Get default feature configurations."""
        return {
            'days_held': FeatureWeightConfig(
                enabled=True,
                weight=-5.0,  # Negative = loses points per day
                decay_point=14
            ),
            'current_pl_pct': FeatureWeightConfig(
                enabled=True,
                weight=1.0,
                stagnation_threshold=2.0
            ),
            'pl_momentum_7d': FeatureWeightConfig(
                enabled=True,
                weight=3.0
            ),
            'pl_momentum_14d': FeatureWeightConfig(
                enabled=False,
                weight=0.0
            ),
            'volatility_7d': FeatureWeightConfig(
                enabled=True,
                weight=0.5
            ),
            'distance_from_high': FeatureWeightConfig(
                enabled=False,
                weight=0.0
            ),
            'distance_from_entry': FeatureWeightConfig(
                enabled=False,
                weight=0.0
            ),
            'max_favorable_excursion': FeatureWeightConfig(
                enabled=False,
                weight=0.0
            ),
            'entropy_7d': FeatureWeightConfig(
                enabled=False,
                weight=0.0
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'immunity_days': self.immunity_days,
            'base_score': self.base_score,
            'swap_threshold': self.swap_threshold,
            'features': {
                name: config.to_dict()
                for name, config in self.features.items()
            },
            'tiebreaker_order': self.tiebreaker_order
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedVulnerabilityConfig':
        """Create from dictionary."""
        features = {}
        for name, config_data in data.get('features', {}).items():
            features[name] = FeatureWeightConfig.from_dict(config_data)

        return cls(
            name=data.get('name', 'Enhanced Default'),
            description=data.get('description', ''),
            immunity_days=data.get('immunity_days', 7),
            base_score=data.get('base_score', 100.0),
            swap_threshold=data.get('swap_threshold', 50.0),
            features=features if features else None,  # Will use defaults
            tiebreaker_order=data.get('tiebreaker_order', ['current_pl_pct', 'days_held'])
        )

    def to_vulnerability_score_params(self):
        """
        Convert to VulnerabilityScoreParams for use with VulnerabilityScorer module.

        Returns:
            VulnerabilityScoreParams object
        """
        # Import here to avoid circular imports
        from ..VulnerabilityScorer.scoring import VulnerabilityScoreParams
        from ..VulnerabilityScorer.features import FeatureWeight

        feature_weights = {}
        for name, config in self.features.items():
            feature_weights[name] = FeatureWeight(
                enabled=config.enabled,
                weight=config.weight,
                decay_point=config.decay_point,
                fast_decay_rate=config.fast_decay_rate,
                slow_decay_rate=config.slow_decay_rate,
                stagnation_threshold=config.stagnation_threshold,
                normalize_min=config.normalize_min,
                normalize_max=config.normalize_max
            )

        return VulnerabilityScoreParams(
            name=self.name,
            description=self.description,
            immunity_days=self.immunity_days,
            base_score=self.base_score,
            swap_threshold=self.swap_threshold,
            features=feature_weights,
            tiebreaker_order=self.tiebreaker_order
        )

    @classmethod
    def conservative_preset(cls) -> 'EnhancedVulnerabilityConfig':
        """Create conservative preset - protects positions longer."""
        features = {
            'days_held': FeatureWeightConfig(enabled=True, weight=-1.0),
            'current_pl_pct': FeatureWeightConfig(enabled=True, weight=2.0),
            'pl_momentum_7d': FeatureWeightConfig(enabled=True, weight=1.0),
            'pl_momentum_14d': FeatureWeightConfig(enabled=False, weight=0.0),
            'volatility_7d': FeatureWeightConfig(enabled=False, weight=0.0),
            'distance_from_high': FeatureWeightConfig(enabled=False, weight=0.0),
            'distance_from_entry': FeatureWeightConfig(enabled=False, weight=0.0),
            'max_favorable_excursion': FeatureWeightConfig(enabled=False, weight=0.0),
            'entropy_7d': FeatureWeightConfig(enabled=False, weight=0.0),
        }
        return cls(
            name="Conservative",
            description="Protects positions longer, unlikely to swap",
            immunity_days=14,
            swap_threshold=30.0,
            features=features
        )

    @classmethod
    def aggressive_preset(cls) -> 'EnhancedVulnerabilityConfig':
        """Create aggressive preset - swaps quickly if no progress."""
        features = {
            'days_held': FeatureWeightConfig(enabled=True, weight=-3.0),
            'current_pl_pct': FeatureWeightConfig(enabled=True, weight=0.5, stagnation_threshold=3.0),
            'pl_momentum_7d': FeatureWeightConfig(enabled=True, weight=5.0),
            'pl_momentum_14d': FeatureWeightConfig(enabled=False, weight=0.0),
            'volatility_7d': FeatureWeightConfig(enabled=True, weight=-0.5),
            'distance_from_high': FeatureWeightConfig(enabled=False, weight=0.0),
            'distance_from_entry': FeatureWeightConfig(enabled=False, weight=0.0),
            'max_favorable_excursion': FeatureWeightConfig(enabled=True, weight=-1.0),
            'entropy_7d': FeatureWeightConfig(enabled=False, weight=0.0),
        }
        return cls(
            name="Aggressive",
            description="Swaps quickly if no progress",
            immunity_days=3,
            swap_threshold=70.0,
            features=features
        )

    @classmethod
    def momentum_focused_preset(cls) -> 'EnhancedVulnerabilityConfig':
        """Create momentum-focused preset - emphasizes recent momentum."""
        features = {
            'days_held': FeatureWeightConfig(enabled=True, weight=-1.0),
            'current_pl_pct': FeatureWeightConfig(enabled=True, weight=1.0),
            'pl_momentum_7d': FeatureWeightConfig(enabled=True, weight=5.0),
            'pl_momentum_14d': FeatureWeightConfig(enabled=True, weight=2.0),
            'volatility_7d': FeatureWeightConfig(enabled=False, weight=0.0),
            'distance_from_high': FeatureWeightConfig(enabled=False, weight=0.0),
            'distance_from_entry': FeatureWeightConfig(enabled=False, weight=0.0),
            'max_favorable_excursion': FeatureWeightConfig(enabled=False, weight=0.0),
            'entropy_7d': FeatureWeightConfig(enabled=False, weight=0.0),
        }
        return cls(
            name="Momentum Focused",
            description="Emphasizes recent momentum over time held",
            immunity_days=7,
            swap_threshold=50.0,
            features=features
        )


# Feature definitions for GUI with comprehensive parameter ranges
ENHANCED_FEATURE_DEFINITIONS = {
    'days_held': {
        'name': 'Days Held',
        'description': 'Days since entry - penalizes older positions',
        'type': 'time',
        'importance': 'HIGH',
        'value_range': '0 to N days',
        'recommended_weight': -5.0,
        'weight_range': (-10.0, 0.0),
        'weight_hint': 'Negative values penalize older trades. Use -1 to -10 for moderate to aggressive aging penalty.',
    },
    'current_pl_pct': {
        'name': 'Current P/L %',
        'description': 'Current profit/loss as percentage of entry price',
        'type': 'profitability',
        'importance': 'HIGH',
        'value_range': '-100% to +infinity',
        'recommended_weight': 1.0,
        'weight_range': (0.0, 5.0),
        'weight_hint': 'Positive values reward profitable trades. Higher values (3-5) strongly protect winners.',
    },
    'pl_momentum_7d': {
        'name': 'P/L Momentum (7d)',
        'description': 'P/L change in last 7 days - measures recent performance trend',
        'type': 'momentum',
        'importance': 'HIGH',
        'value_range': '-infinity to +infinity',
        'recommended_weight': 3.0,
        'weight_range': (0.0, 10.0),
        'weight_hint': 'Higher values (3-5) favor trades with positive recent momentum. Key for trend-following.',
    },
    'pl_momentum_14d': {
        'name': 'P/L Momentum (14d)',
        'description': 'P/L change in last 14 days - measures medium-term trend',
        'type': 'momentum',
        'importance': 'MEDIUM',
        'value_range': '-infinity to +infinity',
        'recommended_weight': 2.0,
        'weight_range': (0.0, 10.0),
        'weight_hint': 'Similar to 7d but smoother. Use 1-3 for balanced momentum consideration.',
    },
    'volatility_7d': {
        'name': 'Volatility (7d)',
        'description': '7-day rolling volatility - measures price fluctuation risk',
        'type': 'risk',
        'importance': 'MEDIUM',
        'value_range': '0% to 100%+ (annualized)',
        'recommended_weight': 0.5,
        'weight_range': (-5.0, 5.0),
        'weight_hint': 'Negative penalizes volatile trades (risk-averse). Positive keeps volatile trades longer.',
    },
    'distance_from_high': {
        'name': 'Distance from 52W High',
        'description': 'Percentage below 52-week high - identifies weakness',
        'type': 'technical',
        'importance': 'MEDIUM',
        'value_range': '-100% to 0%',
        'recommended_weight': 0.0,
        'weight_range': (-5.0, 5.0),
        'weight_hint': 'Negative values (e.g., -2) penalize trades far from highs. Usually left at 0.',
    },
    'distance_from_entry': {
        'name': 'Distance from Entry',
        'description': 'Percentage from entry price (same as current P/L %)',
        'type': 'profitability',
        'importance': 'MEDIUM',
        'value_range': '-100% to +infinity',
        'recommended_weight': 0.0,
        'weight_range': (-5.0, 5.0),
        'weight_hint': 'Overlaps with current_pl_pct. Usually disabled. Use for additional entry reference.',
    },
    'max_favorable_excursion': {
        'name': 'Max Favorable Excursion',
        'description': 'Drawdown from best price since entry - identifies trades that have given back gains',
        'type': 'risk',
        'importance': 'LOW',
        'value_range': '0% to 100%',
        'recommended_weight': -1.0,
        'weight_range': (-5.0, 0.0),
        'weight_hint': 'Negative values (e.g., -1 to -3) penalize trades that peaked and fell back.',
    },
    'entropy_7d': {
        'name': 'Entropy (7d)',
        'description': 'Price action noise - coefficient of variation over 7 days',
        'type': 'technical',
        'importance': 'LOW',
        'value_range': '0% to 100%+',
        'recommended_weight': 0.0,
        'weight_range': (-5.0, 5.0),
        'weight_hint': 'Negative penalizes noisy/choppy price action. Usually left at 0.',
    }
}


# Per-feature parameter definitions with ranges and hints for all FeatureWeight attributes
FEATURE_PARAMETER_DEFINITIONS = {
    'enabled': {
        'type': 'bool',
        'default': True,
        'description': 'Whether this feature is included in scoring'
    },
    'weight': {
        'type': 'float',
        'default': 1.0,
        'min': -10.0,
        'max': 10.0,
        'step': 0.5,
        'description': 'Additive modifier to score (can be negative)',
        'hint': 'Negative = penalizes (e.g., days_held), Positive = rewards (e.g., pl_momentum)'
    },
    'decay_point': {
        'type': 'int',
        'default': 14,
        'min': 1,
        'max': 60,
        'step': 1,
        'description': 'Days after which decay rate may change (for time-based features)',
        'hint': 'For days_held: after this many days, penalty increases. 7-21 is typical range.'
    },
    'fast_decay_rate': {
        'type': 'float',
        'default': 5.0,
        'min': 0.5,
        'max': 20.0,
        'step': 0.5,
        'description': 'Points lost per day for stagnant/underperforming trades',
        'hint': 'Higher values (5-10) aggressively cull underperformers. 1-3 is conservative.'
    },
    'slow_decay_rate': {
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 5.0,
        'step': 0.1,
        'description': 'Points lost per day for performing trades',
        'hint': 'Lower values (0.5-1.5) let winners run longer. 2-3 still ages winners moderately.'
    },
    'stagnation_threshold': {
        'type': 'float',
        'default': 2.0,
        'min': 0.0,
        'max': 10.0,
        'step': 0.5,
        'description': 'P/L % below which trade is considered stagnant (uses fast decay)',
        'hint': 'Trades below this P/L% decay faster. 0-2% is conservative, 3-5% is aggressive.'
    },
    'normalize_min': {
        'type': 'float',
        'default': None,
        'min': -100.0,
        'max': 100.0,
        'step': 1.0,
        'description': 'Minimum value for normalization (optional)',
        'hint': 'Optional: Set min/max to normalize feature values to 0-1 range.'
    },
    'normalize_max': {
        'type': 'float',
        'default': None,
        'min': -100.0,
        'max': 100.0,
        'step': 1.0,
        'description': 'Maximum value for normalization (optional)',
        'hint': 'Optional: Set min/max to normalize feature values to 0-1 range.'
    }
}


# Combined core parameters for backtesting/optimization with ranges and hints
VULNERABILITY_CORE_PARAM_DEFINITIONS = {
    'immunity_days': {
        'type': 'int',
        'min': 1,
        'max': 30,
        'default': 7,
        'step': 1,
        'description': 'Days a new trade is protected from being swapped',
        'hint': 'Protects new positions from immediate swaps. 3-7 for aggressive, 10-14 for conservative.'
    },
    'base_score': {
        'type': 'float',
        'min': 50.0,
        'max': 200.0,
        'default': 100.0,
        'step': 10.0,
        'description': 'Starting score for vulnerability calculation',
        'hint': 'Usually kept at 100. Higher values delay vulnerability threshold crossings.'
    },
    'swap_threshold': {
        'type': 'float',
        'min': 10.0,
        'max': 90.0,
        'default': 50.0,
        'step': 5.0,
        'description': 'Score below which position can be swapped for new signal',
        'hint': 'Lower (20-40) = harder to swap, Higher (60-80) = easier to swap.'
    }
}
