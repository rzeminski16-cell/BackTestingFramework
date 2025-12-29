"""
Factor engineering module for Factor Analysis.

Provides factor computation for:
- Technical indicators
- Fundamental ratios (value, quality, growth)
- Insider activity metrics
- Options-derived metrics
- Market regime classification
"""

from .technical_factors import TechnicalFactors
from .fundamental_factors import FundamentalFactors
from .insider_factors import InsiderFactors
from .options_factors import OptionsFactors
from .regime_factors import RegimeFactors
from .normalizers import FactorNormalizer
from .outlier_handler import OutlierHandler

__all__ = [
    'TechnicalFactors',
    'FundamentalFactors',
    'InsiderFactors',
    'OptionsFactors',
    'RegimeFactors',
    'FactorNormalizer',
    'OutlierHandler',
]
