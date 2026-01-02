"""
Preprocessing module for Factor Analysis.

Provides:
- Trade classification (good/bad/indeterminate)
- Temporal alignment for bias prevention
- Data enrichment and merging
- Data quality scoring
- Multi-log aggregation
"""

from .trade_classifier import TradeClassifier, TradeClass
from .temporal_aligner import TemporalAligner
from .data_enricher import DataEnricher
from .quality_scorer import QualityScorer, QualityScore
from .multi_log_aggregator import MultiLogAggregator

__all__ = [
    'TradeClassifier',
    'TradeClass',
    'TemporalAligner',
    'DataEnricher',
    'QualityScorer',
    'QualityScore',
    'MultiLogAggregator',
]
