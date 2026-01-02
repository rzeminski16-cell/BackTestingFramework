"""
Scenario analysis module for Factor Analysis.

Provides:
- Binary scenario detection (above/below threshold)
- Automatic clustering scenarios
- Scenario validation and statistical testing
- Interaction analysis for factor combinations
"""

from .scenario_detector import ScenarioDetector, Scenario
from .scenario_validator import ScenarioValidator, ValidationResult
from .interaction_analyzer import InteractionAnalyzer

__all__ = [
    'ScenarioDetector',
    'Scenario',
    'ScenarioValidator',
    'ValidationResult',
    'InteractionAnalyzer',
]
