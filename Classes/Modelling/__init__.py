"""
Modelling & Evaluation stage.

Consumes the named, point-in-time run package produced by the data-preparation
stage (``processed_data/runs/<run_id>/``) and answers *when does this strategy
work or fail, and when is a reduce-size / filter overlay justified?* — via
per-trade and per-period models under chronological, leakage-aware validation,
economic-first evaluation led by Adjusted RAR%, staged interpretation, robustness
controls, and a full export set including an exportable scoring function.

The GUI (``ctk_modelling_evaluation_gui.py``) is a thin view over
:class:`~Classes.Modelling.controller.ModellingController`.
"""

from .config import (AdjustedRARConfig, LadderConfig, ModellingConfig,
                     ModellingView, OverlayPolicy, TargetKind, TargetSpec,
                     ValidationConfig, ValidationDesign, WeightMode)
from .run_package import PackageReadiness, RunPackage, RunPackageLoader
from .controller import ModellingController, ModelRunResults, ViewResults
from .scoring_function import ScoringFunction

__all__ = [
    "AdjustedRARConfig", "LadderConfig", "ModellingConfig", "ModellingView",
    "OverlayPolicy", "TargetKind", "TargetSpec", "ValidationConfig",
    "ValidationDesign", "WeightMode", "PackageReadiness", "RunPackage",
    "RunPackageLoader", "ModellingController", "ModelRunResults", "ViewResults",
    "ScoringFunction",
]
