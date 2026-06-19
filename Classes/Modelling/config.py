"""
Configuration models for the Modelling & Evaluation stage.

The *model run* is the primary object here, mirroring how the data-prep stage
treats the data run. A :class:`ModellingConfig` captures every choice the six GUI
screens collect — which prepared run package to analyse, what "works" means
(targets), how to validate honestly, which models to compare, and what to export —
so a run is fully reproducible and auditable from this single record.

This module is GUI-free and import-light (stdlib + dataclasses only) so it can be
used from a CLI or tests without a display or the ML stack installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def slugify(name: str) -> str:
    """Filesystem-safe, stable identifier from a free-form name."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return slug or "model_run"


class ModellingView(str, Enum):
    """The native analytical view(s) a run produces."""
    PER_TRADE = "per_trade"     # one row per trade at entry time
    PER_PERIOD = "per_period"   # one row per fixed period (regime view)
    DUAL = "dual"               # both, reported together


class TargetKind(str, Enum):
    """The supported target families."""
    # Per-trade
    BINARY_GOOD_TRADE = "binary_good_trade"          # cost-aware good/bad label
    CONTINUOUS_NET_RETURN = "continuous_net_return"  # clipped utility-oriented return
    BINARY_TAIL_LOSS = "binary_tail_loss"            # tail-risk flag
    # Per-period
    NEXT_PERIOD_RETURN = "next_period_return"
    NEXT_PERIOD_ADJ_RAR = "next_period_adjusted_rar"
    REGIME_LABEL = "regime_label"                    # favourable / neutral / hostile

    @property
    def is_classification(self) -> bool:
        return self in {
            TargetKind.BINARY_GOOD_TRADE,
            TargetKind.BINARY_TAIL_LOSS,
            TargetKind.REGIME_LABEL,
        }


class ValidationDesign(str, Enum):
    """Chronological validation strategies (never shuffled)."""
    EXPANDING_WALK_FORWARD = "expanding_walk_forward"   # default for period panels
    ROLLING_WALK_FORWARD = "rolling_walk_forward"       # regime-drift sensitivity
    PURGED_EMBARGOED = "purged_embargoed"               # default for overlapping trades


class WeightMode(str, Enum):
    """Sample-weighting regime (always reported with an unweighted parallel view)."""
    EQUAL = "equal"          # pure diagnosis
    CLASS_BALANCED = "class" # inverse-frequency, imbalance control
    ECONOMIC = "economic"    # capital-at-risk / |pl| emphasis


class OverlayPolicy(str, Enum):
    """Economic overlay policies the model predictions are translated into."""
    ALWAYS = "always_trade"
    TOP_QUANTILE = "top_quantile_only"
    REDUCE_IN_HOSTILE = "reduce_size_in_hostile"
    BLOCK_TAIL = "block_tail_risk"


# --------------------------------------------------------------------------- #
# Adjusted RAR% — the primary, house-configurable selection metric.
# --------------------------------------------------------------------------- #
@dataclass
class AdjustedRARConfig:
    """Configurable Adjusted RAR% definition.

    Defaults reproduce the framework's existing house metric
    (``Classes/Core/stable_metrics.py``): RAR% from a log-equity regression,
    multiplied by R² to penalise noisy curves. Every knob is surfaced in the GUI
    so the exact formula remains the user's choice, not a hidden default.
    """
    bars_per_year: int = 365          # matches stable_metrics BARS_PER_YEAR (calendar)
    weight_by_r_squared: bool = True  # RAR% × R²  (False -> raw RAR%)
    clip_min: Optional[float] = None  # optional floor on the reported value
    clip_max: Optional[float] = None  # optional cap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bars_per_year": self.bars_per_year,
            "weight_by_r_squared": self.weight_by_r_squared,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdjustedRARConfig":
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TargetSpec:
    """What "works" means for this run.

    Thresholds are economic, not arbitrary: a trade is "good" only if its net
    return beats estimated costs plus a safety buffer. ``period_freq`` and
    aggregation only matter for per-period targets.
    """
    kinds: List[TargetKind] = field(default_factory=lambda: [TargetKind.BINARY_GOOD_TRADE])
    primary: TargetKind = TargetKind.BINARY_GOOD_TRADE

    # Cost-aware success threshold (percent of notional) for binary good-trade.
    cost_buffer_pct: float = 0.20      # estimated round-trip cost + buffer, in %
    # Tail-loss threshold (percent). A trade is a "tail loss" if pl_pct <= this.
    tail_loss_pct: float = -5.0
    # Clip bounds for the continuous net-return target (economic granularity).
    return_clip_pct: float = 25.0

    # Per-period settings.
    period_freq: str = "W"             # pandas offset alias: D / W / M
    regime_favourable_pct: float = 0.0  # next-period return >= this -> favourable side
    regime_hostile_pct: float = 0.0     # next-period return <  this -> hostile side

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kinds": [k.value for k in self.kinds],
            "primary": self.primary.value,
            "cost_buffer_pct": self.cost_buffer_pct,
            "tail_loss_pct": self.tail_loss_pct,
            "return_clip_pct": self.return_clip_pct,
            "period_freq": self.period_freq,
            "regime_favourable_pct": self.regime_favourable_pct,
            "regime_hostile_pct": self.regime_hostile_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetSpec":
        data = dict(data or {})
        if "kinds" in data:
            data["kinds"] = [TargetKind(k) for k in data["kinds"]]
        if "primary" in data:
            data["primary"] = TargetKind(data["primary"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ValidationConfig:
    """How to test honestly. All designs are chronological."""
    design: ValidationDesign = ValidationDesign.PURGED_EMBARGOED
    n_splits: int = 5
    embargo_days: int = 5          # trade-time embargo after each test block
    min_train_size: int = 30       # minimum training rows before a fold is scored
    nested: bool = True            # nested inner search for tuned models
    inner_splits: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design": self.design.value,
            "n_splits": self.n_splits,
            "embargo_days": self.embargo_days,
            "min_train_size": self.min_train_size,
            "nested": self.nested,
            "inner_splits": self.inner_splits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        data = dict(data or {})
        if "design" in data:
            data["design"] = ValidationDesign(data["design"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LadderConfig:
    """Which models to compare, in conservative order."""
    run_baseline: bool = True
    run_linear: bool = True            # regularised logistic / elastic-net
    run_tree: bool = True              # shallow decision tree
    run_ensemble: bool = False         # constrained HistGradientBoosting (opt-in)
    tune: bool = True                  # small, explicit, nested search
    calibrate: bool = True             # CalibratedClassifierCV for probabilities

    def enabled_tiers(self) -> List[str]:
        tiers = []
        if self.run_baseline:
            tiers.append("baseline")
        if self.run_linear:
            tiers.append("linear")
        if self.run_tree:
            tiers.append("tree")
        if self.run_ensemble:
            tiers.append("ensemble")
        return tiers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_baseline": self.run_baseline,
            "run_linear": self.run_linear,
            "run_tree": self.run_tree,
            "run_ensemble": self.run_ensemble,
            "tune": self.tune,
            "calibrate": self.calibrate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LadderConfig":
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModellingConfig:
    """Top-level configuration for one modelling run over a prepared package."""
    model_run_name: str
    source_run_id: str = ""                       # the data-prep run package id
    runs_root: str = "processed_data/runs"        # where packages live

    view: ModellingView = ModellingView.PER_TRADE

    # Feature scope: families to draw features from (empty -> all present in the
    # package). Optional explicit column allow-list per family.
    feature_families: List[str] = field(default_factory=list)
    feature_columns: Dict[str, List[str]] = field(default_factory=dict)

    target: TargetSpec = field(default_factory=TargetSpec)
    adjusted_rar: AdjustedRARConfig = field(default_factory=AdjustedRARConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)
    weight_mode: WeightMode = WeightMode.EQUAL

    # Economics.
    initial_capital: float = 100_000.0

    # Overlay thresholds (allow / reduce / block) for the exported scoring function.
    top_quantile: float = 0.70        # keep trades scoring in the top (1-q)
    reduce_size_factor: float = 0.50  # size multiplier in hostile/low-confidence state

    notes: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if isinstance(self.view, str):
            self.view = ModellingView(self.view)
        if isinstance(self.weight_mode, str):
            self.weight_mode = WeightMode(self.weight_mode)
        if isinstance(self.target, dict):
            self.target = TargetSpec.from_dict(self.target)
        if isinstance(self.adjusted_rar, dict):
            self.adjusted_rar = AdjustedRARConfig.from_dict(self.adjusted_rar)
        if isinstance(self.validation, dict):
            self.validation = ValidationConfig.from_dict(self.validation)
        if isinstance(self.ladder, dict):
            self.ladder = LadderConfig.from_dict(self.ladder)

    @property
    def model_run_id(self) -> str:
        return slugify(self.model_run_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_run_name": self.model_run_name,
            "model_run_id": self.model_run_id,
            "source_run_id": self.source_run_id,
            "runs_root": self.runs_root,
            "view": self.view.value,
            "feature_families": list(self.feature_families),
            "feature_columns": {k: list(v) for k, v in self.feature_columns.items()},
            "target": self.target.to_dict(),
            "adjusted_rar": self.adjusted_rar.to_dict(),
            "validation": self.validation.to_dict(),
            "ladder": self.ladder.to_dict(),
            "weight_mode": self.weight_mode.value,
            "initial_capital": self.initial_capital,
            "top_quantile": self.top_quantile,
            "reduce_size_factor": self.reduce_size_factor,
            "notes": self.notes,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModellingConfig":
        data = dict(data)
        return cls(
            model_run_name=data["model_run_name"],
            source_run_id=data.get("source_run_id", ""),
            runs_root=data.get("runs_root", "processed_data/runs"),
            view=ModellingView(data.get("view", "per_trade")),
            feature_families=list(data.get("feature_families", [])),
            feature_columns={k: list(v) for k, v in (data.get("feature_columns") or {}).items()},
            target=TargetSpec.from_dict(data.get("target", {})),
            adjusted_rar=AdjustedRARConfig.from_dict(data.get("adjusted_rar", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            ladder=LadderConfig.from_dict(data.get("ladder", {})),
            weight_mode=WeightMode(data.get("weight_mode", "equal")),
            initial_capital=data.get("initial_capital", 100_000.0),
            top_quantile=data.get("top_quantile", 0.70),
            reduce_size_factor=data.get("reduce_size_factor", 0.50),
            notes=data.get("notes", ""),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
