"""
Run configuration and manifest model.

The *run* is the primary object: a named research run carries the base currency,
modelling frequency, the trade source, per-family configuration (including timing
policy), and the benchmark/currency/sector mappings. The :class:`RunManifest` is
the self-describing record written into the exported package so the modelling
stage never has to guess what was selected or how timing rules were applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .schema import Family
from .timing import TimingPolicy, default_timing_for


class ModellingFrequency(str, Enum):
    """Intended cadence the exported package is optimised for."""
    DAILY = "daily"
    WEEKLY = "weekly"
    INTRADAY = "intraday"


def slugify(name: str) -> str:
    """Turn a run name into a filesystem-safe, stable run identifier."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return slug or "unnamed_run"


@dataclass
class FamilyConfig:
    """Configuration for one data family within a run.

    Mirrors the per-family card in the GUI: include/exclude, source, field scope,
    selected series (for commodities/macro), timing policy, and free-form notes.
    """
    family: Family
    include: bool = False
    source: str = "alpha_vantage"
    field_scope: List[str] = field(default_factory=list)
    series: List[str] = field(default_factory=list)
    timing: TimingPolicy = None  # type: ignore[assignment]
    options: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if isinstance(self.family, str):
            self.family = Family(self.family)
        if self.timing is None:
            self.timing = default_timing_for(self.family)
        elif isinstance(self.timing, dict):
            self.timing = TimingPolicy.from_dict(self.timing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family.value,
            "include": self.include,
            "source": self.source,
            "field_scope": list(self.field_scope),
            "series": list(self.series),
            "timing": self.timing.to_dict(),
            "options": dict(self.options),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamilyConfig":
        data = dict(data)
        data["family"] = Family(data["family"])
        if "timing" in data and isinstance(data["timing"], dict):
            data["timing"] = TimingPolicy.from_dict(data["timing"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RunConfig:
    """Top-level configuration for a research run."""
    run_name: str
    base_currency: str = "GBP"
    modelling_frequency: ModellingFrequency = ModellingFrequency.DAILY
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Where the trades came from (descriptor, not the trades themselves).
    trade_source: Dict[str, Any] = field(default_factory=dict)

    families: Dict[Family, FamilyConfig] = field(default_factory=dict)

    # Mappings surfaced in the GUI and written to the manifest.
    benchmark_map: Dict[str, List[str]] = field(default_factory=dict)   # group -> [index symbols]
    currency_map: Dict[str, str] = field(default_factory=dict)          # symbol -> currency
    sector_map: Dict[str, str] = field(default_factory=dict)            # symbol -> sector

    # Warnings the user explicitly acknowledged before allowing export.
    acknowledged_warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.modelling_frequency, str):
            self.modelling_frequency = ModellingFrequency(self.modelling_frequency)
        if not self.families:
            self.families = self.default_families()
        else:
            # Coerce any dict-shaped family configs.
            self.families = {
                (k if isinstance(k, Family) else Family(k)):
                (v if isinstance(v, FamilyConfig) else FamilyConfig.from_dict(v))
                for k, v in self.families.items()
            }

    @property
    def run_id(self) -> str:
        """Stable, filesystem-safe identifier derived from the run name."""
        return slugify(self.run_name)

    @staticmethod
    def default_families() -> Dict[Family, FamilyConfig]:
        """A FamilyConfig per family, all excluded by default with default timing."""
        return {fam: FamilyConfig(family=fam, include=False) for fam in Family.ordered()}

    def included_families(self) -> List[Family]:
        return [fam for fam in Family.ordered()
                if fam in self.families and self.families[fam].include]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "base_currency": self.base_currency,
            "modelling_frequency": self.modelling_frequency.value,
            "notes": self.notes,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "trade_source": dict(self.trade_source),
            "families": {fam.value: cfg.to_dict() for fam, cfg in self.families.items()},
            "benchmark_map": {k: list(v) for k, v in self.benchmark_map.items()},
            "currency_map": dict(self.currency_map),
            "sector_map": dict(self.sector_map),
            "acknowledged_warnings": list(self.acknowledged_warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        data = dict(data)
        families = {
            Family(k): FamilyConfig.from_dict(v)
            for k, v in (data.get("families") or {}).items()
        }
        return cls(
            run_name=data["run_name"],
            base_currency=data.get("base_currency", "GBP"),
            modelling_frequency=ModellingFrequency(data.get("modelling_frequency", "daily")),
            notes=data.get("notes", ""),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            trade_source=dict(data.get("trade_source", {})),
            families=families,
            benchmark_map={k: list(v) for k, v in (data.get("benchmark_map") or {}).items()},
            currency_map=dict(data.get("currency_map", {})),
            sector_map=dict(data.get("sector_map", {})),
            acknowledged_warnings=list(data.get("acknowledged_warnings", [])),
        )


@dataclass
class RunManifest:
    """The self-describing record written to ``run_manifest.json``."""
    run_name: str
    run_id: str
    generated_at: str
    base_currency: str
    modelling_frequency: str
    trade_source: Dict[str, Any]
    family_toggles: Dict[str, bool]
    timing_policies: Dict[str, Dict[str, Any]]
    missing_data_policies: Dict[str, str]
    benchmark_map: Dict[str, List[str]]
    currency_map: Dict[str, str]
    sector_map: Dict[str, str]
    acknowledged_warnings: List[str]
    output_files: Dict[str, str]
    table_row_counts: Dict[str, int]
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    package_version: str = "1.0"

    @classmethod
    def from_run(
        cls,
        config: RunConfig,
        output_files: Dict[str, str],
        table_row_counts: Dict[str, int],
    ) -> "RunManifest":
        return cls(
            run_name=config.run_name,
            run_id=config.run_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            base_currency=config.base_currency,
            modelling_frequency=config.modelling_frequency.value,
            trade_source=dict(config.trade_source),
            family_toggles={fam.value: cfg.include for fam, cfg in config.families.items()},
            timing_policies={
                fam.value: cfg.timing.to_dict()
                for fam, cfg in config.families.items() if cfg.include
            },
            missing_data_policies={
                fam.value: cfg.timing.missing_data_policy.value
                for fam, cfg in config.families.items() if cfg.include
            },
            benchmark_map={k: list(v) for k, v in config.benchmark_map.items()},
            currency_map=dict(config.currency_map),
            sector_map=dict(config.sector_map),
            acknowledged_warnings=list(config.acknowledged_warnings),
            output_files=dict(output_files),
            table_row_counts=dict(table_row_counts),
            notes=config.notes,
            tags=list(config.tags),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "base_currency": self.base_currency,
            "modelling_frequency": self.modelling_frequency,
            "trade_source": self.trade_source,
            "family_toggles": self.family_toggles,
            "timing_policies": self.timing_policies,
            "missing_data_policies": self.missing_data_policies,
            "benchmark_map": self.benchmark_map,
            "currency_map": self.currency_map,
            "sector_map": self.sector_map,
            "acknowledged_warnings": self.acknowledged_warnings,
            "output_files": self.output_files,
            "table_row_counts": self.table_row_counts,
            "notes": self.notes,
            "tags": self.tags,
            "package_version": self.package_version,
        }
