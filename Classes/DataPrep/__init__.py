"""
DataPrep -- point-in-time data-preparation pipeline for strategy diagnostics.

This is a *standalone* pipeline (independent of ``Classes/FactorAnalysis``) whose
sole job is to assemble a named, reproducible, point-in-time-correct research
*run package*: a self-describing directory of normalised per-family tables,
trade/entity mappings, a manifest, a machine-readable data contract, and a
validation report. A separate modelling stage consumes that package; nothing in
this package trains or evaluates models.

Design pillars (see the data-preparation spec):
  * The *run* is the primary object -- everything is tied to a named run.
  * The dataset is *trade-centred*, not price-panel-centred.
  * Every feature row preserves both an observation timestamp and an earliest
    usable (``available_ts``) timestamp so downstream joins can be as-of joins.
  * The exported package is reproducible, auditable and self-describing.

Public surface is intentionally small; the GUI and any CLI build on these.
"""

from .schema import (
    Family,
    FAMILY_TABLE_FILE,
    PACKAGE_ARTEFACTS,
    PROVENANCE_COLUMNS,
    TRADE_REQUIRED_COLUMNS,
    QualityFlag,
)
from .run import (
    ModellingFrequency,
    RunConfig,
    FamilyConfig,
    RunManifest,
)
from .timing import (
    TimingPolicy,
    MissingDataPolicy,
    AvailabilityRule,
    compute_available_ts,
    DEFAULT_FAMILY_TIMING,
)
from .trade_source import TradeSource, TradeUniverseSummary
from .entity_mapping import EntityMapper, EntityMapping
from .families import normalise_family_panel, stamp_provenance
from .validation import Validator, ValidationReport, Finding, Severity
from .package_writer import PackageWriter
from .builder import RunBuilder, RunPackage, ExportBlocked
from .sources import PanelSourceBuilder
from .controller import DataPrepController

__all__ = [
    # schema
    "Family",
    "FAMILY_TABLE_FILE",
    "PACKAGE_ARTEFACTS",
    "PROVENANCE_COLUMNS",
    "TRADE_REQUIRED_COLUMNS",
    "QualityFlag",
    # run / manifest
    "ModellingFrequency",
    "RunConfig",
    "FamilyConfig",
    "RunManifest",
    # timing
    "TimingPolicy",
    "MissingDataPolicy",
    "AvailabilityRule",
    "compute_available_ts",
    "DEFAULT_FAMILY_TIMING",
    # trades / mapping
    "TradeSource",
    "TradeUniverseSummary",
    "EntityMapper",
    "EntityMapping",
    # families
    "normalise_family_panel",
    "stamp_provenance",
    # validation
    "Validator",
    "ValidationReport",
    "Finding",
    "Severity",
    # output
    "PackageWriter",
    # orchestration
    "RunBuilder",
    "RunPackage",
    "ExportBlocked",
    "PanelSourceBuilder",
    "DataPrepController",
]
