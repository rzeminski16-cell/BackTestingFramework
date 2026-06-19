"""
Canonical schema for the data-preparation run package.

Defines the data families, the run-package artefact inventory, the provenance/
timing columns every family table must carry, the required trade keys, and the
quality-flag vocabulary. Everything else in ``Classes/DataPrep`` references these
constants so the contract has a single source of truth.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class Family(str, Enum):
    """The data families a run can include.

    Values double as the table/stem name used in the exported package.
    """
    EQUITY_PRICES = "equity_prices"
    CORPORATE_ACTIONS = "corporate_actions"
    FUNDAMENTALS = "fundamentals_pit"
    INDEX = "index_panel"
    FX = "fx_panel"
    COMMODITIES = "commodities_panel"
    MACRO = "macro_panel"
    UTILITIES = "utilities_panel"

    @property
    def label(self) -> str:
        """Human-readable family name for the GUI / reports."""
        return _FAMILY_LABELS[self]

    @classmethod
    def ordered(cls) -> List["Family"]:
        """Families in the canonical configuration / display order."""
        return [
            cls.EQUITY_PRICES, cls.CORPORATE_ACTIONS, cls.FUNDAMENTALS,
            cls.INDEX, cls.FX, cls.COMMODITIES, cls.MACRO, cls.UTILITIES,
        ]


_FAMILY_LABELS: Dict[Family, str] = {
    Family.EQUITY_PRICES: "Equity price history",
    Family.CORPORATE_ACTIONS: "Corporate actions",
    Family.FUNDAMENTALS: "Fundamentals (point-in-time)",
    Family.INDEX: "Index data",
    Family.FX: "FX",
    Family.COMMODITIES: "Commodities",
    Family.MACRO: "Macro",
    Family.UTILITIES: "Utilities",
}

# Family -> exported parquet filename.
FAMILY_TABLE_FILE: Dict[Family, str] = {
    fam: f"{fam.value}.parquet" for fam in Family
}

# Full run-package artefact inventory (artefact key -> filename). The modelling
# stage should never have to guess what was exported; this is the contract.
PACKAGE_ARTEFACTS: Dict[str, str] = {
    "run_manifest": "run_manifest.json",
    "family_config": "family_config.json",
    "data_contract": "data_contract.json",
    "selected_trades": "selected_trades.parquet",
    "entity_mapping": "entity_mapping.parquet",
    **{f"{fam.value}": FAMILY_TABLE_FILE[fam] for fam in Family},
    "validation_report": "validation_report.json",
    "validation_summary": "validation_summary.html",
}

# Provenance + timing columns every family panel must carry. These are what make
# the package point-in-time-safe and auditable. ``observation_date`` is the
# period the value describes; ``available_ts`` is the earliest moment the value
# could have been known in live trading (the as-of join key).
PROVENANCE_COLUMNS: List[str] = [
    "run_id",
    "family",
    "entity_id",
    "observation_date",
    "available_ts",
    "native_frequency",
    "source_function",
    "source_vendor",
    "retrieved_at",
    "quality_flag",
]

# Minimum trade keys required from the system trade source. Without these the
# run cannot build safe as-of joins, so their absence is a hard failure.
TRADE_REQUIRED_COLUMNS: List[str] = [
    "trade_id",
    "symbol",
    "entry_date",
    "exit_date",
]

# Leakage-sensitive fields the data contract flags for the modelling stage.
LEAKAGE_SENSITIVE_FIELDS: List[str] = [
    "available_ts",
    "observation_date",
    "report_date",
    "report_time",
    "retrieved_at",
    "revision_risk_flag",
]


class QualityFlag(str, Enum):
    """Row-level quality / provenance markers."""
    OK = "ok"
    CARRIED_FORWARD = "carried_forward"   # held from a prior release/observation
    STALE = "stale"                       # older than the family freshness budget
    MISSING = "missing"                   # value absent
    INFERRED_TS = "inferred_timestamp"    # available_ts inferred (no exact release time)
    REVISION_RISK = "revision_risk"       # latest-history value, not a true vintage
    SNAPSHOT = "snapshot"                 # current-snapshot descriptor (e.g. OVERVIEW)
