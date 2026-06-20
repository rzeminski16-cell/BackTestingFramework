"""
Run-package loader — the only thing that touches the data-prep contract.

The modelling stage consumes the named, self-contained package the data-prep
stage writes to ``processed_data/runs/<run_id>/`` and never rebuilds preparation
logic. This module reads the manifest, data contract, selected trades, entity
mapping, and per-family panels, and exposes a readiness check the GUI shows on
the "Choose run package" screen.

The package contract (see ``Classes/DataPrep`` and ``docs/DATA_PREPARATION.md``):
features join to trades with a **backward** ``merge_asof`` on ``available_ts``,
bounded by each family's ``carry_forward_tolerance_days`` — never calendar
equality.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

# Artefact filenames are fixed by the data-prep contract.
MANIFEST_FILE = "run_manifest.json"
CONTRACT_FILE = "data_contract.json"
FAMILY_CONFIG_FILE = "family_config.json"
TRADES_FILE = "selected_trades.parquet"
ENTITY_MAP_FILE = "entity_mapping.parquet"
VALIDATION_REPORT_FILE = "validation_report.json"

# Family table stems (value == filename stem in the package).
FAMILY_FILES: Dict[str, str] = {
    "equity_prices": "equity_prices.parquet",
    "corporate_actions": "corporate_actions.parquet",
    "fundamentals_pit": "fundamentals_pit.parquet",
    "index_panel": "index_panel.parquet",
    "fx_panel": "fx_panel.parquet",
    "commodities_panel": "commodities_panel.parquet",
    "macro_panel": "macro_panel.parquet",
    "utilities_panel": "utilities_panel.parquet",
}

TRADE_REQUIRED_COLUMNS = ["trade_id", "symbol", "entry_date", "exit_date"]


@dataclass
class PackageReadiness:
    """Result of the pre-flight readiness check shown on screen 1."""
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunPackage:
    """In-memory view of a loaded run package."""
    run_id: str
    path: str
    manifest: Dict[str, Any]
    contract: Dict[str, Any]
    trades: pd.DataFrame
    entity_mapping: pd.DataFrame
    panels: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def base_currency(self) -> str:
        return self.manifest.get("base_currency", "GBP")

    @property
    def modelling_frequency(self) -> str:
        return self.manifest.get("modelling_frequency", "daily")

    def family_timing(self, family: str) -> Dict[str, Any]:
        """Timing policy recorded for ``family`` (defaults if unrecorded)."""
        policies = self.manifest.get("timing_policies", {})
        return policies.get(family, {})

    def carry_forward_tolerance(self, family: str) -> int:
        """Max age (days) a value may be carried forward in an as-of join."""
        timing = self.family_timing(family)
        try:
            return int(timing.get("carry_forward_tolerance_days", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def available_families(self) -> List[str]:
        return [fam for fam, df in self.panels.items() if df is not None and not df.empty]

    def leakage_sensitive_fields(self) -> List[str]:
        return list(self.contract.get("leakage_sensitive_fields", []))


class RunPackageLoader:
    """Discovers and loads prepared run packages from a runs root directory."""

    def __init__(self, runs_root: str = "processed_data/runs"):
        self.runs_root = runs_root

    # -- discovery ---------------------------------------------------------- #
    def discover(self) -> List[Dict[str, Any]]:
        """List candidate run packages with light summary metadata."""
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(self.runs_root):
            return out
        for name in sorted(os.listdir(self.runs_root)):
            path = os.path.join(self.runs_root, name)
            manifest_path = os.path.join(path, MANIFEST_FILE)
            if not os.path.isfile(manifest_path):
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (OSError, json.JSONDecodeError):
                manifest = {}
            counts = manifest.get("table_row_counts", {})
            out.append({
                "run_id": manifest.get("run_id", name),
                "path": path,
                "generated_at": manifest.get("generated_at", ""),
                "base_currency": manifest.get("base_currency", ""),
                "modelling_frequency": manifest.get("modelling_frequency", ""),
                "n_trades": counts.get("selected_trades", 0),
                "families": [f for f, on in manifest.get("family_toggles", {}).items() if on],
                "notes": manifest.get("notes", ""),
            })
        return out

    # -- loading ------------------------------------------------------------ #
    def _read_json(self, path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _read_parquet(self, path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load(self, run_id: str) -> RunPackage:
        """Load a full run package by id."""
        path = os.path.join(self.runs_root, run_id)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run package not found: {path}")

        manifest = self._read_json(os.path.join(path, MANIFEST_FILE))
        contract = self._read_json(os.path.join(path, CONTRACT_FILE))
        trades = self._read_parquet(os.path.join(path, TRADES_FILE))
        entity_map = self._read_parquet(os.path.join(path, ENTITY_MAP_FILE))

        trades = self._coerce_trade_dates(trades)

        panels: Dict[str, pd.DataFrame] = {}
        for fam, fname in FAMILY_FILES.items():
            df = self._read_parquet(os.path.join(path, fname))
            if not df.empty:
                df = self._coerce_panel_dates(df)
                panels[fam] = df

        return RunPackage(
            run_id=manifest.get("run_id", run_id),
            path=path,
            manifest=manifest,
            contract=contract,
            trades=trades,
            entity_mapping=entity_map,
            panels=panels,
        )

    @staticmethod
    def _coerce_trade_dates(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        trades = trades.copy()
        for col in ("entry_date", "exit_date"):
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col], errors="coerce")
        return trades

    @staticmethod
    def _coerce_panel_dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ("observation_date", "available_ts", "report_date", "retrieved_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    # -- readiness ---------------------------------------------------------- #
    def check_readiness(self, package: RunPackage) -> PackageReadiness:
        """Validate a loaded package is usable for modelling."""
        errors: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        trades = package.trades
        if trades.empty:
            errors.append("Run package contains no selected trades.")
        else:
            missing = [c for c in TRADE_REQUIRED_COLUMNS if c not in trades.columns]
            if missing:
                errors.append(f"selected_trades is missing required columns: {missing}")
            if "pl" not in trades.columns and "pl_pct" not in trades.columns:
                errors.append("selected_trades has no P/L column (need 'pl' or 'pl_pct').")
            info["n_trades"] = int(len(trades))
            if "entry_date" in trades.columns:
                ed = trades["entry_date"].dropna()
                if not ed.empty:
                    info["date_range"] = [str(ed.min().date()), str(ed.max().date())]
            if "symbol" in trades.columns:
                info["n_symbols"] = int(trades["symbol"].nunique())

        present = package.available_families()
        info["families_present"] = present
        if not present:
            warnings.append(
                "No feature-family panels are present — only trade-intrinsic "
                "features will be available."
            )

        # Sanity: panels carry the as-of join key.
        for fam in present:
            df = package.panels[fam]
            if "available_ts" not in df.columns:
                warnings.append(f"Family '{fam}' has no available_ts column; it will be skipped.")
            elif df["available_ts"].isna().all():
                warnings.append(f"Family '{fam}' has all-null available_ts; it cannot be joined.")

        return PackageReadiness(ok=not errors, errors=errors, warnings=warnings, info=info)
