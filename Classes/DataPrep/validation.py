"""
Validation -- a pre-flight checklist, not an error dump.

Runs technical and *economic* checks over the assembled run before export:
coverage by family, missingness, stale series, unmapped trades/benchmarks,
base-currency feasibility, frequency mismatches and likely leakage scenarios.
Findings carry a severity so the GUI can block on errors and surface warnings
the user must acknowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from .schema import Family, QualityFlag
from .run import RunConfig, ModellingFrequency
from .timing import AvailabilityRule
from .entity_mapping import EntityMapping
from .trade_source import TradeSource

_LOW_FREQUENCIES = {"monthly", "quarterly", "annual", "semiannual", "weekly"}
_HIGH_NULL_FRACTION = 0.25


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """A single validation finding."""
    severity: Severity
    scope: str          # family value or "run" / "trades"
    code: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "scope": self.scope,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    """Aggregated validation outcome with a per-family coverage summary."""
    findings: List[Finding] = field(default_factory=list)
    coverage: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, severity: Severity, scope: str, code: str, message: str, **detail: Any) -> None:
        self.findings.append(Finding(severity, scope, code, message, dict(detail)))

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def is_blocking(self) -> bool:
        return len(self.errors) > 0

    def counts(self) -> Dict[str, int]:
        return {
            "error": len(self.errors),
            "warning": len(self.warnings),
            "info": len([f for f in self.findings if f.severity == Severity.INFO]),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": self.counts(),
            "is_blocking": self.is_blocking,
            "findings": [f.to_dict() for f in self.findings],
            "coverage": self.coverage,
        }


class Validator:
    """Runs the pre-flight checks for a run."""

    def validate(
        self,
        config: RunConfig,
        trades: pd.DataFrame,
        mapping: Optional[EntityMapping],
        panels: Dict[Family, pd.DataFrame],
    ) -> ValidationReport:
        report = ValidationReport()

        self._check_trades(report, trades)
        self._build_coverage(report, config, trades, panels)
        self._check_mapping(report, config, mapping)
        self._check_currency_feasibility(report, config, mapping)
        self._check_panels(report, config, trades, panels)
        self._check_frequency_mismatches(report, config, panels)
        self._check_leakage(report, config, trades, panels)

        return report

    # -- individual checks -------------------------------------------------- #
    def _check_trades(self, report: ValidationReport, trades: pd.DataFrame) -> None:
        for issue in TradeSource.validate(trades):
            report.add(Severity.ERROR, "trades", "trade_keys", issue)

        if not trades.empty and {"entry_date", "exit_date"} <= set(trades.columns):
            bad = trades["exit_date"] < trades["entry_date"]
            if bad.any():
                report.add(
                    Severity.WARNING, "trades", "exit_before_entry",
                    f"{int(bad.sum())} trade(s) have exit_date before entry_date.",
                    count=int(bad.sum()),
                )

    def _build_coverage(
        self,
        report: ValidationReport,
        config: RunConfig,
        trades: pd.DataFrame,
        panels: Dict[Family, pd.DataFrame],
    ) -> None:
        for fam in config.included_families():
            df = panels.get(fam)
            if df is None or df.empty:
                report.coverage[fam.value] = {"rows": 0, "entities": 0}
                report.add(
                    Severity.WARNING, fam.value, "no_data",
                    f"{fam.label} is included but produced no rows.",
                )
                continue

            obs = pd.to_datetime(df.get("observation_date"), errors="coerce")
            null_av = int(df["available_ts"].isna().sum()) if "available_ts" in df else len(df)
            cov = {
                "rows": int(len(df)),
                "entities": int(df["entity_id"].nunique()) if "entity_id" in df else 0,
                "obs_start": obs.min().strftime("%Y-%m-%d") if obs.notna().any() else None,
                "obs_end": obs.max().strftime("%Y-%m-%d") if obs.notna().any() else None,
                "null_available_ts": null_av,
            }
            report.coverage[fam.value] = cov

            if null_av:
                report.add(
                    Severity.WARNING, fam.value, "null_available_ts",
                    f"{fam.label}: {null_av} row(s) have no available_ts (cannot be joined as-of).",
                    count=null_av,
                )

            # Data ends before the trades do -> the tail of trades is uncovered.
            if not trades.empty and "entry_date" in trades.columns and obs.notna().any():
                last_trade = pd.to_datetime(trades["entry_date"], errors="coerce").max()
                if pd.notna(last_trade) and obs.max() < last_trade:
                    report.add(
                        Severity.WARNING, fam.value, "coverage_gap",
                        f"{fam.label} data ends {obs.max():%Y-%m-%d}, before the last trade "
                        f"({last_trade:%Y-%m-%d}); recent trades will be uncovered.",
                    )

    def _check_mapping(
        self,
        report: ValidationReport,
        config: RunConfig,
        mapping: Optional[EntityMapping],
    ) -> None:
        if Family.INDEX not in config.included_families():
            return
        if not config.benchmark_map:
            report.add(
                Severity.WARNING, Family.INDEX.value, "no_benchmark_map",
                "Index family is included but no benchmark mapping is defined.",
            )
            return
        if mapping and mapping.unmapped_benchmarks:
            n = len(mapping.unmapped_benchmarks)
            report.add(
                Severity.WARNING, Family.INDEX.value, "unmapped_benchmarks",
                f"{n} trade(s) have no benchmark mapping.",
                count=n, sample=mapping.unmapped_benchmarks[:10],
            )

    def _check_currency_feasibility(
        self,
        report: ValidationReport,
        config: RunConfig,
        mapping: Optional[EntityMapping],
    ) -> None:
        if mapping is None or mapping.empty or "needs_fx_conversion" not in mapping.table.columns:
            return
        n_needs = int(mapping.table["needs_fx_conversion"].sum())
        if n_needs and Family.FX not in config.included_families():
            report.add(
                Severity.WARNING, "run", "fx_conversion_infeasible",
                f"{n_needs} trade(s) are not in the base currency ({config.base_currency}) "
                "but the FX family is not included, so base-currency conversion is not possible.",
                count=n_needs,
            )

    def _check_panels(
        self,
        report: ValidationReport,
        config: RunConfig,
        trades: pd.DataFrame,
        panels: Dict[Family, pd.DataFrame],
    ) -> None:
        for fam in config.included_families():
            df = panels.get(fam)
            if df is None or df.empty:
                continue
            if "value" in df.columns:
                frac = float(df["value"].isna().mean())
                if frac >= _HIGH_NULL_FRACTION:
                    report.add(
                        Severity.WARNING, fam.value, "high_null_value",
                        f"{fam.label}: {frac:.0%} of values are null.",
                        null_fraction=round(frac, 4),
                    )
            if "quality_flag" in df.columns:
                stale = int((df["quality_flag"] == QualityFlag.STALE.value).sum())
                if stale:
                    report.add(
                        Severity.INFO, fam.value, "stale_rows",
                        f"{fam.label}: {stale} row(s) flagged stale.", count=stale,
                    )

    def _check_frequency_mismatches(
        self,
        report: ValidationReport,
        config: RunConfig,
        panels: Dict[Family, pd.DataFrame],
    ) -> None:
        daily = config.modelling_frequency == ModellingFrequency.DAILY

        # Weekly FX for a daily run.
        if daily and Family.FX in config.included_families():
            fx = panels.get(Family.FX)
            if fx is not None and not fx.empty and "native_frequency" in fx.columns:
                freqs = set(str(x).lower() for x in fx["native_frequency"].dropna().unique())
                if freqs & {"weekly", "monthly"}:
                    report.add(
                        Severity.WARNING, Family.FX.value, "weekly_fx_daily_model",
                        "Weekly/monthly FX is being used for a daily modelling setup; "
                        "base-currency conversion will be approximate.",
                        frequencies=sorted(freqs),
                    )

        # Low-frequency commodity/macro carried far.
        for fam in (Family.COMMODITIES, Family.MACRO):
            if fam not in config.included_families():
                continue
            df = panels.get(fam)
            if df is None or df.empty or "native_frequency" not in df.columns:
                continue
            freqs = set(str(x).lower() for x in df["native_frequency"].dropna().unique())
            low = freqs & _LOW_FREQUENCIES
            if low:
                tol = config.families[fam].timing.carry_forward_tolerance_days
                report.add(
                    Severity.INFO, fam.value, "low_frequency_carry",
                    f"{fam.label} includes low-frequency series ({', '.join(sorted(low))}); "
                    f"values are carried forward up to {tol} days.",
                    frequencies=sorted(low), carry_forward_tolerance_days=tol,
                )

    def _check_leakage(
        self,
        report: ValidationReport,
        config: RunConfig,
        trades: pd.DataFrame,
        panels: Dict[Family, pd.DataFrame],
    ) -> None:
        # Snapshot fundamentals on historical trades.
        if Family.FUNDAMENTALS in config.included_families():
            fcfg = config.families[Family.FUNDAMENTALS]
            uses_snapshot = bool(fcfg.options.get("use_overview_snapshot", False))
            if uses_snapshot and not fcfg.options.get("has_archived_snapshots", False):
                report.add(
                    Severity.WARNING, Family.FUNDAMENTALS.value, "snapshot_on_history",
                    "Present-day OVERVIEW snapshot fundamentals are enabled for historical "
                    "trades without archived snapshots -- this leaks future information.",
                )

        # Macro must be shifted to a conservative availability.
        if Family.MACRO in config.included_families():
            mcfg = config.families[Family.MACRO]
            # Only a real publication lag (or an explicit release date) is safe;
            # same-day / next-session / realtime would use releases too early.
            if mcfg.timing.effective_rule() not in (
                AvailabilityRule.PUBLICATION_LAG, AvailabilityRule.REPORT_DATE
            ):
                report.add(
                    Severity.WARNING, Family.MACRO.value, "macro_not_lagged",
                    "Macro indicators are not shifted by a publication lag; releases would be "
                    "used before they were public. Use a publication lag.",
                )

        # Same-day close assumptions: acknowledge, don't block.
        for fam in (Family.EQUITY_PRICES, Family.INDEX):
            if fam in config.included_families():
                if config.families[fam].timing.allow_same_day_close:
                    report.add(
                        Severity.INFO, fam.value, "same_day_close",
                        f"{fam.label}: same-day close values are treated as usable for same-day "
                        "decisions (valid for close-executed strategies; verify this matches "
                        "your execution model).",
                    )
