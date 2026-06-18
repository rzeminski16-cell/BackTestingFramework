"""
RunBuilder -- orchestrate a run from config + inputs to a written package.

The builder is the single entry point the GUI (and any CLI) drives:

    builder = RunBuilder(config)
    builder.set_trades(trades_df)
    builder.add_panel(Family.COMMODITIES, raw_commodities_df, value_col="value", ...)
    report = builder.validate()
    package = builder.export(report)        # blocks if report has errors

It normalises each family panel through the run's timing policy, builds the
entity mapping, runs validation, and writes the package. It performs no model
training -- that is the downstream stage's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .schema import Family
from .run import RunConfig
from .families import normalise_family_panel
from .entity_mapping import EntityMapper, EntityMapping
from .validation import Validator, ValidationReport
from .package_writer import PackageWriter


class ExportBlocked(RuntimeError):
    """Raised when export is attempted while the validation report has errors."""


@dataclass
class RunPackage:
    """The result of a successful export."""
    run_dir: str
    output_files: Dict[str, str]
    row_counts: Dict[str, int]
    manifest: Dict[str, Any]
    report: ValidationReport = field(default=None)  # type: ignore[assignment]


class RunBuilder:
    """Assembles and exports a data-preparation run package."""

    def __init__(self, config: RunConfig, output_root: str = "processed_data/runs"):
        self.config = config
        self.trades: pd.DataFrame = pd.DataFrame()
        self.panels: Dict[Family, pd.DataFrame] = {}
        self.mapping: Optional[EntityMapping] = None
        self._writer = PackageWriter(output_root)
        self._validator = Validator()

    # -- inputs ------------------------------------------------------------- #
    def set_trades(self, trades: pd.DataFrame) -> "RunBuilder":
        self.trades = trades if trades is not None else pd.DataFrame()
        self.mapping = None  # invalidate any prior mapping
        return self

    def add_panel(self, family: Family, raw_df: pd.DataFrame, **normalise_kwargs: Any) -> "RunBuilder":
        """Normalise a raw family panel through the run's timing policy and store it."""
        if family not in self.config.families:
            raise KeyError(f"family {family} is not configured for this run")
        timing = self.config.families[family].timing
        self.panels[family] = normalise_family_panel(
            raw_df, family=family, run_id=self.config.run_id, timing=timing, **normalise_kwargs
        )
        return self

    def add_normalised_panel(self, family: Family, panel: pd.DataFrame) -> "RunBuilder":
        """Store an already-normalised panel (must carry the provenance columns)."""
        self.panels[family] = panel
        return self

    # -- assembly ----------------------------------------------------------- #
    def build_mapping(self) -> EntityMapping:
        self.mapping = EntityMapper(
            benchmark_map=self.config.benchmark_map,
            currency_map=self.config.currency_map,
            sector_map=self.config.sector_map,
            base_currency=self.config.base_currency,
        ).build(self.trades)
        return self.mapping

    def validate(self) -> ValidationReport:
        if self.mapping is None:
            self.build_mapping()
        return self._validator.validate(self.config, self.trades, self.mapping, self.panels)

    def export(self, report: Optional[ValidationReport] = None) -> RunPackage:
        """
        Write the package. Re-validates if no report is supplied. Raises
        :class:`ExportBlocked` if the (final) report contains errors.
        """
        if report is None:
            report = self.validate()
        if report.is_blocking:
            raise ExportBlocked(
                f"Export blocked: {report.counts()['error']} validation error(s). "
                "Resolve them before exporting."
            )
        if self.mapping is None:
            self.build_mapping()

        result = self._writer.write(self.config, self.trades, self.mapping, self.panels, report)
        return RunPackage(
            run_dir=result["run_dir"],
            output_files=result["output_files"],
            row_counts=result["row_counts"],
            manifest=result["manifest"],
            report=report,
        )
