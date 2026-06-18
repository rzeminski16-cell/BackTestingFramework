"""
DataPrepController -- headless orchestration the GUI (or a CLI) drives.

Holds the run state and sequences the pipeline: discover/load trades, assemble
family panels (local CSVs + Alpha Vantage), validate, and export the package.
Keeping this GUI-free makes the whole flow unit-testable without a display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .schema import Family
from .run import RunConfig
from .trade_source import TradeSource, TradeUniverseSummary
from .sources import PanelSourceBuilder
from .builder import RunBuilder, RunPackage
from .validation import ValidationReport


class DataPrepController:
    """Stateful façade over the DataPrep pipeline for UI use."""

    def __init__(
        self,
        config: RunConfig,
        logs_dir: str = "logs",
        raw_data_dir: str = "raw_data",
        processed_dir: str = "processed_data",
        av_client: Any = None,
        output_root: str = "processed_data/runs",
    ):
        self.config = config
        self.trade_source = TradeSource(logs_dir)
        self.panel_builder = PanelSourceBuilder(raw_data_dir, processed_dir, av_client)
        self.output_root = output_root

        self.trades: pd.DataFrame = pd.DataFrame()
        self.trade_summary: Optional[TradeUniverseSummary] = None
        self.panels: Dict[Family, pd.DataFrame] = {}
        self.assembly_warnings: List[str] = []
        self._builder: Optional[RunBuilder] = None

    # -- trades ------------------------------------------------------------- #
    def discover_trade_logs(self) -> List[Dict[str, Any]]:
        return self.trade_source.discover()

    def load_trades(self, paths: List[str]) -> Tuple[Optional[TradeUniverseSummary], List[str]]:
        """Load the selected trade logs; returns ``(summary, issues)``."""
        self.trades, issues = self.trade_source.load(paths)
        source_files = [str(p) for p in paths]
        self.config.trade_source = {"type": "logs", "paths": source_files}
        self.trade_summary = self.trade_source.summarise(self.trades, source_files)
        return self.trade_summary, issues

    def equity_symbols(self) -> List[str]:
        if self.trades.empty or "symbol" not in self.trades.columns:
            return []
        if "asset_class" in self.trades.columns:
            mask = self.trades["asset_class"].fillna("equity") == "equity"
            return sorted(self.trades.loc[mask, "symbol"].dropna().unique().tolist())
        return sorted(self.trades["symbol"].dropna().unique().tolist())

    def currencies(self) -> List[str]:
        if self.trade_summary:
            return self.trade_summary.currencies
        return []

    # -- assembly / validation / export ------------------------------------ #
    def assemble(self) -> List[str]:
        """Build all included family panels. Returns assembly warnings."""
        self.panels, self.assembly_warnings = self.panel_builder.build_all(
            self.config, self.equity_symbols(), self.currencies()
        )
        self._builder = RunBuilder(self.config, output_root=self.output_root)
        self._builder.set_trades(self.trades)
        for fam, panel in self.panels.items():
            self._builder.add_normalised_panel(fam, panel)
        self._builder.build_mapping()
        return self.assembly_warnings

    def validate(self) -> ValidationReport:
        if self._builder is None:
            self.assemble()
        return self._builder.validate()

    def export(self, report: Optional[ValidationReport] = None) -> RunPackage:
        if self._builder is None:
            self.assemble()
        return self._builder.export(report)
