"""
Data Preparation GUI -- build a point-in-time research run package.

A six-step progressive-disclosure wizard (run setup -> trade selection -> data
family configuration -> mappings & timing -> validation & preview -> export)
that assembles the trade-centred, point-in-time run package the separate
modelling stage consumes. It only prepares and exports inputs; it does not train
or evaluate models.

Run standalone:  python ctk_data_prep_gui.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk

# Ensure the project root is importable when launched directly.
sys.path.insert(0, str(Path(__file__).parent))

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes
from Classes.GUI.ctk_wizard_base import CTkWizardBase, CTkWizardStep

from Classes.DataCollection.config import (
    COMMODITY_SERIES, CORE_COMMODITIES, MACRO_SERIES, APIConfig, CacheConfig,
)
from Classes.DataPrep.schema import Family
from Classes.DataPrep.run import RunConfig, ModellingFrequency
from Classes.DataPrep.timing import AvailabilityRule, MissingDataPolicy
from Classes.DataPrep.trade_source import TradeSource
from Classes.DataPrep.controller import DataPrepController
from Classes.DataPrep.validation import Severity

_AV_FAMILIES = {Family.COMMODITIES, Family.MACRO, Family.CORPORATE_ACTIONS, Family.UTILITIES}
_SEVERITY_COLOUR = {
    Severity.ERROR: Colors.ERROR,
    Severity.WARNING: getattr(Colors, "WARNING", "#d68910"),
    Severity.INFO: getattr(Colors, "TEXT_LINK", "#2471a3"),
}


def _load_api_key() -> str:
    """Best-effort read of the Alpha Vantage key from the data-collection config."""
    for path in (Path("config/data_collection/settings.json"),
                 Path("config/data_collection/settings.json.template")):
        try:
            if path.exists():
                data = json.loads(path.read_text())
                key = (data.get("api", {}) or {}).get("api_key") or data.get("api_key")
                if key:
                    return str(key)
        except Exception:
            continue
    return ""


# ===========================================================================
# Step 1 - Run setup
# ===========================================================================
class RunSetupStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        cfg = self.wizard.config
        Theme.create_hint(
            parent,
            "Name the run and set the global defaults. Everything else is tied to this run.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Run name *", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.name_var = ctk.StringVar(value=cfg.run_name)
        Theme.create_entry(parent, placeholder="e.g. momentum_v3_2026", textvariable=self.name_var
                           ).pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Reporting / base currency", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.ccy_var = ctk.StringVar(value=cfg.base_currency)
        Theme.create_optionmenu(parent, ["GBP", "USD", "EUR", "JPY", "CHF", "CAD", "AUD"],
                                variable=self.ccy_var).pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Intended modelling frequency", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.freq_var = ctk.StringVar(value=cfg.modelling_frequency.value)
        Theme.create_optionmenu(parent, [f.value for f in ModellingFrequency],
                                variable=self.freq_var).pack(anchor="w", pady=(0, Sizes.PAD_M))

        Theme.create_label(parent, "Notes (optional)", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.notes_box = Theme.create_textbox(parent, height=80)
        self.notes_box.pack(fill="x", pady=(0, Sizes.PAD_M))
        if cfg.notes:
            self.notes_box.insert("1.0", cfg.notes)

        Theme.create_label(parent, "Tags (comma-separated, optional)", font=Fonts.LABEL_BOLD).pack(anchor="w")
        self.tags_var = ctk.StringVar(value=", ".join(cfg.tags))
        Theme.create_entry(parent, placeholder="e.g. equities, us, daily", textvariable=self.tags_var
                           ).pack(fill="x")

    def _commit(self) -> None:
        cfg = self.wizard.config
        cfg.run_name = self.name_var.get().strip()
        cfg.base_currency = self.ccy_var.get()
        cfg.modelling_frequency = ModellingFrequency(self.freq_var.get())
        cfg.notes = self.notes_box.get("1.0", "end").strip()
        cfg.tags = [t.strip() for t in self.tags_var.get().split(",") if t.strip()]

    def validate(self) -> bool:
        self.validation_errors = []
        if not self.name_var.get().strip():
            self.validation_errors.append("A run name is required.")
            return False
        self._commit()
        return True

    def get_summary(self) -> Dict[str, str]:
        if not hasattr(self, "name_var"):
            return {}
        return {
            "Run name": self.name_var.get().strip() or "(unset)",
            "Base currency": self.ccy_var.get(),
            "Frequency": self.freq_var.get(),
        }


# ===========================================================================
# Step 2 - Trade selection
# ===========================================================================
class TradeSelectionStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(
            parent,
            "Select one or more backtest trade logs (from logs/). The dataset is "
            "trade-centred, so these trades anchor every feature.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        Theme.create_button(parent, "Refresh list", command=self._populate,
                            style="secondary", width=120).pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.list_frame = Theme.create_frame(parent)
        self.list_frame.pack(fill="x")

        Theme.create_button(parent, "Load selected trades", command=self._load,
                            style="primary", width=180).pack(anchor="w", pady=Sizes.PAD_M)

        self.preview = Theme.create_label(parent, "", font=Fonts.BODY_S,
                                          text_color=Colors.TEXT_SECONDARY, justify="left")
        self.preview.pack(anchor="w")

        self.log_vars: Dict[str, ctk.BooleanVar] = {}
        self.summary = None

    def on_enter(self) -> None:
        self._populate()

    def _populate(self) -> None:
        for w in self.list_frame.winfo_children():
            w.destroy()
        self.log_vars.clear()
        logs = TradeSource("logs").discover()
        if not logs:
            Theme.create_label(self.list_frame, "No trade logs found under logs/.",
                               text_color=Colors.TEXT_MUTED).pack(anchor="w")
            return
        for entry in logs:
            var = ctk.BooleanVar(value=False)
            self.log_vars[entry["path"]] = var
            Theme.create_checkbox(self.list_frame, f"[{entry['kind']}] {entry['name']}",
                                  variable=var).pack(anchor="w", pady=Sizes.PAD_XS)

    def _selected_paths(self) -> List[str]:
        return [p for p, v in self.log_vars.items() if v.get()]

    def _load(self) -> None:
        paths = self._selected_paths()
        if not paths:
            self.preview.configure(text="Select at least one trade log first.",
                                   text_color=Colors.ERROR)
            return
        ts = TradeSource("logs")
        trades, issues = ts.load(paths)
        self.summary = ts.summarise(trades, paths)
        self.wizard.config.trade_source = {"type": "logs", "paths": paths}
        self.wizard.selected_trade_paths = paths

        lines = [
            f"Trades: {self.summary.trade_count}    Symbols: {self.summary.n_symbols}",
            f"Date range: {self.summary.date_range[0]} -> {self.summary.date_range[1]}",
            f"Asset classes: {self.summary.asset_class_mix or 'n/a'}",
            f"Currencies: {', '.join(self.summary.currencies) or 'n/a'}",
        ]
        if issues:
            lines.append("Issues: " + "; ".join(issues))
        self.preview.configure(
            text="\n".join(lines),
            text_color=Colors.ERROR if issues else Colors.TEXT_SECONDARY,
        )
        self.wizard.trade_issues = issues

    def validate(self) -> bool:
        self.validation_errors = []
        if not getattr(self.wizard, "selected_trade_paths", None) or self.summary is None:
            self.validation_errors.append("Load a trade log before continuing.")
            return False
        if getattr(self.wizard, "trade_issues", None):
            self.validation_errors.append("Resolve the trade-log issues above before continuing.")
            return False
        return True

    def get_summary(self) -> Dict[str, str]:
        if not self.summary:
            return {}
        return {
            "Trades": str(self.summary.trade_count),
            "Symbols": str(self.summary.n_symbols),
            "Range": f"{self.summary.date_range[0]} -> {self.summary.date_range[1]}",
        }


# ===========================================================================
# Step 3 - Data family configuration
# ===========================================================================
class FamilyConfigStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(
            parent,
            "Toggle the data families to include. Commodities and macro come from "
            "Alpha Vantage; equities, FX, indices and fundamentals use your local data.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.include_vars: Dict[Family, ctk.BooleanVar] = {}
        self.series_vars: Dict[Family, Dict[str, ctk.BooleanVar]] = {}

        for fam in Family.ordered():
            self._family_card(parent, fam)

    def _family_card(self, parent, fam: Family) -> None:
        cfg = self.wizard.config.families[fam]
        card = Theme.create_card(parent)
        card.pack(fill="x", pady=Sizes.PAD_S)

        header = Theme.create_frame(card)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_XS))

        inc = ctk.BooleanVar(value=cfg.include)
        self.include_vars[fam] = inc
        Theme.create_switch(header, fam.label, variable=inc).pack(side="left")
        source = "Alpha Vantage" if fam in _AV_FAMILIES or fam == Family.FUNDAMENTALS else "Local data"
        Theme.create_label(header, source, font=Fonts.BODY_XS,
                           text_color=Colors.TEXT_MUTED).pack(side="right")

        # Series pickers for commodities & macro.
        catalogue = None
        default_series = []
        if fam == Family.COMMODITIES:
            catalogue, default_series = COMMODITY_SERIES, (cfg.series or CORE_COMMODITIES)
        elif fam == Family.MACRO:
            catalogue, default_series = MACRO_SERIES, (cfg.series or list(MACRO_SERIES.keys()))

        if catalogue:
            grid = Theme.create_frame(card)
            grid.pack(fill="x", padx=Sizes.PAD_L, pady=(0, Sizes.PAD_M))
            self.series_vars[fam] = {}
            for i, (key, spec) in enumerate(catalogue.items()):
                var = ctk.BooleanVar(value=key in default_series)
                self.series_vars[fam][key] = var
                label = f"{key} — {spec.get('rationale', spec.get('name', ''))}"
                Theme.create_checkbox(grid, label, variable=var).grid(
                    row=i // 2, column=i % 2, sticky="w", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)
        else:
            Theme.create_hint(card, self._family_hint(fam)).pack(
                anchor="w", padx=Sizes.PAD_L, pady=(0, Sizes.PAD_M))

    @staticmethod
    def _family_hint(fam: Family) -> str:
        return {
            Family.EQUITY_PRICES: "Local OHLCV from raw_data/daily. Raw vs adjusted is set on the next screen.",
            Family.CORPORATE_ACTIONS: "Dividends & splits as separate events (for adjusted/raw reconciliation and age features).",
            Family.FUNDAMENTALS: "Release-aware point-in-time fundamentals from processed_data/fundamentals (indexed by report date).",
            Family.INDEX: "Benchmarks & regime proxies (e.g. VIX). Mapped on the next screen.",
            Family.FX: "Conversion + FX regime. Your local GBP pairs are weekly; flagged for daily models.",
            Family.UTILITIES: "Support layer: market status & calendar helpers (not predictive features).",
        }.get(fam, "")

    def _commit(self) -> None:
        for fam, var in self.include_vars.items():
            self.wizard.config.families[fam].include = bool(var.get())
        for fam, series_map in self.series_vars.items():
            self.wizard.config.families[fam].series = [k for k, v in series_map.items() if v.get()]

    def validate(self) -> bool:
        self.validation_errors = []
        self._commit()
        if not self.wizard.config.included_families():
            self.validation_errors.append("Include at least one data family.")
            return False
        return True

    def get_summary(self) -> Dict[str, str]:
        if not self.include_vars:
            return {}
        inc = [fam.label for fam, v in self.include_vars.items() if v.get()]
        return {"Included": ", ".join(inc) if inc else "(none)"}


# ===========================================================================
# Step 4 - Mappings & timing
# ===========================================================================
class MappingsTimingStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(
            parent,
            "The expert screen: benchmark mapping and per-family timing rules. These "
            "decide how observation dates become point-in-time available timestamps.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        bm = Theme.create_card(parent)
        bm.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_label(bm, "Benchmark symbols for equities (comma-separated)",
                           font=Fonts.LABEL_BOLD).pack(anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, 0))
        existing = ",".join(self.wizard.config.benchmark_map.get("equity", []))
        self.bench_var = ctk.StringVar(value=existing or "SPX")
        Theme.create_entry(bm, placeholder="e.g. SPX, DJI", textvariable=self.bench_var
                           ).pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))
        self.vix_var = ctk.BooleanVar(value="VIX" in self.wizard.config.benchmark_map.get("regime", []))
        Theme.create_checkbox(bm, "Include VIX as a regime proxy", variable=self.vix_var
                              ).pack(anchor="w", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.timing_holder = Theme.create_frame(parent)
        self.timing_holder.pack(fill="x", pady=Sizes.PAD_S)
        self.timing_widgets: Dict[Family, Dict[str, ctk.Variable]] = {}

    def on_enter(self) -> None:
        for w in self.timing_holder.winfo_children():
            w.destroy()
        self.timing_widgets.clear()
        Theme.create_header(self.timing_holder, "Timing policy by included family", size="s").pack(anchor="w")

        for fam in self.wizard.config.included_families():
            self._timing_card(fam)

    def _timing_card(self, fam: Family) -> None:
        timing = self.wizard.config.families[fam].timing
        card = Theme.create_card(self.timing_holder)
        card.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(card, fam.label, font=Fonts.LABEL_BOLD).pack(
            anchor="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_S, 0))

        row = Theme.create_frame(card)
        row.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_S))

        rule_var = ctk.StringVar(value=timing.availability_rule.value)
        Theme.create_label(row, "availability", font=Fonts.BODY_XS,
                           text_color=Colors.TEXT_MUTED).grid(row=0, column=0, sticky="w")
        Theme.create_optionmenu(row, [r.value for r in AvailabilityRule], variable=rule_var,
                                width=160).grid(row=1, column=0, padx=(0, Sizes.PAD_S))

        lag_var = ctk.StringVar(value=str(timing.publication_lag_days))
        Theme.create_label(row, "pub lag (days)", font=Fonts.BODY_XS,
                           text_color=Colors.TEXT_MUTED).grid(row=0, column=1, sticky="w")
        Theme.create_entry(row, textvariable=lag_var, width=90).grid(row=1, column=1, padx=Sizes.PAD_S)

        tol_var = ctk.StringVar(value=str(timing.carry_forward_tolerance_days))
        Theme.create_label(row, "carry tol (days)", font=Fonts.BODY_XS,
                           text_color=Colors.TEXT_MUTED).grid(row=0, column=2, sticky="w")
        Theme.create_entry(row, textvariable=tol_var, width=90).grid(row=1, column=2, padx=Sizes.PAD_S)

        same_day_var = ctk.BooleanVar(value=timing.allow_same_day_close)
        Theme.create_checkbox(row, "same-day close usable", variable=same_day_var).grid(
            row=1, column=3, padx=Sizes.PAD_S)

        self.timing_widgets[fam] = {
            "rule": rule_var, "lag": lag_var, "tol": tol_var, "same_day": same_day_var,
        }

    def _commit(self) -> None:
        cfg = self.wizard.config
        benches = [b.strip().upper() for b in self.bench_var.get().split(",") if b.strip()]
        if benches:
            cfg.benchmark_map["equity"] = benches
        if self.vix_var.get():
            cfg.benchmark_map["regime"] = sorted(set(cfg.benchmark_map.get("regime", []) + ["VIX"]))
        elif "regime" in cfg.benchmark_map:
            cfg.benchmark_map["regime"] = [s for s in cfg.benchmark_map["regime"] if s != "VIX"]

        for fam, w in self.timing_widgets.items():
            timing = cfg.families[fam].timing
            timing.availability_rule = AvailabilityRule(w["rule"].get())
            timing.publication_lag_days = _safe_int(w["lag"].get(), timing.publication_lag_days)
            timing.carry_forward_tolerance_days = _safe_int(w["tol"].get(), timing.carry_forward_tolerance_days)
            timing.allow_same_day_close = bool(w["same_day"].get())

    def validate(self) -> bool:
        self.validation_errors = []
        self._commit()
        return True

    def get_summary(self) -> Dict[str, str]:
        if not hasattr(self, "bench_var"):
            return {}
        return {"Benchmarks": self.bench_var.get() or "(none)"}


# ===========================================================================
# Step 5 - Validation & preview
# ===========================================================================
class ValidationStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(
            parent,
            "Assemble the panels and run the pre-flight checklist. Errors block export; "
            "warnings should be reviewed.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        Theme.create_button(parent, "Assemble & validate", command=self._run,
                            style="primary", width=200).pack(anchor="w", pady=(0, Sizes.PAD_M))

        self.status = Theme.create_label(parent, "Not yet run.", font=Fonts.BODY_M,
                                         text_color=Colors.TEXT_SECONDARY)
        self.status.pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.results = Theme.create_frame(parent)
        self.results.pack(fill="both", expand=True)

    def _run(self) -> None:
        self.status.configure(text="Assembling panels and validating...",
                              text_color=Colors.TEXT_SECONDARY)
        self.wizard.root.update_idletasks()
        try:
            controller = self.wizard.build_controller()
            controller.load_trades(self.wizard.config.trade_source.get("paths", []))
            warnings = controller.assemble()
            report = controller.validate()
        except Exception as exc:  # pragma: no cover - UI feedback
            self.status.configure(text=f"Validation failed: {exc}", text_color=Colors.ERROR)
            traceback.print_exc()
            return

        self.wizard.controller = controller
        self.wizard.report = report
        counts = report.counts()
        self.status.configure(
            text=(f"Errors: {counts['error']}   Warnings: {counts['warning']}   "
                  f"Info: {counts['info']}   |   "
                  + ("BLOCKED" if report.is_blocking else "OK to export")),
            text_color=Colors.ERROR if report.is_blocking else getattr(Colors, "SUCCESS", "#2ecc71"),
        )
        self._render(report, warnings)

    def _render(self, report, assembly_warnings: List[str]) -> None:
        for w in self.results.winfo_children():
            w.destroy()

        if assembly_warnings:
            Theme.create_label(self.results, "Assembly notes:", font=Fonts.LABEL_BOLD).pack(anchor="w")
            for w in assembly_warnings:
                Theme.create_label(self.results, f"  • {w}", font=Fonts.BODY_XS,
                                   text_color=Colors.TEXT_MUTED).pack(anchor="w")

        Theme.create_label(self.results, "Coverage by family:", font=Fonts.LABEL_BOLD).pack(
            anchor="w", pady=(Sizes.PAD_S, 0))
        for fam, cov in report.coverage.items():
            Theme.create_label(
                self.results,
                f"  {fam}: {cov.get('rows', 0)} rows, {cov.get('entities', 0)} entities, "
                f"{cov.get('obs_start')} -> {cov.get('obs_end')}",
                font=Fonts.BODY_XS, text_color=Colors.TEXT_SECONDARY,
            ).pack(anchor="w")

        Theme.create_label(self.results, "Findings:", font=Fonts.LABEL_BOLD).pack(
            anchor="w", pady=(Sizes.PAD_S, 0))
        if not report.findings:
            Theme.create_label(self.results, "  No findings.", font=Fonts.BODY_XS,
                               text_color=Colors.TEXT_MUTED).pack(anchor="w")
        for f in report.findings:
            Theme.create_label(
                self.results, f"  [{f.severity.value}] {f.scope}: {f.message}",
                font=Fonts.BODY_XS, text_color=_SEVERITY_COLOUR.get(f.severity, Colors.TEXT_SECONDARY),
                wraplength=620, justify="left",
            ).pack(anchor="w")

    def validate(self) -> bool:
        self.validation_errors = []
        report = getattr(self.wizard, "report", None)
        if report is None:
            self.validation_errors.append("Run 'Assemble & validate' before continuing.")
            return False
        if report.is_blocking:
            self.validation_errors.append("Validation has errors that block export. Resolve them first.")
            return False
        return True

    def get_summary(self) -> Dict[str, str]:
        report = getattr(self.wizard, "report", None)
        if report is None:
            return {}
        c = report.counts()
        return {"Validation": f"{c['error']}E / {c['warning']}W / {c['info']}I"}


# ===========================================================================
# Step 6 - Export
# ===========================================================================
class ExportStep(CTkWizardStep):
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        Theme.create_hint(
            parent,
            "Export the self-describing run package (Parquet tables + manifest + data "
            "contract + validation report). Press Export to write it.",
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))
        self.info = Theme.create_label(parent, "", font=Fonts.BODY_M,
                                       text_color=Colors.TEXT_SECONDARY, justify="left")
        self.info.pack(anchor="w")

    def on_enter(self) -> None:
        cfg = self.wizard.config
        out = Path("processed_data/runs") / cfg.run_id
        included = ", ".join(f.label for f in cfg.included_families())
        self.info.configure(text=(
            f"Run id: {cfg.run_id}\n"
            f"Output directory: {out}\n"
            f"Included families: {included}\n"
            f"Acknowledged warnings will be recorded in the manifest."
        ))

    def get_summary(self) -> Dict[str, str]:
        return {"Output": str(Path('processed_data/runs') / self.wizard.config.run_id)}


# ===========================================================================
# Wizard
# ===========================================================================
class DataPrepWizard(CTkWizardBase):
    def __init__(self):
        super().__init__(title="Data Preparation — Research Run Builder", width=1180, height=820)
        self.config = RunConfig(run_name="")
        self.controller: Optional[DataPrepController] = None
        self.report = None
        self.selected_trade_paths: List[str] = []
        self.trade_issues: List[str] = []

        self.add_step(RunSetupStep(self, "Run setup"))
        self.add_step(TradeSelectionStep(self, "Trade selection"))
        self.add_step(FamilyConfigStep(self, "Data families"))
        self.add_step(MappingsTimingStep(self, "Mappings & timing"))
        self.add_step(ValidationStep(self, "Validation & preview"))
        self.add_step(ExportStep(self, "Export"))

        self.on_complete = self._export
        self.start()

    def _get_final_button_text(self) -> str:
        return "Export"

    def build_controller(self) -> DataPrepController:
        """Construct a controller, wiring an AV client when AV families are used."""
        av_client = None
        if any(f in _AV_FAMILIES for f in self.config.included_families()):
            key = _load_api_key()
            if key:
                from Classes.DataCollection.alpha_vantage_client import AlphaVantageClient
                av_client = AlphaVantageClient(APIConfig(api_key=key), CacheConfig())
        return DataPrepController(self.config, av_client=av_client)

    def _export(self) -> None:
        # Record acknowledged warnings, then write the package.
        if self.report is not None:
            self.config.acknowledged_warnings = [
                f"{f.scope}:{f.code}" for f in self.report.warnings
            ]
        try:
            controller = self.controller or self.build_controller()
            if self.controller is None:
                controller.load_trades(self.config.trade_source.get("paths", []))
                controller.assemble()
            package = controller.export(self.report)
        except Exception as exc:  # pragma: no cover - UI feedback
            self._dialog("Export failed", str(exc), error=True)
            traceback.print_exc()
            return
        self._dialog("Export complete",
                     f"Run package written to:\n{package.run_dir}\n\n"
                     f"Tables: {', '.join(package.output_files)}")

    def _dialog(self, title: str, message: str, error: bool = False) -> None:
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(title)
        dlg.geometry("520x240")
        dlg.transient(self.root)
        dlg.grab_set()
        Theme.create_label(dlg, message, font=Fonts.BODY_M,
                           text_color=Colors.ERROR if error else Colors.TEXT_PRIMARY,
                           wraplength=470, justify="left").pack(expand=True, padx=20, pady=20)
        Theme.create_button(dlg, "OK", command=dlg.destroy, width=100).pack(pady=(0, 20))


def _safe_int(text: str, default: int) -> int:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def main():
    DataPrepWizard().run()


if __name__ == "__main__":
    main()
