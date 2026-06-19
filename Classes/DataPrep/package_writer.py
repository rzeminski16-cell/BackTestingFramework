"""
Package writer -- emit the self-describing run package.

Writes a run directory containing the manifest, the selected trades, the entity
mapping, one normalised Parquet table per included family, a machine-readable
data contract, and both machine- and human-readable validation outputs. This is
the sole hand-off to the modelling stage.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .schema import (
    Family,
    FAMILY_TABLE_FILE,
    PACKAGE_ARTEFACTS,
    PROVENANCE_COLUMNS,
    LEAKAGE_SENSITIVE_FIELDS,
)
from .run import RunConfig, RunManifest
from .entity_mapping import EntityMapping
from .validation import ValidationReport


class PackageWriter:
    """Writes a complete run package to disk under ``output_root/<run_id>``."""

    def __init__(self, output_root: str = "processed_data/runs"):
        self.output_root = Path(output_root)

    def write(
        self,
        config: RunConfig,
        trades: pd.DataFrame,
        mapping: Optional[EntityMapping],
        panels: Dict[Family, pd.DataFrame],
        report: ValidationReport,
    ) -> Dict[str, Any]:
        """
        Write the package and return ``{run_dir, output_files, row_counts, manifest}``.

        Only included families with non-empty panels are written; the manifest
        records exactly what was emitted.
        """
        run_dir = self.output_root / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        output_files: Dict[str, str] = {}
        row_counts: Dict[str, int] = {}

        # selected_trades
        self._write_parquet(trades, run_dir / PACKAGE_ARTEFACTS["selected_trades"])
        output_files["selected_trades"] = PACKAGE_ARTEFACTS["selected_trades"]
        row_counts["selected_trades"] = int(len(trades)) if trades is not None else 0

        # entity_mapping
        if mapping is not None and not mapping.empty:
            self._write_parquet(mapping.table, run_dir / PACKAGE_ARTEFACTS["entity_mapping"])
            output_files["entity_mapping"] = PACKAGE_ARTEFACTS["entity_mapping"]
            row_counts["entity_mapping"] = int(len(mapping.table))

        # family panels (only included + non-empty)
        for fam in config.included_families():
            df = panels.get(fam)
            if df is None or df.empty:
                continue
            fname = FAMILY_TABLE_FILE[fam]
            self._write_parquet(df, run_dir / fname)
            output_files[fam.value] = fname
            row_counts[fam.value] = int(len(df))

        # validation report (json) + summary (html)
        (run_dir / PACKAGE_ARTEFACTS["validation_report"]).write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        output_files["validation_report"] = PACKAGE_ARTEFACTS["validation_report"]
        (run_dir / PACKAGE_ARTEFACTS["validation_summary"]).write_text(
            self._render_summary_html(config, report), encoding="utf-8"
        )
        output_files["validation_summary"] = PACKAGE_ARTEFACTS["validation_summary"]

        # family_config
        family_config = {fam.value: cfg.to_dict() for fam, cfg in config.families.items()}
        (run_dir / PACKAGE_ARTEFACTS["family_config"]).write_text(
            json.dumps(family_config, indent=2, default=str), encoding="utf-8"
        )
        output_files["family_config"] = PACKAGE_ARTEFACTS["family_config"]

        # data_contract
        (run_dir / PACKAGE_ARTEFACTS["data_contract"]).write_text(
            json.dumps(self._build_data_contract(config, output_files, panels), indent=2, default=str),
            encoding="utf-8",
        )
        output_files["data_contract"] = PACKAGE_ARTEFACTS["data_contract"]

        # manifest (written last so it can list every output file)
        manifest = RunManifest.from_run(config, output_files, row_counts)
        (run_dir / PACKAGE_ARTEFACTS["run_manifest"]).write_text(
            json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        output_files["run_manifest"] = PACKAGE_ARTEFACTS["run_manifest"]

        return {
            "run_dir": str(run_dir),
            "output_files": output_files,
            "row_counts": row_counts,
            "manifest": manifest.to_dict(),
        }

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _write_parquet(df: pd.DataFrame, path: Path) -> None:
        frame = df if df is not None else pd.DataFrame()
        # Object columns holding pd.NA only would become an all-null column;
        # parquet handles that fine. Stringify any leftover complex objects.
        frame.to_parquet(path, index=False)

    def _build_data_contract(
        self,
        config: RunConfig,
        output_files: Dict[str, str],
        panels: Dict[Family, pd.DataFrame],
    ) -> Dict[str, Any]:
        families: Dict[str, Any] = {}
        for fam in config.included_families():
            cfg = config.families[fam]
            df = panels.get(fam)
            native_freqs = []
            units = []
            if df is not None and not df.empty:
                if "native_frequency" in df.columns:
                    native_freqs = sorted(str(x) for x in df["native_frequency"].dropna().unique())
                if "unit" in df.columns:
                    units = sorted(str(x) for x in df["unit"].dropna().unique())[:20]
            families[fam.value] = {
                "table_file": output_files.get(fam.value),
                "timing": cfg.timing.to_dict(),
                "missing_data_policy": cfg.timing.missing_data_policy.value,
                "native_frequencies": native_freqs,
                "units": units,
            }

        return {
            "package_version": "1.0",
            "run_id": config.run_id,
            "base_currency": config.base_currency,
            "modelling_frequency": config.modelling_frequency.value,
            "join_semantics": (
                "Join features to trades with an as-of BACKWARD merge on available_ts "
                "(merge_asof), bounded by each family's carry_forward_tolerance_days. "
                "Never join on calendar-date equality."
            ),
            "timestamp_semantics": {
                "observation_date": "The period/observation the value describes.",
                "available_ts": "Earliest moment the value could have been known in live "
                                "trading; this is the as-of join key.",
                "report_date": "Explicit release date for fundamentals where available.",
                "retrieved_at": "When the value was ingested from the vendor.",
            },
            "mandatory_columns": PROVENANCE_COLUMNS,
            "leakage_sensitive_fields": LEAKAGE_SENSITIVE_FIELDS,
            "families": families,
            "files": output_files,
        }

    @staticmethod
    def _render_summary_html(config: RunConfig, report: ValidationReport) -> str:
        counts = report.counts()

        def esc(x: Any) -> str:
            return html.escape(str(x))

        rows = []
        for f in report.findings:
            colour = {"error": "#c0392b", "warning": "#d68910", "info": "#2471a3"}.get(
                f.severity.value, "#444"
            )
            rows.append(
                f"<tr><td style='color:{colour};font-weight:600'>{esc(f.severity.value)}</td>"
                f"<td>{esc(f.scope)}</td><td>{esc(f.code)}</td><td>{esc(f.message)}</td></tr>"
            )
        findings_rows = "\n".join(rows) or "<tr><td colspan='4'>No findings.</td></tr>"

        cov_rows = []
        for fam, cov in report.coverage.items():
            cov_rows.append(
                f"<tr><td>{esc(fam)}</td><td>{esc(cov.get('rows', 0))}</td>"
                f"<td>{esc(cov.get('entities', 0))}</td>"
                f"<td>{esc(cov.get('obs_start'))}</td><td>{esc(cov.get('obs_end'))}</td></tr>"
            )
        cov_table = "\n".join(cov_rows) or "<tr><td colspan='5'>No families included.</td></tr>"

        status = "BLOCKED (errors present)" if report.is_blocking else "OK to export"
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Validation summary -- {esc(config.run_name)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color:#222; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.5rem; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: .5rem; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: .9rem; }}
 th {{ background: #f4f6f7; }}
 .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; color:#fff; font-size:.8rem; }}
</style></head><body>
<h1>Validation summary &mdash; {esc(config.run_name)}</h1>
<p>Status: <strong>{esc(status)}</strong> &nbsp;|&nbsp;
 Errors: {counts['error']} &nbsp; Warnings: {counts['warning']} &nbsp; Info: {counts['info']}</p>
<p>Base currency: {esc(config.base_currency)} &nbsp;|&nbsp;
 Modelling frequency: {esc(config.modelling_frequency.value)} &nbsp;|&nbsp;
 Run id: {esc(config.run_id)}</p>
<h2>Coverage by family</h2>
<table><thead><tr><th>Family</th><th>Rows</th><th>Entities</th><th>Obs start</th><th>Obs end</th></tr></thead>
<tbody>{cov_table}</tbody></table>
<h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>Scope</th><th>Code</th><th>Message</th></tr></thead>
<tbody>{findings_rows}</tbody></table>
</body></html>"""
