"""
Self-contained HTML report for factor analysis.

Renders an AnalysisOutput into a single standalone .html file (inline CSS, no
external assets or network) - a lightweight, shareable alternative to the GUI:
data summary, key findings, factor correlations (color-scaled), hypothesis-test
results, and detected scenarios.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1f2933}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px;color:#243b53;border-bottom:2px solid #d9e2ec;padding-bottom:4px}
.sub{color:#627d98;font-size:13px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;box-shadow:0 1px 2px rgba(0,0,0,.06);margin-bottom:8px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid #e4e7eb}
th{background:#243b53;color:#fff;font-weight:600;position:sticky;top:0}
tr:nth-child(even) td{background:#fafbfc}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.card{background:#fff;border-radius:6px;padding:10px 12px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.card .k{color:#627d98;font-size:11px;text-transform:uppercase} .card .v{font-size:18px;font-weight:600}
.finding{background:#fff;border-left:4px solid #2bb0ed;padding:8px 12px;margin:6px 0;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.warn{border-left-color:#f0b429}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.mono{font-variant-numeric:tabular-nums}
"""


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _corr_color(r: float) -> str:
    # red (negative) -> white (0) -> green (positive), clamped at |r|=0.5
    try:
        x = max(-1.0, min(1.0, float(r) / 0.5))
    except (TypeError, ValueError):
        return ""
    if x >= 0:
        g = 255; rr = int(255 * (1 - x)); b = int(255 * (1 - x))
    else:
        rr = 255; g = int(255 * (1 + x)); b = int(255 * (1 + x))
    return f"background:rgb({rr},{g},{b})"


def _cards(d: dict) -> str:
    items = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            continue
        items.append(f'<div class="card"><div class="k">{_esc(k)}</div><div class="v mono">{_esc(v)}</div></div>')
    return f'<div class="kv">{"".join(items)}</div>' if items else ""


def _records_table(records, columns=None) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return "<p class='sub'>No data.</p>"
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            style = ""
            if isinstance(val, float):
                if any(t in str(c).lower() for t in ("corr", "_r", "pearson", "spearman")) and -1 <= val <= 1:
                    style = _corr_color(val)
                val = f"{val:.4f}"
            cells.append(f'<td class="mono" style="{style}">{_esc(val)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _correlations_section(tier1: dict) -> str:
    if not isinstance(tier1, dict):
        return ""
    parts = []
    for key in ("correlations_pearson", "correlations_spearman"):
        block = tier1.get(key)
        if not block:
            continue
        # block may be {factor: {correlation/r, p_value}} or a list of dicts
        if isinstance(block, dict):
            records = []
            for factor, stats in block.items():
                if isinstance(stats, dict):
                    rec = {"factor": factor}
                    rec.update(stats)
                    records.append(rec)
                else:
                    records.append({"factor": factor, "value": stats})
        else:
            records = block
        df = pd.DataFrame(records)
        if df.empty:
            continue
        # sort by absolute correlation if a correlation-like column exists
        corr_col = next((c for c in df.columns if any(t in c.lower() for t in ("corr", "_r", "value"))), None)
        if corr_col:
            df = df.reindex(df[corr_col].abs().sort_values(ascending=False).index)
        parts.append(f"<h2>{key.replace('_', ' ').title()}</h2>" + _records_table(df.head(30).to_dict("records")))
    return "".join(parts)


def _generic_section(title: str, obj: Any) -> str:
    if obj is None or (hasattr(obj, "__len__") and len(obj) == 0):
        return ""
    html_out = f"<h2>{_esc(title)}</h2>"
    if isinstance(obj, dict):
        scalar = {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}
        if scalar:
            html_out += _cards(scalar)
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                html_out += f"<h3 class='sub'>{_esc(k)}</h3>" + _records_table(v[:30])
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        html_out += _records_table(obj[:30])
    return html_out


def generate_html_report(result: Any, output_path) -> Path:
    """Render an AnalysisOutput to a standalone HTML file. Returns the path."""
    output_path = Path(output_path)
    ts = getattr(result, "timestamp", datetime.now().isoformat())

    sections = [f'<h1>Factor Analysis Report</h1><div class="sub">Generated {_esc(ts)}</div>']

    data_summary = getattr(result, "data_summary", None) or {}
    if data_summary:
        sections.append("<h2>Data Summary</h2>" + _cards(data_summary))

    findings = getattr(result, "key_findings", None) or []
    if findings:
        sections.append("<h2>Key Findings</h2>" +
                        "".join(f'<div class="finding">{_esc(f)}</div>' for f in findings))

    warns = getattr(result, "warnings", None) or []
    if warns:
        sections.append("<h2>Warnings</h2>" +
                        "".join(f'<div class="finding warn">{_esc(w)}</div>' for w in warns))

    sections.append(_correlations_section(getattr(result, "tier1", None) or {}))
    sections.append(_generic_section("Hypothesis Tests (Tier 2)", getattr(result, "tier2", None)))
    sections.append(_generic_section("Scenarios", getattr(result, "scenarios", None)))

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Factor Analysis Report</title><style>{_CSS}</style></head>"
           f"<body><div class='wrap'>{''.join(sections)}</div></body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
