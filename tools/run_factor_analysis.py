#!/usr/bin/env python3
"""
Headless factor-analysis runner.

Builds the point-in-time fundamental aggregate from the rebuilt per-symbol
panels (raw_data/fundamentals/{SYMBOL}_fundamental.csv -> derived ratios), runs
the FactorAnalyzer against your trade logs with the full fundamental factor set
enabled, and writes Excel + JSON + a standalone HTML report. No GUI required.

Examples:
    python tools/run_factor_analysis.py logs/MyStrategy
    python tools/run_factor_analysis.py logs/MyStrategy --output-dir reports/fa --eps-only
    python tools/run_factor_analysis.py path/to/one_trades.csv
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Classes.FactorAnalysis.analyzer import FactorAnalyzer
from Classes.FactorAnalysis.config.factor_config import FactorAnalysisConfig
from Classes.FactorAnalysis.data.fundamental_panel import build_aggregate
from Classes.FactorAnalysis.data import insider_panel
from Classes.FactorAnalysis.output.html_generator import generate_html_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trade_logs", help="Folder of *_trades.csv (or a single trades CSV)")
    parser.add_argument("--raw-data", default="raw_data", help="Raw data root (default: raw_data)")
    parser.add_argument("--output-dir", default=None, help="Where to write reports")
    parser.add_argument("--pattern", default="*_trades.csv", help="Trade-log glob for folder mode")
    parser.add_argument("--price-data", default=None, help="Optional price CSV for technical factors")
    parser.add_argument("--eps-only", action="store_true",
                        help="Restrict fundamentals to EPS factors (default: full value/quality/growth)")
    parser.add_argument("--no-fundamentals", action="store_true", help="Skip fundamental factors")
    parser.add_argument("--no-insider", action="store_true", help="Skip insider factors")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw_data)
    fdir = raw / "fundamentals"
    trade_path = Path(args.trade_logs)
    out = Path(args.output_dir) if args.output_dir else (
        (trade_path if trade_path.is_dir() else trade_path.parent) / "factor_analysis_output")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Build the point-in-time fundamental aggregate from the new panels.
    fundamental_df = None
    if not args.no_fundamentals:
        fundamental_df = build_aggregate(fdir, out_path=fdir / "fundamental_data.csv")
        if fundamental_df is not None and not fundamental_df.empty:
            n_sym = fundamental_df["symbol"].nunique() if "symbol" in fundamental_df.columns else 0
            print(f"Fundamental aggregate: {len(fundamental_df)} rows, {n_sym} symbols "
                  f"-> {fdir / 'fundamental_data.csv'}")
        else:
            print("No fundamental panels found; proceeding without fundamental factors.")
            fundamental_df = None

    # 1b. Insider: clean raw transactions for the analyzer, and write the
    # point-in-time aggregate panel (raw_data/insider_transactions/insider_data.csv).
    insider_df = None
    if not args.no_insider:
        idir = raw / "insider_transactions"
        insider_df = insider_panel.load_transactions(idir)
        if insider_df is not None and not insider_df.empty:
            panel = insider_panel.build_aggregate(idir, out_path=idir / "insider_data.csv")
            n_sym = insider_df["symbol"].nunique() if "symbol" in insider_df.columns else 0
            print(f"Insider: {len(insider_df)} cleaned transactions, {n_sym} symbols "
                  f"-> aggregate panel {len(panel)} rows ({idir / 'insider_data.csv'})")
        else:
            insider_df = None

    # 2. Configure: enable the full fundamental factor set unless --eps-only.
    # Config dataclasses are frozen, so rebuild with dataclasses.replace.
    config = FactorAnalysisConfig()
    fe = config.factor_engineering
    replacements = {}
    if not args.eps_only:
        replacements.update(
            eps_only_fundamentals=False,
            quality=dataclasses.replace(fe.quality, enabled=True),
            growth=dataclasses.replace(fe.growth, enabled=True),
            # Value factors need price-derived ratios (P/E, P/B) not in the panel -> left off.
        )
    if insider_df is not None:
        replacements["insider"] = dataclasses.replace(fe.insider, enabled=True)
    if replacements:
        config = dataclasses.replace(config, factor_engineering=dataclasses.replace(fe, **replacements))

    analyzer = FactorAnalyzer(config=config, verbose=not args.quiet)

    # 3. Load trade logs (reuses the loader's normalization).
    if trade_path.is_dir():
        trades_df, _, _ = analyzer.trade_loader.load_from_directory(trade_path, pattern=args.pattern)
    else:
        loaded = analyzer.trade_loader.load_single(trade_path)
        trades_df = loaded[0] if isinstance(loaded, tuple) else loaded
    if trades_df is None or trades_df.empty:
        print(f"No trades loaded from {trade_path}")
        return 1
    print(f"Loaded {len(trades_df)} trades.")

    # 4. Run the analysis.
    result = analyzer.analyze(trades_df, price_data=args.price_data,
                              fundamental_data=fundamental_df, insider_data=insider_df)
    if not result.success:
        print(f"Analysis failed: {result.error}")
        return 1

    # 5. Write reports. Each format is independent so one failure can't block the rest.
    written = []
    for label, fn in (
        ("factor_analysis.html", lambda: generate_html_report(result, out / "factor_analysis.html")),
        ("factor_analysis.json", lambda: analyzer.generate_json_payload(result, str(out / "factor_analysis.json"))),
        ("factor_analysis.xlsx", lambda: analyzer.generate_excel_report(result, str(out / "factor_analysis.xlsx"))),
    ):
        try:
            fn()
            written.append(label)
        except Exception as exc:
            print(f"  [warn] {label} generation failed: {exc}")

    if not args.quiet:
        try:
            analyzer.print_summary(result)
        except Exception:
            pass
    print(f"\nReports written to {out}/: {', '.join(written) if written else 'none'}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
