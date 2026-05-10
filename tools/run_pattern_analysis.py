"""
CLI for the entry-pattern analyzer.

For each trade in a trade log, looks at the lookback windows before entry and
detects MA-crossover buy/sell signals (same logic as the AlphaTrend strategy:
``MA(L)`` shifted by ``offset`` crossing the unshifted ``MA(L)``). Reports the
density and pattern of these signals so you can study how pre-entry signal
behaviour relates to trade outcome.

Examples
--------

Default combos / windows::

    python tools/run_pattern_analysis.py --trade-log logs/foo/trades.csv

Pick a few combos via ``--ma`` (TYPE:LENGTH:OFFSET)::

    python tools/run_pattern_analysis.py \\
        --trade-log logs/foo/trades.csv \\
        --ma EMA:14:5 --ma EMA:20:5 --ma SMA:50:10 \\
        --windows 30 60 90 120

Use a YAML/JSON config of combos (and optionally windows)::

    python tools/run_pattern_analysis.py \\
        --trade-log logs/foo/trades.csv \\
        --config tools/example_pattern_combos.yaml

CLI flags take precedence over the config file when both are given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python tools/run_pattern_analysis.py ...`` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Classes.Analysis.pattern_analysis_report import write_report
from Classes.Analysis.pattern_analyzer import (
    PatternAnalyzer,
    default_combos,
    default_windows,
    load_combos_from_config,
    load_trade_log,
    parse_combo_string,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze MA-crossover signal patterns prior to each trade entry.",
    )
    parser.add_argument(
        "--trade-log",
        required=True,
        help="Path to the trade log CSV.",
    )
    parser.add_argument(
        "--data-path",
        default="raw_data/daily",
        help="Directory containing {SYMBOL}_daily.csv files (default: raw_data/daily).",
    )
    parser.add_argument(
        "--ma",
        action="append",
        default=[],
        metavar="TYPE:LENGTH:OFFSET",
        help="MA combo to analyze. Repeatable. e.g. --ma EMA:20:5 --ma SMA:14:10. "
             "TYPE in {SMA, EMA}, LENGTH in {7, 14, 20, 30, 50}, OFFSET >= 1.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON file with 'combos' (list) and optional 'windows' (list).",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=None,
        help="Lookback windows in days (default: 30 60 90 120).",
    )
    parser.add_argument(
        "--output",
        default="pattern_analysis_report.xlsx",
        help="Output Excel file (default: pattern_analysis_report.xlsx).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trade progress output.",
    )

    args = parser.parse_args()

    combos = []
    windows = None

    if args.config:
        combos, windows = load_combos_from_config(Path(args.config))

    if args.ma:
        cli_combos = [parse_combo_string(s) for s in args.ma]
        # CLI takes precedence over config.
        combos = cli_combos

    if not combos:
        combos = default_combos()
        if not args.quiet:
            print("No combos specified; using default starter set:")
            for c in combos:
                print(f"  - {c.label}")

    if args.windows:
        windows = args.windows
    if not windows:
        windows = default_windows()

    trade_log_path = Path(args.trade_log)
    if not trade_log_path.exists():
        print(f"Trade log not found: {trade_log_path}", file=sys.stderr)
        return 2

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Price data path not found: {data_path}", file=sys.stderr)
        return 2

    print(f"Loading trades from {trade_log_path}...")
    trades = load_trade_log(trade_log_path)
    print(f"  Loaded {len(trades)} trade(s)")

    analyzer = PatternAnalyzer(data_path=data_path, combos=combos, windows=windows)

    def progress(done: int, total: int, message: str) -> None:
        if args.quiet:
            return
        print(f"  [{done}/{total}] {message}")

    print(
        f"Analyzing {len(trades)} trade(s) across {len(combos)} combo(s) "
        f"and windows {windows}..."
    )
    result = analyzer.analyze(trades, progress=progress)

    output_path = Path(args.output)
    print(f"Writing Excel report to {output_path}...")
    write_report(result, output_path, trade_log_path=trade_log_path)
    print("Done.")
    print(f"  Per-trade rows: {len(result.features)}")
    print(f"  Raw signal rows: {len(result.raw_signals)}")
    print(f"  Skipped trades: {len(result.skipped_trades)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
