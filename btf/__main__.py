"""
btf — unified command-line interface for the BackTestingFramework.

    python -m btf backtest   --strategy AlphaTrendV1Strategy --symbols AAPL
    python -m btf backtest   --strategy AlphaTrendV1Strategy --basket Technology
    python -m btf optimize   --strategy AlphaTrendV1Strategy --symbol AAPL
    python -m btf montecarlo --trade-log logs/run/trades.csv
    python -m btf list       all
    python -m btf dashboard  [--run processed_data/runs/<run>/<model_run>]

Every command is headless and exits non-zero on failure, so the CLI can sit
inside shell scripts, cron jobs, or CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow `python btf/__main__.py` and installed-package execution alike.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btf import __version__  # noqa: E402
from btf import commands  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btf",
        description="BackTestingFramework CLI — headless backtests, "
                    "optimization, Monte Carlo, and dashboard launching.",
    )
    parser.add_argument("--version", action="version",
                        version=f"btf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- list ------------------------------------------------------------ #
    p_list = sub.add_parser("list", help="List strategies, securities, "
                                         "baskets, or benchmarks")
    p_list.add_argument("what", nargs="?", default="all",
                        choices=["strategies", "securities", "baskets",
                                 "benchmarks", "all"])
    p_list.add_argument("--data-dir", default=str(commands.DEFAULT_DATA_DIR),
                        help="Securities data directory (default: raw_data/daily)")
    p_list.set_defaults(func=commands.cmd_list)

    # ---- backtest --------------------------------------------------------- #
    p_bt = sub.add_parser("backtest", help="Run a single-security or "
                                           "portfolio backtest")
    p_bt.add_argument("--strategy", required=True,
                      help="Strategy class name (see: btf list strategies)")
    p_bt.add_argument("--symbols", nargs="+",
                      help="One symbol = single-security engine; several = "
                           "portfolio engine")
    p_bt.add_argument("--basket", help="Basket name (portfolio mode)")
    p_bt.add_argument("--param", action="append", metavar="NAME=VALUE",
                      help="Strategy parameter override (repeatable)")
    p_bt.add_argument("--capital", type=float, default=100_000.0)
    p_bt.add_argument("--commission", type=float, default=0.001,
                      help="Commission (fraction for percentage mode, "
                           "currency for --fixed-commission)")
    p_bt.add_argument("--fixed-commission", action="store_true",
                      help="Interpret --commission as a fixed amount per trade")
    p_bt.add_argument("--slippage", type=float, default=0.1,
                      help="Slippage percent per fill (default 0.1)")
    p_bt.add_argument("--start", help="Backtest start date (YYYY-MM-DD)")
    p_bt.add_argument("--end", help="Backtest end date (YYYY-MM-DD)")
    p_bt.add_argument("--base-currency", default="GBP")
    p_bt.add_argument("--no-fx", action="store_true",
                      help="Disable FX conversion / currency validation "
                           "(single-currency data, e.g. external CSVs that "
                           "are not in the security registry)")
    p_bt.add_argument("--data-dir", default=str(commands.DEFAULT_DATA_DIR))
    p_bt.add_argument("--intrabar-stops", action="store_true",
                      help="Trigger stops/TP on bar high/low with gap fills")
    p_bt.add_argument("--next-bar-open", action="store_true",
                      help="Fill signals at the next bar's open "
                           "(single-security engine only)")
    p_bt.add_argument("--trades-csv", help="Write the trade log to this CSV")
    p_bt.add_argument("--json", help="Write the metrics dict to this JSON file")
    p_bt.add_argument("--report", action="store_true",
                      help="Generate the full Excel report")
    p_bt.add_argument("--report-dir", default="reports/cli",
                      help="Directory for --report output (default reports/cli)")
    p_bt.add_argument("--interactive", action="store_true",
                      help="Interactive (discretionary) mode is GUI-only for "
                           "now; use ctk_backtest_gui.py")
    p_bt.set_defaults(func=commands.cmd_backtest)

    # ---- optimize ---------------------------------------------------------- #
    p_opt = sub.add_parser("optimize", help="Walk-forward optimization "
                                            "(single security)")
    p_opt.add_argument("--strategy", required=True)
    p_opt.add_argument("--symbol", required=True)
    p_opt.add_argument("--mode", choices=["rolling", "anchored"],
                       help="Walk-forward mode (default: config file setting)")
    p_opt.add_argument("--optimize-params", nargs="+", metavar="PARAM",
                       help="Only optimize these parameters (default: all)")
    p_opt.add_argument("--config", default="config/optimization_config.yaml")
    p_opt.add_argument("--data-dir", default=str(commands.DEFAULT_DATA_DIR))
    p_opt.add_argument("--report", action="store_true",
                       help="Generate the Excel optimization report")
    p_opt.add_argument("--report-dir", default="reports/cli")
    p_opt.set_defaults(func=commands.cmd_optimize)

    # ---- montecarlo -------------------------------------------------------- #
    p_mc = sub.add_parser("montecarlo", help="Bootstrap Monte Carlo from a "
                                             "trade log or an equity curve")
    p_mc.add_argument("--trade-log", nargs="+",
                      help="Trade-log CSV file(s) (framework schema)")
    p_mc.add_argument("--daily-curve",
                      help="Equity-curve/price CSV — bootstrap DAILY returns "
                           "instead of per-trade returns")
    p_mc.add_argument("--source", choices=["pct", "r"], default="pct",
                      help="Trade-log return source: pct returns or R-multiples")
    p_mc.add_argument("--simulations", type=int, default=5000)
    p_mc.add_argument("--steps", type=int, default=200,
                      help="Steps per path (trades, or days with --daily-curve)")
    p_mc.add_argument("--capital", type=float, default=100_000.0)
    p_mc.add_argument("--risk", type=float, default=0.01,
                      help="Risk per trade as an equity fraction "
                           "(ignored with --daily-curve, which uses 1.0)")
    p_mc.add_argument("--block-size", type=int,
                      help="Use block bootstrap with this block length")
    p_mc.add_argument("--periods-per-year", type=float,
                      help="Steps per year for annualized distributions "
                           "(default: 252 with --daily-curve, else unset)")
    p_mc.add_argument("--seed", type=int, default=None)
    p_mc.add_argument("--json", help="Write metrics JSON to this file")
    p_mc.set_defaults(func=commands.cmd_montecarlo)

    # ---- ingest ------------------------------------------------------------ #
    p_ing = sub.add_parser("ingest", help="Convert raw CSVs to typed, "
                                          "validated Parquet (faster loads)")
    p_ing.add_argument("--data-dir", default=str(commands.DEFAULT_DATA_DIR))
    p_ing.add_argument("--force", action="store_true",
                       help="Re-write Parquet even when up to date")
    p_ing.set_defaults(func=commands.cmd_ingest)

    # ---- signals ------------------------------------------------------------ #
    p_sig = sub.add_parser("signals", help="Live-paper bridge: the strategy's "
                                           "action on the latest collected bar "
                                           "(ENTER/EXIT/HOLDING/FLAT)")
    p_sig.add_argument("--strategy", required=True)
    p_sig.add_argument("--symbols", nargs="+")
    p_sig.add_argument("--basket")
    p_sig.add_argument("--param", action="append", metavar="NAME=VALUE")
    p_sig.add_argument("--capital", type=float, default=100_000.0)
    p_sig.add_argument("--base-currency", default="GBP")
    p_sig.add_argument("--no-fx", action="store_true")
    p_sig.add_argument("--data-dir", default=str(commands.DEFAULT_DATA_DIR))
    p_sig.add_argument("--json", help="Write per-symbol signals JSON (for cron)")
    p_sig.set_defaults(func=commands.cmd_signals)

    # ---- new-strategy ------------------------------------------------------ #
    p_new = sub.add_parser("new-strategy", help="Scaffold a new strategy file "
                                                "(auto-discovered, no registry "
                                                "edit needed)")
    p_new.add_argument("name", help="Strategy class name, e.g. "
                                    "MeanReversionStrategy")
    p_new.add_argument("--direction", choices=["long", "short"], default="long")
    p_new.add_argument("--indicators", nargs="+", metavar="COLUMN",
                       help="Indicator columns the strategy needs from raw "
                            "data (default: atr_14)")
    p_new.add_argument("--output-dir", default="strategies")
    p_new.add_argument("--force", action="store_true",
                       help="Overwrite an existing file")
    p_new.set_defaults(func=commands.cmd_new_strategy)

    # ---- dashboard --------------------------------------------------------- #
    p_dash = sub.add_parser("dashboard", help="Launch the Streamlit results "
                                              "dashboard")
    p_dash.add_argument("--run", help="Open this exported model-run directory "
                                      "directly (sets MODEL_RUN_DIR)")
    p_dash.set_defaults(func=commands.cmd_dashboard)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
