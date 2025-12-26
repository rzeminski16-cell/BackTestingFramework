#!/usr/bin/env python3
"""
CLI script for running backtest analysis.

This script provides a command-line interface for the BacktestAnalyzer module.

Usage:
    python run_backtest_analysis.py <trade_logs_folder> [options]

Examples:
    # Basic usage with defaults
    python run_backtest_analysis.py logs/AlphaTrendStrategy_Fundamental_Constraints_Base

    # With custom output directory
    python run_backtest_analysis.py logs/AlphaTrendStrategy_Fundamental_Constraints_Base -o analysis_output/my_analysis

    # With custom thresholds
    python run_backtest_analysis.py logs/AlphaTrendStrategy_Fundamental_Constraints_Base --gb-threshold 7.5 --min-trades 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Classes.Analysis.backtest_analyzer import BacktestAnalyzer, AnalysisConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze backtest trade logs to generate fundamental and technical features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python run_backtest_analysis.py logs/AlphaTrendStrategy_Fundamental_Constraints_Base

    # With custom output directory
    python run_backtest_analysis.py logs/MyStrategy -o analysis_output/MyAnalysis

    # With custom thresholds
    python run_backtest_analysis.py logs/MyStrategy --gb-threshold 7.5 --calmar-threshold 0.75

Output Structure:
    output_directory/
    ├── fundamental_features/     # Per-security CSVs with period_GB_flag
    │   └── {SYMBOL}_fundamental_features.csv
    ├── technical_features/       # Master file with all trades and indicators
    │   └── technical_features_master.csv
    ├── weekly_data/              # Weekly indicator data for validation
    │   └── {SYMBOL}_weekly_indicators.csv
    ├── summaries/                # Summary reports
    │   ├── overall_summary.csv
    │   ├── technical_features_summary.csv
    │   └── trade_counts_by_year.csv
    └── analysis_config.json      # Configuration used for this analysis
        """
    )

    # Required arguments
    parser.add_argument(
        "trade_logs_folder",
        type=str,
        help="Path to folder containing trade log CSV files"
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: analysis_output/<strategy_name>)"
    )

    parser.add_argument(
        "-r", "--raw-data",
        type=str,
        default="raw_data",
        help="Path to raw price data directory (default: raw_data)"
    )

    parser.add_argument(
        "-c", "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for calculations (default: 10000)"
    )

    # Threshold customization
    parser.add_argument(
        "--gb-threshold",
        type=float,
        default=5.0,
        help="Profit threshold for GB_Flag 'Good' classification (default: 5.0%%)"
    )

    parser.add_argument(
        "--calmar-threshold",
        type=float,
        default=0.5,
        help="Calmar ratio threshold for period 'good' classification (default: 0.5)"
    )

    parser.add_argument(
        "--max-dd-threshold",
        type=float,
        default=25.0,
        help="Max drawdown threshold for period 'good' classification (default: 25.0%%)"
    )

    parser.add_argument(
        "--min-trades",
        type=int,
        default=4,
        help="Minimum trades per security per year for inclusion (default: 4)"
    )

    parser.add_argument(
        "--days-after",
        type=int,
        default=21,
        help="Days after exit to check for Outcomes_Flag (default: 21)"
    )

    # Outcomes flag thresholds
    parser.add_argument(
        "--full-ride-profit",
        type=float,
        default=5.0,
        help="Minimum profit for FullRideGood (default: 5.0%%)"
    )

    parser.add_argument(
        "--full-ride-extension",
        type=float,
        default=10.0,
        help="Price extension tolerance for FullRideGood as %% of entry (default: 10.0%%)"
    )

    parser.add_argument(
        "--early-exit-extension",
        type=float,
        default=5.0,
        help="Major extension threshold for EarlyExitGood (default: 5.0%%)"
    )

    parser.add_argument(
        "--missed-opp-surge",
        type=float,
        default=10.0,
        help="Price surge threshold for MissedOpportunity (default: 10.0%%)"
    )

    # Options
    parser.add_argument(
        "--no-weekly",
        action="store_true",
        help="Skip saving weekly indicator data"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate paths
    trade_logs_folder = Path(args.trade_logs_folder)
    if not trade_logs_folder.exists():
        print(f"Error: Trade logs folder not found: {trade_logs_folder}")
        sys.exit(1)

    raw_data_dir = Path(args.raw_data)
    if not raw_data_dir.exists():
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        sys.exit(1)

    # Create configuration
    config = AnalysisConfig(
        raw_data_directory=raw_data_dir,
        output_directory=Path(args.output) if args.output else None,
        save_weekly_data=not args.no_weekly,

        # GB_Flag thresholds
        gb_profit_good_threshold=args.gb_threshold,

        # Period classification thresholds
        calmar_good_threshold=args.calmar_threshold,
        max_dd_good_threshold=args.max_dd_threshold,

        # Trade filtering
        min_trades_per_year=args.min_trades,

        # Outcomes flag thresholds
        full_ride_profit_min=args.full_ride_profit,
        full_ride_price_extension=args.full_ride_extension,
        early_exit_major_extension=args.early_exit_extension,
        missed_opportunity_surge=args.missed_opp_surge,
        days_after_exit=args.days_after,
    )

    if args.verbose:
        print("\nConfiguration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        print()

    # Run analysis
    try:
        analyzer = BacktestAnalyzer(config)
        results = analyzer.analyze(
            trade_logs_folder,
            initial_capital=args.capital
        )

        print("\n" + "=" * 60)
        print("Generated Files:")
        print("=" * 60)

        if results.get('config_file'):
            print(f"  Config: {results['config_file']}")

        if results.get('fundamental_features'):
            print(f"  Fundamental Features: {len(results['fundamental_features'])} files")

        if results.get('technical_features'):
            print(f"  Technical Features: {results['technical_features']}")

        if results.get('summaries'):
            for name, path in results['summaries'].items():
                print(f"  Summary ({name}): {path}")

        print("\nAnalysis complete!")
        return 0

    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
