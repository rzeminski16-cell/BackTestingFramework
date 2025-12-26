"""
Main BacktestAnalyzer class that orchestrates the analysis process.

This is the primary interface for running backtest analysis to generate:
1. Fundamental features CSVs (per security with period_GB_flag)
2. Technical features master CSV (all trades with indicators and flags)
3. Summary reports
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from .config import AnalysisConfig
from .indicators import WeeklyIndicatorCalculator
from .fundamental_features import FundamentalFeaturesGenerator
from .technical_features import TechnicalFeaturesGenerator


class BacktestAnalyzer:
    """
    Main orchestrator for backtest analysis.

    Generates three groups of output files:
    1. Fundamental Features: Per-security CSVs with monthly rows and yearly period_GB_flag
    2. Technical Features: Master CSV with filtered trades, indicators, and flags
    3. Summary Reports: Statistics and coverage reports

    Usage:
        config = AnalysisConfig(
            raw_data_directory=Path("raw_data"),
            output_directory=Path("analysis_output")
        )
        analyzer = BacktestAnalyzer(config)
        results = analyzer.analyze(Path("logs/StrategyName"))
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config

        # Validate required directories
        if config.raw_data_directory is None:
            raise ValueError("raw_data_directory must be specified in config")

        self.raw_data_dir = Path(config.raw_data_directory)
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")

        # Initialize components
        self.indicator_calculator = WeeklyIndicatorCalculator(config, self.raw_data_dir)
        self.fundamental_generator = FundamentalFeaturesGenerator(config)
        self.technical_generator = TechnicalFeaturesGenerator(config, self.indicator_calculator)

    def analyze(self, trade_logs_folder: Path,
                output_dir: Optional[Path] = None,
                initial_capital: float = 10000.0) -> Dict[str, Path]:
        """
        Run full analysis on a trade logs folder.

        Args:
            trade_logs_folder: Path to folder containing trade log CSVs
            output_dir: Output directory (overrides config if provided)
            initial_capital: Initial capital for metric calculations

        Returns:
            Dictionary with paths to generated files
        """
        trade_logs_folder = Path(trade_logs_folder)

        # Determine output directory
        if output_dir is not None:
            output_base = Path(output_dir)
        elif self.config.output_directory is not None:
            output_base = self.config.output_directory
        else:
            # Default: create analysis_output next to trade_logs_folder
            output_base = trade_logs_folder.parent / "analysis_output" / trade_logs_folder.name

        # Create output subdirectories
        fundamental_dir = output_base / "fundamental_features"
        technical_dir = output_base / "technical_features"
        weekly_dir = output_base / "weekly_data"
        summary_dir = output_base / "summaries"

        for dir_path in [fundamental_dir, technical_dir, weekly_dir, summary_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"=" * 60)
        print(f"Backtest Analysis")
        print(f"=" * 60)
        print(f"Trade logs folder: {trade_logs_folder}")
        print(f"Output directory:  {output_base}")
        print(f"=" * 60)

        results = {
            'output_directory': output_base,
            'fundamental_features': {},
            'technical_features': None,
            'summaries': {},
        }

        # Save configuration
        config_file = output_base / "analysis_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        results['config_file'] = config_file
        print(f"\nSaved configuration to {config_file}")

        # Generate fundamental features
        print(f"\n{'=' * 60}")
        print("Generating Fundamental Features...")
        print(f"{'=' * 60}")

        fundamental_results = self.fundamental_generator.process_trade_logs_folder(
            trade_logs_folder, fundamental_dir, initial_capital
        )
        results['fundamental_features'] = fundamental_results

        print(f"\nGenerated fundamental features for {len(fundamental_results)} securities")

        # Generate technical features
        print(f"\n{'=' * 60}")
        print("Generating Technical Features...")
        print(f"{'=' * 60}")

        master_df = self.technical_generator.generate_technical_features(
            trade_logs_folder,
            save_weekly=self.config.save_weekly_data,
            weekly_output_dir=weekly_dir if self.config.save_weekly_data else None
        )

        if not master_df.empty:
            # Save master file
            master_file = technical_dir / "technical_features_master.csv"
            master_df.to_csv(master_file, index=False)
            results['technical_features'] = master_file
            print(f"\nSaved technical features master to {master_file}")
            print(f"Total trades in master file: {len(master_df)}")

            # Generate and save summary statistics
            summary_stats = self.technical_generator.generate_summary_stats(master_df)
            if not summary_stats.empty:
                summary_file = summary_dir / "technical_features_summary.csv"
                summary_stats.to_csv(summary_file, index=False)
                results['summaries']['technical_summary'] = summary_file
                print(f"Saved technical summary to {summary_file}")

        # Generate filtered trade counts report
        print(f"\n{'=' * 60}")
        print("Generating Trade Counts Report...")
        print(f"{'=' * 60}")

        trade_counts = self.technical_generator.generate_filtered_trade_counts(trade_logs_folder)
        if not trade_counts.empty:
            counts_file = summary_dir / "trade_counts_by_year.csv"
            trade_counts.to_csv(counts_file, index=False)
            results['summaries']['trade_counts'] = counts_file
            print(f"Saved trade counts to {counts_file}")

            # Print summary
            total_before = trade_counts['trade_count'].sum()
            included = trade_counts[trade_counts['included']]
            total_after = included['trade_count'].sum()
            excluded = total_before - total_after

            print(f"\nTrade Count Summary:")
            print(f"  Total trades:    {total_before}")
            print(f"  Included trades: {total_after}")
            print(f"  Excluded trades: {excluded}")
            print(f"  Inclusion rate:  {total_after / total_before * 100:.1f}%")

        # Generate overall summary report
        print(f"\n{'=' * 60}")
        print("Generating Overall Summary...")
        print(f"{'=' * 60}")

        overall_summary = self._generate_overall_summary(
            fundamental_results, master_df if not master_df.empty else None
        )
        if not overall_summary.empty:
            overall_file = summary_dir / "overall_summary.csv"
            overall_summary.to_csv(overall_file, index=False)
            results['summaries']['overall'] = overall_file
            print(f"Saved overall summary to {overall_file}")

        print(f"\n{'=' * 60}")
        print("Analysis Complete!")
        print(f"{'=' * 60}")
        print(f"\nOutput saved to: {output_base}")

        return results

    def _generate_overall_summary(self, fundamental_results: Dict[str, Path],
                                   master_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Generate an overall summary combining fundamental and technical analysis.

        Args:
            fundamental_results: Dictionary of fundamental feature file paths
            master_df: Master technical features DataFrame

        Returns:
            Summary DataFrame
        """
        summary_rows = []

        # Summary info
        summary_rows.append({
            'metric': 'Analysis Date',
            'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

        summary_rows.append({
            'metric': 'Securities with Fundamental Features',
            'value': len(fundamental_results),
        })

        if master_df is not None and not master_df.empty:
            summary_rows.append({
                'metric': 'Total Trades in Technical Master',
                'value': len(master_df),
            })

            summary_rows.append({
                'metric': 'Unique Securities in Technical Master',
                'value': master_df['symbol'].nunique(),
            })

            # GB_Flag distribution
            gb_counts = master_df['GB_Flag'].value_counts()
            for flag, count in gb_counts.items():
                summary_rows.append({
                    'metric': f'GB_Flag {flag} Count',
                    'value': count,
                })
                summary_rows.append({
                    'metric': f'GB_Flag {flag} Percentage',
                    'value': f"{count / len(master_df) * 100:.1f}%",
                })

            # Outcomes_Flag distribution
            outcome_counts = master_df['Outcomes_Flag'].value_counts()
            for flag, count in outcome_counts.items():
                summary_rows.append({
                    'metric': f'Outcomes_Flag {flag} Count',
                    'value': count,
                })
                summary_rows.append({
                    'metric': f'Outcomes_Flag {flag} Percentage',
                    'value': f"{count / len(master_df) * 100:.1f}%",
                })

            # Year range
            if 'entry_year' in master_df.columns:
                summary_rows.append({
                    'metric': 'Year Range',
                    'value': f"{master_df['entry_year'].min()} - {master_df['entry_year'].max()}",
                })

        return pd.DataFrame(summary_rows)

    def analyze_single_security(self, trade_log_file: Path,
                                 symbol: str,
                                 output_dir: Path,
                                 initial_capital: float = 10000.0) -> Dict[str, Path]:
        """
        Analyze a single security's trade log.

        Args:
            trade_log_file: Path to the trade log CSV
            symbol: Security symbol
            output_dir: Output directory
            initial_capital: Initial capital for calculations

        Returns:
            Dictionary with paths to generated files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Generate fundamental features
        fundamental_df = self.fundamental_generator.process_trade_log_file(
            trade_log_file, symbol, initial_capital
        )
        if fundamental_df is not None and not fundamental_df.empty:
            fundamental_file = output_dir / f"{symbol}_fundamental_features.csv"
            fundamental_df.to_csv(fundamental_file, index=False)
            results['fundamental_features'] = fundamental_file

        # Generate weekly indicators
        weekly_df = self.indicator_calculator.calculate_all_indicators(symbol)
        if weekly_df is not None:
            weekly_file = output_dir / f"{symbol}_weekly_indicators.csv"
            weekly_df.to_csv(weekly_file, index=False)
            results['weekly_indicators'] = weekly_file

        # Load and process trades for technical features
        trades = self.technical_generator.load_trade_log(trade_log_file)
        if trades is not None and not trades.empty:
            trades['symbol'] = symbol
            trades['entry_year'] = trades['entry_date'].dt.year

            # Process each trade
            processed = []
            for _, trade in trades.iterrows():
                processed.append(self.technical_generator.process_single_trade(trade, symbol))

            technical_df = pd.DataFrame(processed)
            technical_file = output_dir / f"{symbol}_technical_features.csv"
            technical_df.to_csv(technical_file, index=False)
            results['technical_features'] = technical_file

        return results

    def clear_cache(self):
        """Clear all cached data from indicator calculator."""
        self.indicator_calculator.clear_cache()


def run_analysis(trade_logs_folder: str,
                 raw_data_directory: str,
                 output_directory: Optional[str] = None,
                 initial_capital: float = 10000.0,
                 **config_overrides) -> Dict[str, Path]:
    """
    Convenience function to run analysis with minimal setup.

    Args:
        trade_logs_folder: Path to folder containing trade log CSVs
        raw_data_directory: Path to raw price data directory
        output_directory: Optional output directory
        initial_capital: Initial capital for calculations
        **config_overrides: Additional config parameters to override

    Returns:
        Dictionary with paths to generated files

    Example:
        results = run_analysis(
            trade_logs_folder="logs/AlphaTrendStrategy",
            raw_data_directory="raw_data",
            output_directory="analysis_output/AlphaTrendStrategy",
            gb_profit_good_threshold=7.5,  # Override default 5%
        )
    """
    config = AnalysisConfig(
        raw_data_directory=Path(raw_data_directory),
        output_directory=Path(output_directory) if output_directory else None,
        **config_overrides
    )

    analyzer = BacktestAnalyzer(config)
    return analyzer.analyze(
        Path(trade_logs_folder),
        initial_capital=initial_capital
    )
