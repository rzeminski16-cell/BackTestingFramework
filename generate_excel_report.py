"""
Standalone script to generate Excel reports from backtest results.

This script demonstrates how to generate comprehensive Excel reports
for your backtesting results. The report includes:
- Summary Dashboard with all key metrics
- Detailed Trade Log
- Performance Analysis (drawdown, monthly returns, distributions)
- Visualizations (charts and graphs)
- Market Condition Breakdown

Usage:
    python generate_excel_report.py
"""

from pathlib import Path
from datetime import datetime

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from Classes.Analysis.performance_metrics import PerformanceMetrics

from strategies.examples import PartialExitStrategy


def main():
    """Generate Excel report for a sample backtest."""

    print("\n" + "="*80)
    print("EXCEL REPORT GENERATOR")
    print("="*80 + "\n")

    # ========== CONFIGURATION ==========

    # Backtest configuration
    initial_capital = 100000.0
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
    )

    # Security to backtest
    symbol = 'AAPL'

    # Strategy configuration
    strategy = PartialExitStrategy(
        position_size=0.2,
        first_target_pct=0.10,
        second_target_pct=0.20,
        stop_loss_pct=0.06
    )

    # Report configuration
    report_output_dir = Path('logs/reports')
    risk_free_rate = 0.02  # 2% annual risk-free rate
    benchmark_name = "S&P 500"

    # ========== RUN BACKTEST ==========

    print(f"Loading data for {symbol}...")
    data_loader = DataLoader(Path('raw_data'))
    data = data_loader.load_csv(symbol, required_columns=['date', 'close', 'rsi_14'])
    print(f"✓ Loaded {len(data)} bars\n")

    print("Running backtest...")
    engine = SingleSecurityEngine(config)
    result = engine.run(symbol, data, strategy)
    print("✓ Backtest completed\n")

    # Display basic metrics
    print("PERFORMANCE SUMMARY")
    print("-" * 80)
    metrics = PerformanceMetrics.calculate_metrics(result)
    print(f"Total Return:     ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate:         {metrics['win_rate']*100:.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print("-" * 80 + "\n")

    # ========== GENERATE EXCEL REPORT ==========

    print("Generating comprehensive Excel report...")
    print(f"  - Initial Capital: ${initial_capital:,.2f}")
    print(f"  - Risk-Free Rate:  {risk_free_rate*100}%")
    print(f"  - Benchmark:       {benchmark_name}\n")

    # Create report generator
    excel_generator = ExcelReportGenerator(
        output_directory=report_output_dir,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        benchmark_name=benchmark_name
    )

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{strategy.get_name()}_{symbol}_{timestamp}.xlsx"

    report_path = excel_generator.generate_report(
        result=result,
        filename=report_filename
    )

    print("="*80)
    print("✓ EXCEL REPORT GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nReport Location: {report_path}")
    print(f"\nThe report includes 5 comprehensive sheets:")
    print("  1. Summary Dashboard     - Overall performance and risk metrics")
    print("  2. Trade Log             - Detailed transaction history")
    print("  3. Performance Analysis  - Drawdown analysis, monthly returns, distributions")
    print("  4. Visualizations        - Equity curve, drawdown chart, return distribution")
    print("  5. Market Conditions     - Template for market regime breakdown")
    print("\n" + "="*80)
    print("\nKey Metrics in the Report:")
    print("  ✓ Total Return, CAGR, Annualized Return")
    print("  ✓ Win Rate, Profit Factor, Risk/Reward Ratio")
    print("  ✓ Sharpe Ratio, Sortino Ratio, Calmar Ratio")
    print("  ✓ Maximum Drawdown, Volatility, Downside Deviation")
    print("  ✓ Win/Loss Streaks, Trade Duration Statistics")
    print("  ✓ Monthly/Weekly Consistency Metrics")
    print("  ✓ Trade Return Distribution")
    print("  ✓ Interactive Charts and Visualizations")
    print("="*80 + "\n")

    # Optional: Generate reports for multiple backtests
    generate_multiple = input("Would you like to generate reports for multiple symbols? (y/n): ")

    if generate_multiple.lower() == 'y':
        print("\nGenerating reports for multiple symbols...")

        # List of symbols to backtest
        symbols = ['MSFT', 'JPM']  # Add more symbols as needed

        for sym in symbols:
            try:
                print(f"\n  Processing {sym}...")

                # Load data
                data = data_loader.load_csv(sym, required_columns=['date', 'close', 'sma_50'])

                # Run backtest
                result = engine.run(sym, data, strategy)

                # Generate report
                report_path = excel_generator.generate_report(
                    result=result,
                    filename=f"{strategy.get_name()}_{sym}_{timestamp}.xlsx"
                )

                print(f"  ✓ Report generated: {report_path.name}")

            except Exception as e:
                print(f"  ✗ Failed to generate report for {sym}: {e}")

        print(f"\n✓ All reports saved to: {report_output_dir}")

    print("\nDone! Open the Excel files to view your comprehensive backtest reports.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
