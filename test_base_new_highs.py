"""
Test script for BaseNewHighs strategy.

This script tests the BaseNewHighs strategy (Balanced preset from Pine Script)
on available securities and generates comprehensive Excel reports.
"""
from pathlib import Path
from datetime import datetime

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Analysis.trade_logger import TradeLogger
from Classes.Analysis.excel_report_generator import ExcelReportGenerator

from strategies.base_new_highs_strategy import BaseNewHighsStrategy


def test_base_new_highs_strategy(symbol: str = 'AAPL', generate_excel: bool = True):
    """
    Test BaseNewHighs strategy on a single security.

    Args:
        symbol: Security symbol to test
        generate_excel: Whether to generate Excel report
    """
    print("\n" + "="*80)
    print(f"TESTING: BaseNewHighs Strategy on {symbol}")
    print("="*80)

    # Configure commission
    commission = CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # 0.1% commission
    )

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,  # Use framework's default capital
        commission=commission
    )

    # Load data
    data_loader = DataLoader(Path('raw_data/processed_exports'))

    try:
        # BaseNewHighs strategy requires: date, open, high, low, close
        data = data_loader.load_csv(symbol, required_columns=['date', 'open', 'high', 'low', 'close'])
        print(f"✓ Loaded {len(data)} bars for {symbol}")
        print(f"  Date range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")

        # Check if we have enough data (need at least 200 bars for SMA 200)
        if len(data) < 200:
            raise ValueError(f"Insufficient data for {symbol}: need at least 200 bars, got {len(data)}")

    except Exception as e:
        print(f"✗ Error loading data for {symbol}: {e}")
        return None

    # Create strategy with default parameters (Balanced preset)
    strategy = BaseNewHighsStrategy(
        new_high_n=14,
        sma_length=200,
        ma_fast_length=7,
        ma_slow_length=21,
        ma_lookback_k=14,
        sar_start=0.02,
        sar_increment=0.02,
        sar_maximum=0.2,
        ema_sell_length=14,
        ema_sell_threshold=3.0,
        atr_length=14,
        atr_multiplier=3.0,
        risk_percent=0.5
    )

    print(f"\n✓ Created strategy: {strategy}")
    print(f"\nStrategy Rules (Balanced Preset):")
    print(f"  Entry Rules:")
    print(f"    - New High: Close > highest high in past {strategy.new_high_n} candles")
    print(f"    - SMA: Close >= SMA({strategy.sma_length})")
    print(f"    - MA Cross: EMA({strategy.ma_fast_length}) x EMA({strategy.ma_slow_length}) within {strategy.ma_lookback_k} bars")
    print(f"    - SAR: Parabolic SAR < close (uptrend)")
    print(f"  Exit Rules:")
    print(f"    - Stop Loss: ATR({strategy.atr_length}) * {strategy.atr_multiplier}")
    print(f"    - Sell Signal: EMA({strategy.ema_sell_length}) > {strategy.ema_sell_threshold}% above price")
    print(f"  Position Sizing:")
    print(f"    - Risk: {strategy.risk_percent}% of equity per trade")
    print(f"    - Size: (Equity * {strategy.risk_percent}%) / Stop Distance")

    # Run backtest
    print(f"\n{'='*80}")
    print(f"Running backtest...")
    print(f"{'='*80}")

    engine = SingleSecurityEngine(config)
    result = engine.run(symbol, data, strategy)

    # Calculate and display metrics
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*80}")

    metrics = PerformanceMetrics.calculate_metrics(result)
    PerformanceMetrics.print_metrics(metrics)

    # Log trades to CSV
    logger = TradeLogger(Path('logs'))
    log_filename = logger.log_trades(symbol, strategy.get_name(), result.trades)
    print(f"\n✓ Trade log saved: {log_filename}")

    # Generate Excel report if requested
    if generate_excel:
        print(f"\n{'='*80}")
        print("Generating Excel Report...")
        print(f"{'='*80}")

        report_generator = ExcelReportGenerator(
            output_directory=Path('logs/reports'),
            initial_capital=config.initial_capital,
            risk_free_rate=0.02,  # 2% risk-free rate
            benchmark_name="S&P 500"
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"BaseNewHighs_{symbol}_{timestamp}.xlsx"

        report_path = report_generator.generate_report(
            result=result,
            filename=report_filename
        )

        print(f"✓ Excel report generated: {report_path}")
        print(f"\nThe report includes:")
        print("  - Sheet 1: Summary Dashboard (all key metrics)")
        print("  - Sheet 2: Trade Log (detailed transaction history)")
        print("  - Sheet 3: Performance Analysis (drawdown, monthly returns, distributions)")
        print("  - Sheet 4: Visualizations (equity curve, drawdown chart, return distribution)")
        print("  - Sheet 5: Market Conditions (template for regime analysis)")

    print(f"\n{'='*80}")
    print("Test completed successfully!")
    print(f"{'='*80}\n")

    return result


def test_multiple_securities(symbols: list = None, generate_excel: bool = True):
    """
    Test BaseNewHighs strategy on multiple securities.

    Args:
        symbols: List of symbols to test (None = all available)
        generate_excel: Whether to generate Excel reports
    """
    print("\n" + "="*80)
    print("TESTING: BaseNewHighs Strategy on Multiple Securities")
    print("="*80)

    # Get available symbols if not specified
    if symbols is None:
        data_loader = DataLoader(Path('raw_data/processed_exports'))
        symbols = data_loader.get_available_symbols()
        print(f"Found {len(symbols)} available securities: {', '.join(symbols)}")

    results = {}
    failed = []

    for symbol in symbols:
        try:
            result = test_base_new_highs_strategy(symbol, generate_excel)
            if result:
                results[symbol] = result
        except Exception as e:
            print(f"\n✗ Failed to test {symbol}: {e}")
            failed.append((symbol, str(e)))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if results:
        print(f"\n{'Symbol':<10} {'Trades':<10} {'P/L':<15} {'Return %':<12} {'Win Rate':<12}")
        print("-" * 65)

        for symbol, result in results.items():
            metrics = PerformanceMetrics.calculate_metrics(result)
            print(f"{symbol:<10} {result.num_trades:<10} ${result.total_return:>12,.2f} "
                  f"{result.total_return_pct:>10.2f}% {metrics['win_rate']:>10.1f}%")

    if failed:
        print(f"\nFailed securities ({len(failed)}):")
        for symbol, error in failed:
            print(f"  - {symbol}: {error}")

    print("\n" + "="*80 + "\n")

    return results


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Test specific symbol(s)
        symbols = sys.argv[1:]
        if len(symbols) == 1:
            test_base_new_highs_strategy(symbols[0], generate_excel=True)
        else:
            test_multiple_securities(symbols, generate_excel=True)
    else:
        # Default: test AAPL
        test_base_new_highs_strategy('AAPL', generate_excel=True)


if __name__ == "__main__":
    main()
