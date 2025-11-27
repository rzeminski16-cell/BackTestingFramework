"""
Main entry point for backtesting framework.

This script demonstrates how to use the framework for:
1. Single security backtesting
2. Portfolio backtesting
3. Parameter optimization
4. Trade logging and performance analysis
"""
from pathlib import Path
from datetime import datetime

from Classes.Config.config import (
    BacktestConfig, PortfolioConfig, OptimizationConfig,
    CommissionConfig, CommissionMode
)
from Classes.Data.data_loader import DataLoader
from Classes.Data.security_registry import SecurityRegistry
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Optimization.optimizer import StrategyOptimizer
from Classes.Analysis.trade_logger import TradeLogger
from Classes.Analysis.performance_metrics import PerformanceMetrics

from strategies.examples import SimpleMAStrategy, AdvancedTrailingStopStrategy, PartialExitStrategy


def example_single_security_backtest():
    """
    Example: Backtest a single security with a strategy.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Security Backtest")
    print("="*80)

    # Configure commission (percentage-based)
    commission = CommissionConfig(
        mode=CommissionMode.PERCENTAGE,
        value=0.001  # 0.1% commission
    )

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        commission=commission,
        # start_date=datetime(2020, 1, 1),  # Optional date filtering
        # end_date=datetime(2023, 12, 31)
    )

    # Load data
    data_loader = DataLoader(Path('raw_data'))
    data = data_loader.load_csv('AAPL', required_columns=['date', 'close', 'sma_50'])

    print(f"Loaded {len(data)} bars for AAPL")

    # Create strategy
    strategy = SimpleMAStrategy(
        ma_period=50,
        position_size=0.2,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )

    # Run backtest
    engine = SingleSecurityEngine(config)
    result = engine.run('AAPL', data, strategy)

    # Calculate and display metrics
    metrics = PerformanceMetrics.calculate_metrics(result)
    PerformanceMetrics.print_metrics(metrics)

    # Log trades
    logger = TradeLogger(Path('logs'))
    logger.log_trades('AAPL', strategy.get_name(), result.trades)

    return result


def example_portfolio_backtest():
    """
    Example: Backtest multiple securities with shared capital.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Portfolio Backtest")
    print("="*80)

    # Configure commission (fixed per trade)
    commission = CommissionConfig(
        mode=CommissionMode.FIXED,
        value=3.0  # £3 per trade
    )

    # Configure portfolio backtest
    config = PortfolioConfig(
        initial_capital=100000.0,
        commission=commission,
        max_positions=3,  # Max 3 concurrent positions
        position_size_limit=0.3,  # Max 30% per position
        total_allocation_limit=0.9  # Max 90% total allocation
    )

    # Load data for multiple securities
    data_loader = DataLoader(Path('raw_data'))
    symbols = ['AAPL', 'MSFT', 'JPM']

    data_dict = {}
    for symbol in symbols:
        try:
            data = data_loader.load_csv(symbol, required_columns=['date', 'close', 'sma_50'])
            data_dict[symbol] = data
            print(f"Loaded {len(data)} bars for {symbol}")
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")

    if not data_dict:
        print("No data loaded, skipping portfolio backtest")
        return

    # Create strategy
    strategy = SimpleMAStrategy(ma_period=50, position_size=0.3)

    # Run portfolio backtest
    engine = PortfolioEngine(config)
    results = engine.run(data_dict, strategy)

    # Display results for each security
    print(f"\nPortfolio Results:")
    print(f"{'Symbol':<10} {'Trades':<10} {'P/L':<15} {'Return %':<12}")
    print("-" * 50)

    total_pl = 0
    for symbol, result in results.items():
        total_pl += result.total_return
        print(f"{symbol:<10} {result.num_trades:<10} ${result.total_return:>12,.2f} {result.total_return_pct:>10.2f}%")

    print("-" * 50)
    print(f"{'TOTAL':<10} {'':<10} ${total_pl:>12,.2f} {total_pl/config.initial_capital*100:>10.2f}%")

    # Log all trades
    logger = TradeLogger(Path('logs'))
    for symbol, result in results.items():
        logger.log_trades(symbol, strategy.get_name(), result.trades)

    return results


def example_parameter_optimization():
    """
    Example: Optimize strategy parameters for best performance.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Parameter Optimization")
    print("="*80)

    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
    )

    # Configure optimization
    opt_config = OptimizationConfig(
        metric='total_return_pct',  # Optimize for return percentage
        per_security=False,  # Find global best parameters
        min_trades=5,  # Require at least 5 trades
        maximize=True  # Maximize the metric
    )

    # Define parameter grid to search
    param_grid = {
        'ma_period': [20, 50, 100],
        'position_size': [0.1, 0.2, 0.3],
        'stop_loss_pct': [0.03, 0.05, 0.08]
    }

    print(f"Testing {3 * 3 * 3} parameter combinations...")

    # Load data
    data_loader = DataLoader(Path('raw_data'))
    data = data_loader.load_csv('AAPL')

    # Run optimization
    optimizer = StrategyOptimizer(opt_config, backtest_config)
    results = optimizer.optimize(SimpleMAStrategy, param_grid, 'AAPL', data)

    # Display results
    print(f"\n{results.summary()}")

    best = results.get_best()
    print(f"\nBest parameters: {best.parameters}")
    print(f"Return: {best.backtest_result.total_return_pct:.2f}%")
    print(f"Trades: {best.backtest_result.num_trades}")

    return results


def example_advanced_strategy():
    """
    Example: Use advanced strategy with trailing stops.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Advanced Strategy with Trailing Stops")
    print("="*80)

    config = BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
    )

    # Load data with required indicators
    data_loader = DataLoader(Path('raw_data'))
    data = data_loader.load_csv('AAPL', required_columns=[
        'date', 'close', 'sma_200', 'ema_14', 'rsi_14'
    ])

    print(f"Loaded {len(data)} bars for AAPL")

    # Create advanced strategy
    strategy = AdvancedTrailingStopStrategy(
        sma_period=200,
        ema_period=14,
        rsi_period=14,
        position_size=0.25,
        breakeven_profit_pct=0.05  # Move to breakeven after 5% profit
    )

    # Run backtest
    engine = SingleSecurityEngine(config)
    result = engine.run('AAPL', data, strategy)

    # Display metrics
    metrics = PerformanceMetrics.calculate_metrics(result)
    PerformanceMetrics.print_metrics(metrics)

    # Log trades
    logger = TradeLogger(Path('logs'))
    logger.log_trades('AAPL', strategy.get_name(), result.trades)

    return result


def example_partial_exits():
    """
    Example: Strategy with partial position exits.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Strategy with Partial Exits")
    print("="*80)

    config = BacktestConfig(
        initial_capital=100000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
    )

    # Load data
    data_loader = DataLoader(Path('raw_data'))
    data = data_loader.load_csv('BTCUSD', required_columns=['date', 'close', 'rsi_14'])

    print(f"Loaded {len(data)} bars for BTCUSD")

    # Create partial exit strategy
    strategy = PartialExitStrategy(
        rsi_period=14,
        position_size=0.3,
        first_target_pct=0.10,  # Take 50% profit at 10% gain
        second_target_pct=0.20   # Take another 50% at 20% gain
    )

    # Run backtest
    engine = SingleSecurityEngine(config)
    result = engine.run('BTCUSD', data, strategy)

    # Display metrics
    metrics = PerformanceMetrics.calculate_metrics(result)
    PerformanceMetrics.print_metrics(metrics)

    # Show trades with partial exits
    print("\nTrades with partial exits:")
    for trade in result.trades:
        if trade.partial_exits > 0:
            print(f"  {trade.entry_date.strftime('%Y-%m-%d')} -> {trade.exit_date.strftime('%Y-%m-%d')}: "
                  f"P/L ${trade.pl:.2f} ({trade.pl_pct:.2f}%), "
                  f"{trade.partial_exits} partial exits")

    # Log trades
    logger = TradeLogger(Path('logs'))
    logger.log_trades('BTCUSD', strategy.get_name(), result.trades)

    return result


def main():
    """
    Main function - run all examples.
    """
    print("\n" + "="*80)
    print("BACKTESTING FRAMEWORK - EXAMPLES")
    print("="*80)

    try:
        # Example 1: Single security backtest
        example_single_security_backtest()

        # Example 2: Portfolio backtest
        example_portfolio_backtest()

        # Example 3: Parameter optimization
        # example_parameter_optimization()  # Commented out by default (takes time)

        # Example 4: Advanced strategy
        # example_advanced_strategy()  # Uncomment if you have required indicators

        # Example 5: Partial exits
        # example_partial_exits()  # Uncomment to run

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Examples completed! Check the 'logs' directory for trade logs.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
