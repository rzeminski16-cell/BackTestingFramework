# Backtesting Framework Architecture

## Overview
A professional-grade backtesting framework for testing trading strategies across multiple securities with detailed trade logging and optimization capabilities.

## Design Principles
1. **Accuracy First**: Match TradingView results as closely as possible
2. **Modularity**: Clean separation of concerns
3. **Flexibility**: Support both simple and complex strategies
4. **Extensibility**: Easy to add new features and strategy types
5. **Type Safety**: Use type hints throughout

## Directory Structure

```
BackTestingFramework/
├── Classes/
│   ├── Config/
│   │   └── config.py                    # Configuration dataclasses
│   ├── Models/
│   │   ├── signal.py                    # Signal types and definitions
│   │   ├── position.py                  # Position tracking
│   │   ├── trade.py                     # Trade records
│   │   └── order.py                     # Order types
│   ├── Data/
│   │   ├── data_loader.py               # CSV loading
│   │   ├── data_validator.py            # Schema validation
│   │   └── security_registry.py         # Available securities tracking
│   ├── Strategy/
│   │   ├── base_strategy.py             # Abstract base strategy
│   │   └── strategy_context.py          # Context passed to strategies
│   ├── Engine/
│   │   ├── position_manager.py          # Position state management
│   │   ├── trade_executor.py            # Order execution
│   │   ├── single_security_engine.py    # Single security backtesting
│   │   └── portfolio_engine.py          # Portfolio backtesting
│   ├── Optimization/
│   │   ├── optimizer.py                 # Parameter optimization
│   │   └── optimization_result.py       # Results tracking
│   └── Analysis/
│       ├── trade_logger.py              # Trade logging to CSV
│       ├── performance_metrics.py       # Performance calculations
│       └── report_generator.py          # Report generation (placeholder)
├── strategies/                           # User-defined strategies
│   ├── examples/
│   │   ├── ma_cross_strategy.py
│   │   ├── advanced_strategy.py
│   │   └── conditional_exit_strategy.py
├── raw_data/                             # TradingView CSV files
├── logs/                                 # Trade logs per backtest
├── reports/                              # Generated reports
├── config/
│   └── security_metadata.json           # Security classifications
├── backtest.py                           # Main entry point
└── README.md                            # Documentation
```

## Core Components

### 1. Configuration (`Classes/Config/`)
- **BacktestConfig**: Backtest parameters (capital, dates, commission)
- **CommissionConfig**: Commission modes (percentage or fixed)
- **PortfolioConfig**: Portfolio-level settings

### 2. Models (`Classes/Models/`)
- **Signal**: Strategy signals (BUY, SELL, ADJUST_STOP, PARTIAL_EXIT)
- **Order**: Order representation with execution details
- **Position**: Position state with partial exit tracking
- **Trade**: Complete trade records with entry/exit details

### 3. Data Layer (`Classes/Data/`)
- **DataLoader**: Load CSV files with proper type conversion
- **DataValidator**: Validate schema and required columns
- **SecurityRegistry**: Track available securities and metadata

### 4. Strategy Layer (`Classes/Strategy/`)
- **BaseStrategy**: Abstract base class with rich feature support
  - Required columns declaration
  - Signal generation (entry, exit, adjustments)
  - Position sizing
  - Partial exit logic
  - Conditional stop loss adjustments
- **StrategyContext**: Immutable context passed to strategy methods

### 5. Execution Engine (`Classes/Engine/`)
- **PositionManager**:
  - Track position state (full position, partial exits)
  - Handle stop loss/take profit checks
  - Support indicator-based trailing stops
- **TradeExecutor**:
  - Execute orders at close prices
  - Calculate commissions (percentage or fixed)
  - Create trade records
- **SingleSecurityEngine**:
  - Backtest one security at a time
  - Isolated capital per security
- **PortfolioEngine**:
  - Backtest multiple securities simultaneously
  - Shared capital pool
  - Portfolio-level position sizing

### 6. Optimization (`Classes/Optimization/`)
- **StrategyOptimizer**:
  - Grid search over parameter combinations
  - Per-security optimization
  - Best parameter selection
- **OptimizationResult**:
  - Track results for each parameter set
  - Rank by performance metrics

### 7. Analysis (`Classes/Analysis/`)
- **TradeLogger**:
  - Write detailed CSV logs per security/strategy
  - One row per trade with all details
- **PerformanceMetrics**:
  - Calculate standard metrics (P/L, win rate, Sharpe, drawdown)
  - Support both single-security and portfolio metrics
- **ReportGenerator**:
  - Placeholder for future reporting features

## Data Flow

### Single Security Backtest:
```
Raw CSV → DataValidator → DataLoader → SingleSecurityEngine
                                              ↓
                                        Strategy.generate_signal()
                                              ↓
                                        PositionManager.update()
                                              ↓
                                        TradeExecutor.execute()
                                              ↓
                                    Trades List + Equity Curve
                                              ↓
                                    TradeLogger + PerformanceMetrics
```

### Portfolio Backtest:
```
Multiple CSVs → DataValidator → DataLoader → PortfolioEngine
                                                    ↓
                                    For each security on each day:
                                        Strategy.generate_signal()
                                                    ↓
                                        Shared capital allocation
                                                    ↓
                                        PositionManager.update()
                                                    ↓
                                        TradeExecutor.execute()
                                                    ↓
                            Combined Trades + Portfolio Equity Curve
                                                    ↓
                                    TradeLogger + PerformanceMetrics
```

### Optimization Flow:
```
Strategy + Parameter Grid → StrategyOptimizer
                                    ↓
                    For each parameter combination:
                        Run SingleSecurityEngine or PortfolioEngine
                                    ↓
                            Store OptimizationResult
                                    ↓
                    Rank and select best parameters
```

## Strategy Features

### Basic Features:
- Entry signals with position sizing
- Exit signals (full position close)
- Stop loss (fixed price or percentage)
- Take profit levels

### Advanced Features:
- **Partial Exits**: Scale out of positions at different levels
- **Trailing Stops**: Dynamic stop loss adjustments
- **Conditional Stops**: Adjust stops based on P/L thresholds and indicators
  - Example: "If P/L > 5%, move stop to EMA 14"
- **Multi-condition Logic**: Complex entry/exit rules
- **Time-based Rules**: Max holding period, time-of-day filters

## Execution Model

- **Entry**: Execute at bar's closing price
- **Exit**: Execute at bar's closing price
- **No Slippage**: Exact execution at close
- **Commission**: Configurable (percentage of trade value OR fixed amount)
- **Position Types**: LONG only
- **Fills**: Assume all orders fill at close price

## Key Design Decisions

1. **Bar-by-bar Processing**: Process each bar sequentially to avoid lookahead bias
2. **Point-in-time Data**: Strategies only see historical data up to current bar
3. **Immutable Context**: Strategy methods receive immutable context objects
4. **State Management**: PositionManager maintains all position state
5. **Clean Separation**: Strategies define logic, engines handle execution
6. **Type Safety**: Comprehensive type hints for better IDE support and fewer bugs

## Accuracy Guarantees

To match TradingView results:
1. Execute all trades at closing prices (same as Pine Script default)
2. No lookahead bias (strategies only see past data)
3. Precise commission calculations
4. Exact replication of signal logic from Pine Script
5. Bar-by-bar execution matching TradingView's model

## Extension Points

- Custom signal types (inherit from Signal)
- Custom position sizing algorithms
- Custom commission models
- Custom performance metrics
- Strategy families (inherit from BaseStrategy)
