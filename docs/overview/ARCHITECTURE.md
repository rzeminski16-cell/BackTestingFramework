# System Architecture

This document provides a high-level view of how the BackTesting Framework is organized and how its components interact.

---

## High-Level Architecture

The framework is built as a layered system where each layer has distinct responsibilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                                                                             │
│   ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│   │ Backtest │  │Optimization│  │Edge Analysis │  │  Factor Analysis  │   │
│   │   GUI    │  │    GUIs    │  │     GUI      │  │       GUI         │   │
│   └────┬─────┘  └─────┬──────┘  └──────┬───────┘  └─────────┬─────────┘   │
│        │              │                │                    │             │
└────────┼──────────────┼────────────────┼────────────────────┼─────────────┘
         │              │                │                    │
         ▼              ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │
│   │ Backtest Runner │  │   Optimizer     │  │    Analysis Tools       │   │
│   │                 │  │                 │  │                         │   │
│   │ • Single Mode   │  │ • Walk-Forward  │  │ • E-Ratio Calculator    │   │
│   │ • Portfolio Mode│  │ • Univariate    │  │ • Factor Analyzer       │   │
│   │                 │  │ • Bayesian      │  │ • Vulnerability Scorer  │   │
│   └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘   │
│            │                    │                        │                 │
└────────────┼────────────────────┼────────────────────────┼─────────────────┘
             │                    │                        │
             ▼                    ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                      │
│                                                                             │
│   ┌──────────────┐  ┌────────────────┐  ┌─────────────┐  ┌─────────────┐  │
│   │   Engines    │  │   Strategies   │  │   Models    │  │  Analysis   │  │
│   │              │  │                │  │             │  │             │  │
│   │ • Single     │  │ • BaseStrategy │  │ • Position  │  │ • Metrics   │  │
│   │ • Portfolio  │  │ • AlphaTrend   │  │ • Trade     │  │ • Reports   │  │
│   │ • Position   │  │ • Custom       │  │ • Signal    │  │ • Logging   │  │
│   └──────┬───────┘  └───────┬────────┘  └──────┬──────┘  └──────┬──────┘  │
│          │                  │                  │                │          │
└──────────┼──────────────────┼──────────────────┼────────────────┼──────────┘
           │                  │                  │                │
           ▼                  ▼                  ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│                                                                             │
│   ┌───────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│   │  DataLoader   │  │  Configuration  │  │      Data Storage           │  │
│   │               │  │                 │  │                             │  │
│   │ • CSV parsing │  │ • BacktestConf  │  │ • raw_data/daily/           │  │
│   │ • Validation  │  │ • PortfolioConf │  │ • raw_data/fundamentals/    │  │
│   │ • Alignment   │  │ • StrategyConf  │  │ • logs/backtests/           │  │
│   └───────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Map

This diagram shows how the main components connect and communicate:

```
                                    ┌─────────────────┐
                                    │  User Request   │
                                    │  (via GUI/API)  │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
           ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
           │  BacktestConf  │      │ PortfolioConf  │      │ OptimizeConf   │
           │                │      │                │      │                │
           │ • symbol       │      │ • symbols[]    │      │ • param_ranges │
           │ • date_range   │      │ • max_positions│      │ • train_period │
           │ • capital      │      │ • capital_mode │      │ • test_period  │
           │ • commission   │      │ • commission   │      │ • metric       │
           └───────┬────────┘      └───────┬────────┘      └───────┬────────┘
                   │                       │                       │
                   │                       │                       │
                   ▼                       ▼                       ▼
           ┌───────────────────────────────────────────────────────────────┐
           │                                                               │
           │                         DataLoader                            │
           │                                                               │
           │   Reads CSV files from raw_data/daily/{symbol}.csv            │
           │   Validates required columns exist                            │
           │   Aligns dates for portfolio mode                             │
           │                                                               │
           └───────────────────────────────┬───────────────────────────────┘
                                           │
                                           │ DataFrame(s)
                                           ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │                              Strategy                                    │
    │                                                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                                                                 │   │
    │   │  1. prepare_data(df) ──► Pre-calculate indicators              │   │
    │   │                                                                 │   │
    │   │  2. generate_signal(context) ──► Returns Signal                │   │
    │   │         │                           │                          │   │
    │   │         │                           │                          │   │
    │   │         ▼                           ▼                          │   │
    │   │   StrategyContext              ┌─────────┐                     │   │
    │   │   • current_bar                │ Signal  │                     │   │
    │   │   • position (if any)          │         │                     │   │
    │   │   • historical data            │ BUY     │                     │   │
    │   │   • parameters                 │ SELL    │                     │   │
    │   │                                │ HOLD    │                     │   │
    │   │                                │ ADJUST  │                     │   │
    │   │                                └─────────┘                     │   │
    │   │                                                                 │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └────────────────────────────────────────┬─────────────────────────────────┘
                                             │
                                             │ Signals
                                             ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │                           Backtesting Engine                             │
    │                                                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                                                                 │   │
    │   │   SingleSecurityEngine          OR        PortfolioEngine       │   │
    │   │   ├── PositionManager                     ├── PositionManager   │   │
    │   │   ├── TradeExecutor                       ├── TradeExecutor     │   │
    │   │   └── EquityTracker                       ├── CapitalAllocator  │   │
    │   │                                           └── VulnerabilityScorer│   │
    │   │                                                                 │   │
    │   │   For each bar:                                                 │   │
    │   │   1. Create StrategyContext                                     │   │
    │   │   2. Get Signal from Strategy                                   │   │
    │   │   3. Execute Signal via TradeExecutor                           │   │
    │   │   4. Update PositionManager                                     │   │
    │   │   5. Track equity                                               │   │
    │   │                                                                 │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └────────────────────────────────────────┬─────────────────────────────────┘
                                             │
                                             │ BacktestResult
                                             ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │                           Results & Analysis                             │
    │                                                                          │
    │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
    │   │ PerformanceMetrics│  │  TradeLogger     │  │ ExcelReportGenerator │  │
    │   │                  │  │                  │  │                      │  │
    │   │ • Total Return   │  │ • trades.csv     │  │ • Charts             │  │
    │   │ • Sharpe Ratio   │  │ • parameters.json│  │ • Summary            │  │
    │   │ • Max Drawdown   │  │ • equity curve   │  │ • Trade details      │  │
    │   │ • Win Rate       │  │                  │  │                      │  │
    │   │ • 50+ metrics    │  │                  │  │                      │  │
    │   └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
```

---

## Single Security Backtest Flow

A detailed view of how a single security backtest executes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SINGLE SECURITY BACKTEST                            │
└─────────────────────────────────────────────────────────────────────────────┘

      BacktestConfig                    DataLoader
      ┌─────────────┐                  ┌───────────┐
      │symbol: AAPL │                  │           │
      │capital: 100k│ ─────────────►   │ Load CSV  │
      │commission   │                  │ Validate  │
      └─────────────┘                  └─────┬─────┘
                                             │
                                             │ DataFrame with OHLCV + indicators
                                             ▼
                                   ┌─────────────────────┐
                                   │  Strategy.prepare() │
                                   │                     │
                                   │  Pre-calculate any  │
                                   │  strategy-specific  │
                                   │  indicators         │
                                   └──────────┬──────────┘
                                              │
                                              │ Prepared DataFrame
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         BAR-BY-BAR LOOP                                  │
    │                                                                         │
    │   for each bar (day) in data:                                          │
    │                                                                         │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │                     StrategyContext                          │  │
    │      │                                                              │  │
    │      │  • current_bar: {date, open, high, low, close, volume, ...} │  │
    │      │  • position: Position or None                               │  │
    │      │  • historical: all previous bars                            │  │
    │      │  • params: strategy parameters                              │  │
    │      └─────────────────────────────┬────────────────────────────────┘  │
    │                                    │                                    │
    │                                    ▼                                    │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │              strategy.generate_signal(context)               │  │
    │      │                                                              │  │
    │      │  Returns one of:                                            │  │
    │      │  • Signal(BUY, price, stop_loss)                            │  │
    │      │  • Signal(SELL)                                             │  │
    │      │  • Signal(HOLD)                                             │  │
    │      │  • Signal(PARTIAL_EXIT, quantity)                           │  │
    │      │  • Signal(ADJUST_STOP, new_stop)                            │  │
    │      └─────────────────────────────┬────────────────────────────────┘  │
    │                                    │                                    │
    │                                    ▼                                    │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │                    TradeExecutor                             │  │
    │      │                                                              │  │
    │      │  If BUY and no position:                                    │  │
    │      │    → Calculate position size (capital × risk)               │  │
    │      │    → Open Position at close price                           │  │
    │      │    → Deduct commission                                      │  │
    │      │                                                              │  │
    │      │  If SELL and has position:                                  │  │
    │      │    → Close Position at close price                          │  │
    │      │    → Calculate P/L                                          │  │
    │      │    → Deduct commission                                      │  │
    │      │    → Create Trade record                                    │  │
    │      └─────────────────────────────┬────────────────────────────────┘  │
    │                                    │                                    │
    │                                    ▼                                    │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │                   PositionManager                            │  │
    │      │                                                              │  │
    │      │  Tracks:                                                     │  │
    │      │  • current_position: Position or None                       │  │
    │      │  • closed_trades: list[Trade]                               │  │
    │      │  • equity_curve: list[float]                                │  │
    │      └──────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ After all bars processed
                                              ▼
                                   ┌─────────────────────┐
                                   │   BacktestResult    │
                                   │                     │
                                   │  • trades[]        │
                                   │  • equity_curve[]  │
                                   │  • final_capital   │
                                   │  • parameters_used │
                                   └──────────┬──────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │ PerformanceMetrics  │
                                   │                     │
                                   │ Calculate 50+ stats │
                                   └──────────┬──────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │   Output Files      │
                                   │                     │
                                   │ • trades.csv        │
                                   │ • report.xlsx       │
                                   │ • parameters.json   │
                                   └─────────────────────┘
```

---

## Portfolio Backtest Flow

Portfolio mode introduces capital management and multi-security coordination:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PORTFOLIO BACKTEST                                 │
└─────────────────────────────────────────────────────────────────────────────┘

      PortfolioConfig                 DataLoader
      ┌───────────────┐              ┌───────────────┐
      │symbols: [...]│              │               │
      │max_positions │ ────────────► │ Load all CSVs │
      │capital_mode  │              │ Align dates   │
      │allocation %  │              │ Create index  │
      └───────────────┘              └───────┬───────┘
                                             │
                                             │ Dict[symbol → DataFrame]
                                             ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        DAILY PROCESSING LOOP                            │
    │                                                                         │
    │   for each trading day in aligned_dates:                               │
    │                                                                         │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │              SIGNAL COLLECTION                               │  │
    │      │                                                              │  │
    │      │  for each symbol in portfolio:                              │  │
    │      │      context = create_context(symbol, date)                 │  │
    │      │      signal = strategy.generate_signal(context)             │  │
    │      │      if signal.type == BUY:                                 │  │
    │      │          pending_signals.append((symbol, signal))           │  │
    │      └─────────────────────────────┬────────────────────────────────┘  │
    │                                    │                                    │
    │                                    ▼                                    │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │              CAPITAL AVAILABILITY CHECK                      │  │
    │      │                                                              │  │
    │      │  available_capital = total_capital - allocated_capital      │  │
    │      │  current_positions = count(open_positions)                  │  │
    │      │                                                              │  │
    │      │  Can we open new positions?                                 │  │
    │      │  • available_capital > minimum_position_size?               │  │
    │      │  • current_positions < max_positions?                       │  │
    │      └─────────────────────────────┬────────────────────────────────┘  │
    │                                    │                                    │
    │                           ┌────────┴────────┐                          │
    │                           │                 │                          │
    │               Capital Available    Capital Constrained                 │
    │                           │                 │                          │
    │                           ▼                 ▼                          │
    │      ┌────────────────────────┐  ┌──────────────────────────────────┐  │
    │      │   EXECUTE SIGNALS      │  │  CAPITAL CONTENTION RESOLUTION   │  │
    │      │                        │  │                                  │  │
    │      │  Execute pending BUY   │  │  Mode: DEFAULT                   │  │
    │      │  signals until capital │  │  → Skip excess signals           │  │
    │      │  or position limit     │  │  → First-come-first-served       │  │
    │      │  reached               │  │                                  │  │
    │      │                        │  │  Mode: VULNERABILITY_SCORE       │  │
    │      └────────────────────────┘  │  → Score all positions           │  │
    │                                  │  → Close weakest if new signal   │  │
    │                                  │    has better score              │  │
    │                                  │  → Open new position             │  │
    │                                  └──────────────────────────────────┘  │
    │                                                                         │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │              POSITION UPDATES                                │  │
    │      │                                                              │  │
    │      │  for each open position:                                    │  │
    │      │      • Check stop loss triggers                             │  │
    │      │      • Process SELL signals                                 │  │
    │      │      • Process ADJUST_STOP signals                          │  │
    │      │      • Update position value with current price             │  │
    │      └─────────────────────────────┬────────────────────────────────┘  │
    │                                    │                                    │
    │                                    ▼                                    │
    │      ┌──────────────────────────────────────────────────────────────┐  │
    │      │              PORTFOLIO VALUE UPDATE                          │  │
    │      │                                                              │  │
    │      │  portfolio_value = cash + sum(position_values)              │  │
    │      │  equity_curve.append(portfolio_value)                       │  │
    │      └──────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ After all days processed
                                              ▼
                            ┌──────────────────────────────────┐
                            │       PortfolioResult            │
                            │                                  │
                            │  • per_symbol_results{}          │
                            │  • portfolio_trades[]            │
                            │  • portfolio_equity_curve[]      │
                            │  • aggregated_metrics            │
                            └──────────────────────────────────┘
```

---

## Optimization System Flow

How parameter optimization works:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WALK-FORWARD OPTIMIZATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

    Parameter Ranges                    Historical Data
    ┌────────────────┐                 ┌────────────────────────────────────┐
    │ atr_mult: 1-5  │                 │                                    │
    │ period: 10-50  │                 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
    │ ...           │                 │ 2018    2019    2020    2021    2022│
    └───────┬────────┘                 └───────────────────┬────────────────┘
            │                                              │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      WALK-FORWARD PERIODS                               │
    │                                                                         │
    │   Period 1:                                                            │
    │   ┌────────────────────────────┬────────────────┐                      │
    │   │     TRAINING (in-sample)   │   TEST (OOS)   │                      │
    │   │     Find best params       │   Validate     │                      │
    │   │     2018-01 to 2019-06     │   2019-07-12   │                      │
    │   └────────────────────────────┴────────────────┘                      │
    │                                                                         │
    │   Period 2:                                                            │
    │          ┌────────────────────────────┬────────────────┐               │
    │          │     TRAINING (in-sample)   │   TEST (OOS)   │               │
    │          │     2018-07 to 2020-00     │   2020-01-06   │               │
    │          └────────────────────────────┴────────────────┘               │
    │                                                                         │
    │   Period 3:                                                            │
    │                ┌────────────────────────────┬────────────────┐         │
    │                │     TRAINING (in-sample)   │   TEST (OOS)   │         │
    │                │     2019-01 to 2020-06     │   2020-07-12   │         │
    │                └────────────────────────────┴────────────────┘         │
    │                                                                         │
    │   ... continues rolling forward                                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      BAYESIAN OPTIMIZATION                              │
    │                      (within each training period)                      │
    │                                                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                                                                 │  │
    │   │  Iteration 1: Random parameters → Backtest → Sharpe = 0.8      │  │
    │   │  Iteration 2: Random parameters → Backtest → Sharpe = 1.2      │  │
    │   │  ...                                                           │  │
    │   │  Iteration N: Build model of parameter → performance           │  │
    │   │              → Explore promising regions                       │  │
    │   │              → Find optimal parameters                         │  │
    │   │                                                                 │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    └────────────────────────────────────────┬────────────────────────────────┘
                                             │
                                             ▼
                            ┌──────────────────────────────────┐
                            │     Optimization Result          │
                            │                                  │
                            │  • Best parameters per period    │
                            │  • Out-of-sample performance     │
                            │  • Parameter stability analysis  │
                            │  • Robustness metrics            │
                            └──────────────────────────────────┘
```

---

## File Structure Map

Where to find each component in the codebase:

```
BackTestingFramework/
│
├── Classes/                          # ◄── CORE FRAMEWORK
│   │
│   ├── Config/                       # Configuration objects
│   │   ├── config.py                 # BacktestConfig, PortfolioConfig
│   │   ├── capital_contention.py     # Capital allocation modes
│   │   └── basket.py                 # Security basket definitions
│   │
│   ├── Data/                         # Data loading and validation
│   │   ├── data_loader.py            # CSV loading, normalization
│   │   ├── data_validator.py         # Data quality checks
│   │   └── security_registry.py      # Security metadata
│   │
│   ├── Engine/                       # Backtesting engines
│   │   ├── single_security_engine.py # Single symbol backtests
│   │   ├── portfolio_engine.py       # Multi-symbol backtests
│   │   ├── position_manager.py       # Position tracking
│   │   ├── trade_executor.py         # Trade execution
│   │   └── vulnerability_score.py    # Position scoring
│   │
│   ├── Strategy/                     # Strategy framework
│   │   ├── base_strategy.py          # Abstract base class
│   │   ├── strategy_context.py       # Context passed to strategies
│   │   └── fundamental_rules.py      # Fundamental analysis rules
│   │
│   ├── Models/                       # Data models
│   │   ├── trade.py                  # Completed trade record
│   │   ├── position.py               # Open position
│   │   ├── signal.py                 # Trading signals
│   │   └── order.py                  # Order dataclass
│   │
│   ├── Analysis/                     # Analysis and reporting
│   │   ├── performance_metrics.py    # 50+ performance metrics
│   │   ├── trade_logger.py           # CSV trade logs
│   │   ├── excel_report_generator.py # Excel reports
│   │   └── backtest_analyzer/        # Deep trade analysis
│   │
│   ├── Optimization/                 # Parameter optimization
│   │   ├── walk_forward_optimizer.py # Walk-forward analysis
│   │   ├── univariate_optimizer.py   # Single parameter sweeps
│   │   └── sensitivity_analyzer.py   # Parameter sensitivity
│   │
│   ├── FactorAnalysis/               # Factor analysis system
│   │   ├── analyzer.py               # Main analysis engine
│   │   ├── factors/                  # Factor calculations
│   │   └── scenarios/                # Scenario detection
│   │
│   ├── VulnerabilityScorer/          # Position scoring
│   │   ├── scoring.py                # Score calculations
│   │   └── portfolio_simulator.py    # Portfolio simulation
│   │
│   └── RuleTester/                   # Rule-based testing
│       ├── rule_engine.py            # Rule evaluation
│       └── strategy_exit_rules.py    # Exit rule testing
│
├── strategies/                       # ◄── TRADING STRATEGIES
│   ├── alphatrend_strategy.py        # Production AlphaTrend
│   └── random_base_strategy.py       # Random baseline
│
├── backtesting/                      # ◄── ALTERNATIVE SYSTEM
│   ├── engine.py                     # Simplified engine
│   ├── portfolio.py                  # Portfolio management
│   └── strategies/                   # Example strategies
│
├── config/                           # ◄── CONFIGURATION
│   ├── strategy_parameters.json      # Parameter definitions
│   ├── baskets/                      # Security baskets
│   └── vulnerability_presets/        # Scoring presets
│
├── raw_data/                         # ◄── DATA STORAGE
│   ├── daily/                        # Daily price CSVs
│   ├── weekly/                       # Weekly price CSVs
│   └── fundamentals/                 # Fundamental data
│
├── logs/                             # ◄── OUTPUT
│   ├── backtests/                    # Backtest results
│   └── optimization_reports/         # Optimization results
│
└── *.py                              # ◄── GUI APPLICATIONS
    ├── ctk_main_gui.py               # Main launcher
    ├── ctk_backtest_gui.py           # Backtest GUI
    ├── ctk_optimization_gui.py       # Walk-forward GUI
    ├── ctk_univariate_optimization_gui.py
    ├── ctk_edge_analysis_gui.py      # E-ratio GUI
    ├── ctk_vulnerability_gui.py      # Vulnerability GUI
    └── ctk_factor_analysis_gui.py    # Factor analysis GUI
```

---

## Key Interfaces

### Strategy Interface

All strategies implement this interface:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BaseStrategy                                      │
│                                                                             │
│   Methods that must be implemented:                                         │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                                                                   │    │
│   │  get_name() → str                                                 │    │
│   │      Returns the strategy's display name                         │    │
│   │                                                                   │    │
│   │  required_columns() → list[str]                                  │    │
│   │      Returns columns needed in the data (e.g., ['close', 'atr']) │    │
│   │                                                                   │    │
│   │  get_parameter_info() → dict                                     │    │
│   │      Returns parameter definitions with types, defaults, ranges   │    │
│   │                                                                   │    │
│   │  generate_signal(context: StrategyContext) → Signal              │    │
│   │      Core method: decides BUY/SELL/HOLD for current bar          │    │
│   │                                                                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   Optional methods:                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                                                                   │    │
│   │  prepare_data(df) → df                                           │    │
│   │      Pre-calculate any strategy-specific indicators              │    │
│   │                                                                   │    │
│   │  calculate_position_size(context) → float                        │    │
│   │      Custom position sizing logic                                │    │
│   │                                                                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Signal Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Signal Types                                    │
│                                                                             │
│   ┌─────────────┐                                                          │
│   │    BUY      │  Open a new long position                                │
│   │             │  Includes: price, stop_loss, optional quantity           │
│   └─────────────┘                                                          │
│                                                                             │
│   ┌─────────────┐                                                          │
│   │    SELL     │  Close the current position entirely                     │
│   │             │  Includes: reason (stop_loss, signal, time_exit, etc.)   │
│   └─────────────┘                                                          │
│                                                                             │
│   ┌─────────────┐                                                          │
│   │    HOLD     │  No action - maintain current state                      │
│   │             │                                                          │
│   └─────────────┘                                                          │
│                                                                             │
│   ┌─────────────┐                                                          │
│   │PARTIAL_EXIT │  Close part of the position                              │
│   │             │  Includes: quantity to exit                              │
│   └─────────────┘                                                          │
│                                                                             │
│   ┌─────────────┐                                                          │
│   │ADJUST_STOP  │  Move stop loss to new level                             │
│   │             │  Includes: new_stop_price                                │
│   └─────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
                        DATA COLLECTION
                              │
                              ▼
    ┌────────────────────────────────────────────────┐
    │              raw_data/daily/*.csv              │
    │         (OHLCV + 50+ technical indicators)     │
    └────────────────────────────────────────────────┘
                              │
                              ▼
                         DATA LOADER
                              │
                              ▼
    ┌────────────────────────────────────────────────┐
    │                  STRATEGY                       │
    │          (generate signals bar-by-bar)         │
    └────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────┐
    │              BACKTESTING ENGINE                 │
    │           (execute trades, track P/L)          │
    └────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────┐
    │                  RESULTS                        │
    │        logs/backtests/*_trades.csv             │
    │        logs/backtests/*_report.xlsx            │
    └────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────┐
    │                 ANALYSIS                        │
    │     (E-ratio, Factor Analysis, Optimization)   │
    └────────────────────────────────────────────────┘
                              │
                              ▼
                         INSIGHTS
                              │
                              ▼
                    STRATEGY REFINEMENT
```

---

## Related Documentation

- [INTRODUCTION.md](INTRODUCTION.md) — Framework overview and getting started
- [QUICK_START.md](QUICK_START.md) — Run your first backtest
- [Strategy Development](../strategy-development/STRATEGY_GUIDE.md) — Creating custom strategies
- [Configuration Reference](../reference/CONFIGURATION.md) — All configuration options
