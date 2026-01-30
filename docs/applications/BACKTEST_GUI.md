# Backtest GUI

The Backtest GUI provides a wizard-based interface for running single security and portfolio backtests.

---

## Launching

```bash
python ctk_backtest_gui.py
```

Or click "Backtesting" from the main launcher.

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKTEST WIZARD STEPS                                │
└─────────────────────────────────────────────────────────────────────────────┘

    Step 1              Step 2              Step 3              Step 4
  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
  │  Mode   │   →    │Security │   →    │Strategy │   →    │ Capital │
  │Selection│        │Selection│        │  Setup  │        │ Config  │
  └─────────┘        └─────────┘        └─────────┘        └─────────┘
                                                                 │
                                                                 ▼
    Step 7              Step 6              Step 5
  ┌─────────┐        ┌─────────┐        ┌─────────┐
  │ Results │   ←    │   Run   │   ←    │  Date   │
  │ Review  │        │Backtest │        │  Range  │
  └─────────┘        └─────────┘        └─────────┘
```

---

## Step 1: Mode Selection

Choose between single security or portfolio testing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SELECT MODE                                       │
│                                                                             │
│   ┌──────────────────────────────┐  ┌──────────────────────────────┐       │
│   │                              │  │                              │       │
│   │     SINGLE SECURITY          │  │        PORTFOLIO             │       │
│   │                              │  │                              │       │
│   │  Test strategy on one        │  │  Test across multiple        │       │
│   │  ticker with isolated        │  │  securities with shared      │       │
│   │  capital                     │  │  capital pool                │       │
│   │                              │  │                              │       │
│   │  Best for:                   │  │  Best for:                   │       │
│   │  • Strategy development      │  │  • Realistic simulation      │       │
│   │  • Parameter tuning          │  │  • Capital allocation        │       │
│   │  • Quick testing             │  │  • Diversification           │       │
│   │                              │  │                              │       │
│   └──────────────────────────────┘  └──────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 2: Security Selection

### Single Security Mode

Select one ticker from available data:

```
Available Securities:
┌────────────────────────────────────────────────────────┐
│  Search: [____________]                                │
│                                                        │
│  ○ AAPL    Apple Inc.                                 │
│  ● MSFT    Microsoft Corporation        ◄── Selected  │
│  ○ GOOGL   Alphabet Inc.                              │
│  ○ AMZN    Amazon.com Inc.                            │
│  ...                                                   │
└────────────────────────────────────────────────────────┘

Data available: 2018-01-02 to 2023-12-29 (1,507 bars)
```

### Portfolio Mode

Select multiple securities or use a basket:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY SELECTION                                   │
│                                                                             │
│   Use Basket: [Large_Cap_15          ▼]     [Load Basket]                  │
│                                                                             │
│   Or select individually:                                                  │
│                                                                             │
│   Available                    Selected                                    │
│   ┌─────────────────────┐     ┌─────────────────────┐                     │
│   │ AAPL                │     │ MSFT         [X]    │                     │
│   │ GOOGL               │ ──► │ NVDA         [X]    │                     │
│   │ AMZN                │     │ JPM          [X]    │                     │
│   │ META                │ ◄── │                     │                     │
│   │ ...                 │     │                     │                     │
│   └─────────────────────┘     └─────────────────────┘                     │
│                                                                             │
│   Selected: 3 securities                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 3: Strategy Setup

### Select Strategy

Choose from available strategies:

```
Available Strategies:
┌────────────────────────────────────────────────────────┐
│  ● AlphaTrendStrategy                   ◄── Selected  │
│  ○ RandomControlStrategy                              │
│  ○ [Custom strategies appear here]                    │
└────────────────────────────────────────────────────────┘
```

### Configure Parameters

Adjust strategy-specific parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALPHATRENDSTRATEGY PARAMETERS                            │
│                                                                             │
│   Parameter              Value          Range                              │
│   ─────────────────────────────────────────────────────                    │
│   atr_multiplier         [  1.5  ]      0.5 - 5.0                         │
│   atr_period             [  14   ]      5 - 50                            │
│   mfi_period             [  14   ]      5 - 30                            │
│   volume_sma_period      [  20   ]      10 - 50                           │
│   sma_exit_period        [  10   ]      5 - 30                            │
│   risk_per_trade         [ 0.02  ]      0.01 - 0.05                       │
│   max_holding_days       [  10   ]      5 - 30                            │
│                                                                             │
│   [Load Preset ▼]     [Save Preset]     [Reset Defaults]                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Use Presets

Load saved parameter configurations:

```
Available Presets:
├── default          Original strategy defaults
├── conservative     Lower risk, tighter stops
├── aggressive       Higher risk, wider stops
└── optimized_2023   Parameters from 2023 optimization
```

---

## Step 4: Capital Configuration

### Basic Settings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CAPITAL CONFIGURATION                                  │
│                                                                             │
│   Initial Capital:     [$ 100,000    ]                                     │
│                                                                             │
│   Commission Mode:     ○ Percentage   ● Fixed                              │
│   Commission Value:    [$ 5.00       ]  (per trade)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Portfolio Mode Settings

Additional options for portfolio backtests:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO SETTINGS                                       │
│                                                                             │
│   Max Positions:           [  3  ]       Maximum concurrent positions      │
│                                                                             │
│   Position Size Limit:     [ 40  ] %     Max capital per position          │
│                                                                             │
│   Total Allocation Limit:  [ 95  ] %     Max total capital deployed        │
│                                                                             │
│   Capital Contention:      ○ Default (first-come-first-served)             │
│                            ● Vulnerability Score (score-based)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 5: Date Range (Optional)

Filter the backtest period:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATE RANGE                                          │
│                                                                             │
│   ☐ Use full data range (2018-01-02 to 2023-12-29)                        │
│                                                                             │
│   ☑ Custom range:                                                          │
│      Start Date: [2020-01-01]                                              │
│      End Date:   [2023-12-31]                                              │
│                                                                             │
│   Resulting period: 1,007 trading days                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 6: Run Backtest

Review settings and execute:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKTEST SUMMARY                                     │
│                                                                             │
│   Mode:           Portfolio                                                 │
│   Securities:     MSFT, NVDA, JPM (3 total)                                │
│   Strategy:       AlphaTrendStrategy                                        │
│   Capital:        $100,000                                                  │
│   Commission:     $5.00 fixed                                               │
│   Date Range:     2020-01-01 to 2023-12-31                                 │
│   Max Positions:  3                                                         │
│                                                                             │
│                    [  Run Backtest  ]                                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Progress: ████████████████████████████░░░░  75%                   │  │
│   │  Processing: NVDA (2 of 3)                                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 7: Review Results

### Metrics Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BACKTEST RESULTS                                      │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  PERFORMANCE METRICS                                              │    │
│   ├───────────────────────────────────────────────────────────────────┤    │
│   │                                                                   │    │
│   │  Total Return:      +24.5%  ($24,500)                            │    │
│   │  Sharpe Ratio:      1.42                                          │    │
│   │  Max Drawdown:      -8.7%   ($8,700)                             │    │
│   │                                                                   │    │
│   │  Total Trades:      47                                            │    │
│   │  Win Rate:          55.3%                                         │    │
│   │  Profit Factor:     1.85                                          │    │
│   │  Avg Trade:         +0.52%                                        │    │
│   │                                                                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Equity Curve

Visual representation of portfolio value over time:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Portfolio Value                                                            │
│  $125k ┤                                                    ╭──────        │
│        │                                              ╭─────╯              │
│  $115k ┤                                    ╭────────╯                     │
│        │                              ╭─────╯                              │
│  $105k ┤                     ╭────────╯                                    │
│        │            ╭────────╯                                             │
│  $100k ┼────────────╯                                                      │
│        │                                                                   │
│   $95k ┤                                                                   │
│        ┼────────┬────────┬────────┬────────┬────────┬────────┬─────────   │
│        2020    2020.5   2021    2021.5   2022    2022.5   2023            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Trade Log

Detailed list of all trades:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRADE LOG                                                    [Export CSV] │
│                                                                             │
│  #   Symbol  Entry Date   Entry$   Exit Date    Exit$    P/L%    Reason   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  1   MSFT    2020-02-15   185.50   2020-02-28   192.30   +3.7%   Signal   │
│  2   NVDA    2020-03-10   245.00   2020-03-13   238.20   -2.8%   Stop     │
│  3   JPM     2020-03-25   95.50    2020-04-08   102.75   +7.6%   Signal   │
│  4   MSFT    2020-04-20   175.00   2020-05-01   180.50   +3.1%   Time     │
│  ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Per-Symbol Breakdown (Portfolio Mode)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PER-SYMBOL PERFORMANCE                                                     │
│                                                                             │
│  Symbol   Trades   Win Rate   Return    Sharpe   Contribution              │
│  ────────────────────────────────────────────────────────────              │
│  MSFT       18      61.1%     +12.3%     1.65      +$7,380                 │
│  NVDA       15      53.3%     +9.8%      1.28      +$5,880                 │
│  JPM        14      50.0%     +2.4%      0.85      +$1,440                 │
│  ────────────────────────────────────────────────────────────              │
│  TOTAL      47      55.3%     +24.5%     1.42     +$14,700                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files

Results are saved automatically:

### Single Security

```
logs/backtests/single_security/{backtest_name}/
├── {strategy}_{symbol}_trades.csv        Trade log
├── {strategy}_{symbol}_parameters.json   Parameters used
└── reports/
    └── {strategy}_{symbol}_report.xlsx   Excel report
```

### Portfolio

```
logs/backtests/portfolio/{backtest_name}/
├── portfolio_trades.csv                  All trades
├── portfolio_summary.json                Aggregated metrics
├── trades/
│   └── {symbol}_trades.csv               Per-symbol trades
└── reports/
    └── portfolio_report.xlsx             Excel report
```

---

## Tips

### Starting Out

- Begin with single security mode to understand the strategy
- Use the included AlphaTrendStrategy before creating custom strategies
- Start with default parameters before optimizing

### Portfolio Testing

- Limit to 3-5 positions initially to understand capital flow
- Watch for capital contention events (signals skipped)
- Compare DEFAULT vs VULNERABILITY_SCORE contention modes

### Interpreting Results

- Sharpe > 1.0 suggests good risk-adjusted returns
- Win rate alone is not enough—check profit factor
- Watch max drawdown to understand worst-case scenarios

---

## Related Documentation

- [Backtesting Concepts](../concepts/BACKTESTING.md) — Core concepts
- [Portfolio Mode](../concepts/PORTFOLIO_MODE.md) — Multi-security details
- [Optimization Tools](OPTIMIZATION_GUI.md) — Parameter optimization
- [Configuration Reference](../reference/CONFIGURATION.md) — All settings
