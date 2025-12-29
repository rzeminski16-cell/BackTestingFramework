# BackTesting Framework - User Guide

A flow-based guide for developing, testing, and refining trading strategies.

---

## Overview

This guide follows the complete workflow from raw data collection through strategy refinement. The process is iterative - after initial backtesting, you'll cycle through optimization, analysis, and adjustment until achieving desired results.

```
                              BACKTESTING FRAMEWORK WORKFLOW

    +------------------+
    |  1. RAW DATA     |
    |   COLLECTION     |
    +--------+---------+
             |
             v
    +------------------+     +----------------------+
    |  2. STRATEGY     |<--->|  Strategy Explorer   |
    |   CREATION       |     |  (Understanding)     |
    +--------+---------+     +----------------------+
             |
             v
    +------------------+
    |  3. INITIAL      |
    |   BACKTESTING    |
    +--------+---------+
             |
             v
    +------------------+
    |  4. PARAMETER    |
    |   OPTIMIZATION   |
    +--------+---------+
             |
             v
    +------------------+
    |  5. VULNERABILITY|
    |   SCORING        |
    +--------+---------+
             |
             v
    +------------------+
    |  6. STATISTICAL  |  <-- Not yet implemented
    |   EVALUATION     |
    +--------+---------+
             |
             v
    +------------------+
    |  7. "WHAT IF"    |  <-- Not yet implemented
    |   ANALYSIS       |
    +--------+---------+
             |
             v
    +------------------+
    |  8. PER-TRADE    |  <-- Not yet implemented
    |   ANALYSIS       |
    +--------+---------+
             |
             |  +------------------------------------------+
             +->|  ITERATE: Repeat steps 2-8 until         |
                |  desired strategy performance achieved   |
                +------------------------------------------+
```

---

## Step 1: Raw Data Collection

Before running any backtests, you need historical market data. The framework collects data from Alpha Vantage API.

### Running the Data Collection GUI

```bash
python apps/data_collection_gui.py
```

### Data Types to Collect

| Data Type | Description | Storage Location |
|-----------|-------------|------------------|
| **Daily Prices** | Daily OHLCV + 50+ technical indicators | `raw_data/daily/{ticker}.csv` |
| **Weekly Prices** | Weekly OHLCV data | `raw_data/weekly/{ticker}.csv` |
| **Fundamental Data** | Financial statements, ratios, earnings | `raw_data/fundamentals/{ticker}/` |
| **Insider Data** | Insider transactions (buys/sells) | `raw_data/insider_transactions/{ticker}.csv` |
| **Forex Pairs** | Currency exchange rates for cost calculations | `raw_data/forex/{pair}.csv` |
| **Options** | Historical options chains (if available) | `raw_data/options/{ticker}/{year}/` |

### Collection Workflow

```
Data Collection GUI
        |
        v
+-------------------+
| 1. Select Preset  |  (All Stocks, Sectors, Custom, etc.)
+-------------------+
        |
        v
+-------------------+
| 2. Choose Data    |  (Daily, Weekly, Fundamentals, Insider, Options)
+-------------------+
        |
        v
+-------------------+
| 3. Set Options    |  (Full history vs. Update only)
+-------------------+
        |
        v
+-------------------+
| 4. Start Fetch    |  (Rate-limited: 75 req/min Premium)
+-------------------+
        |
        v
+-------------------+
| 5. Monitor Logs   |  (logs/data_collection/)
+-------------------+
```

### Ticker Presets

Pre-configured ticker lists are available in `config/data_collection/tickers.json`:
- **All Stocks** - 165+ US equities
- **All ETFs** - 46 exchange-traded funds
- **Sector-specific** - Tech, Healthcare, Financials, etc.
- **Forex Pairs** - Major currency pairs (for FX cost calculations)

### Tips

- **First Run**: Collect full history for your target securities
- **Updates**: Use "Update" mode to fetch only new data
- **Forex**: Collect relevant forex pairs (e.g., USDGBP) if backtesting with currency conversion costs
- **API Rate Limits**: Premium tier allows 75 requests/minute; the collector handles rate limiting automatically

---

## Step 2: Strategy Creation / Improvement

### Using the Included Strategy: AlphaTrend

The framework includes a production-ready strategy called **AlphaTrend** located at `strategies/alphatrend_strategy.py`.

**AlphaTrend Features:**
- Adaptive ATR-based trend following
- Dynamic MFI thresholds via percentile analysis
- Volume confirmation filter
- SMA-based exits with grace period & momentum protection
- Risk-based position sizing (default 2%)

**Required Data Columns:**
```
date, open, high, low, close, volume, atr_14, ema_50, mfi_14
```

### Strategy Understanding: AlphaTrend Explorer

To visualize and understand how AlphaTrend behaves on your data:

```bash
streamlit run apps/alphatrend_explorer.py
```

This opens an interactive dashboard showing:
- Price charts with indicator overlays
- Entry/exit signal visualization
- Parameter sensitivity analysis
- Trade outcome breakdowns

### Creating a New Strategy

To create a custom strategy, follow the standard strategy structure:

```
strategies/
    your_strategy.py
```

**Strategy Template:**

```python
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Models.signal import Signal, SignalType

class YourStrategy(BaseStrategy):
    """Your strategy description."""

    def __init__(self, param1: float = 10.0, param2: int = 14):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    @staticmethod
    def get_name() -> str:
        return "YourStrategy"

    @staticmethod
    def required_columns() -> list[str]:
        return ['date', 'open', 'high', 'low', 'close', 'volume']

    @staticmethod
    def get_parameter_info() -> dict:
        return {
            'param1': {
                'type': float,
                'default': 10.0,
                'min': 1.0,
                'max': 50.0,
                'description': 'Description of param1'
            },
            'param2': {
                'type': int,
                'default': 14,
                'min': 5,
                'max': 30,
                'description': 'Description of param2'
            }
        }

    def generate_signals(self, data, current_position=None):
        signals = []
        for i in range(len(data)):
            # Your entry/exit logic here
            if self._should_buy(data, i):
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    timestamp=data.iloc[i]['date'],
                    price=data.iloc[i]['close']
                ))
            elif self._should_sell(data, i, current_position):
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    timestamp=data.iloc[i]['date'],
                    price=data.iloc[i]['close']
                ))
            else:
                signals.append(Signal(
                    signal_type=SignalType.HOLD,
                    timestamp=data.iloc[i]['date']
                ))
        return signals
```

### Strategy Explorer for Custom Strategies

For new strategies, you'll need to create custom exploration scripts similar to `apps/alphatrend_explorer.py` to visualize and understand your strategy's behavior.

---

## Step 3: Initial Backtesting

Run backtests to evaluate strategy performance on historical data.

### Running the Backtest Wizard

```bash
python ctk_backtest_gui.py
```

### Backtest Workflow

```
Backtest Wizard
      |
      v
+------------------+
| 1. Select Mode   |  Single Security OR Portfolio
+------------------+
      |
      v
+------------------+
| 2. Choose        |  Select ticker(s) from available data
|    Securities    |
+------------------+
      |
      v
+------------------+
| 3. Select        |  AlphaTrendStrategy or custom
|    Strategy      |
+------------------+
      |
      v
+------------------+
| 4. Configure     |  Adjust strategy-specific parameters
|    Parameters    |
+------------------+
      |
      v
+------------------+
| 5. Set Capital   |  Initial capital, commission mode/value
|    & Commission  |
+------------------+
      |
      v
+------------------+
| 6. Date Range    |  Full history or specific period
|    (Optional)    |
+------------------+
      |
      v
+------------------+
| 7. Run Backtest  |
+------------------+
      |
      v
+------------------+
| 8. Review        |  Metrics, trade log, equity curve
|    Results       |
+------------------+
```

### Single Security vs Portfolio Mode

| Mode | Use Case | Capital Handling |
|------|----------|------------------|
| **Single Security** | Test strategy on one ticker | Isolated capital per security |
| **Portfolio** | Test across multiple tickers | Shared capital pool with position limits |

### Key Configuration Options

| Setting | Description | Example |
|---------|-------------|---------|
| **Initial Capital** | Starting portfolio value | 100,000 |
| **Commission Mode** | Percentage or Fixed | Percentage |
| **Commission Value** | Cost per trade | 0.001 (0.1%) |
| **Position Limits** | Max simultaneous positions (Portfolio) | 3 |

### Output Location

Results are saved to:
```
logs/backtests/
    single_security/
        {backtest_name}/
            {strategy}_{symbol}_trades.csv
            {strategy}_{symbol}_parameters.json
            reports/
                {strategy}_{symbol}_report.xlsx
    portfolio/
        {backtest_name}/
            trades/
            reports/
```

### Key Metrics to Evaluate

| Metric | Description | Target |
|--------|-------------|--------|
| **Total Return %** | Overall profit/loss | Positive |
| **Win Rate** | % of profitable trades | 40-60% (trend strategies) |
| **Profit Factor** | Gross profit / gross loss | > 1.5 |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Max Drawdown** | Largest peak-to-trough decline | < 20-25% |
| **Avg Trade Duration** | Mean holding period | Strategy-dependent |

---

## Step 4: Parameter Optimization

Find optimal strategy parameters using walk-forward optimization.

### Running the Optimization Wizard

```bash
python ctk_optimization_gui.py
```

### Optimization Workflow

```
Optimization Wizard
        |
        v
+--------------------+
| 1. Select Strategy |
+--------------------+
        |
        v
+--------------------+
| 2. Choose          |  Can optimize on single or multiple securities
|    Securities      |
+--------------------+
        |
        v
+--------------------+
| 3. Define Search   |  Parameter ranges to explore
|    Space           |
+--------------------+
        |
        v
+--------------------+
| 4. Select Speed    |  Quick / Fast / Full
|    Mode            |
+--------------------+
        |
        v
+--------------------+
| 5. Enable          |  Test parameter robustness
|    Sensitivity     |
+--------------------+
        |
        v
+--------------------+
| 6. Run             |  Bayesian optimization + walk-forward
|    Optimization    |
+--------------------+
        |
        v
+--------------------+
| 7. Review Results  |  Best parameters, in-sample vs out-of-sample
+--------------------+
```

### Speed Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Quick** | Fewer iterations, faster | Initial exploration |
| **Fast** | Balanced speed/thoroughness | Regular optimization |
| **Full** | Maximum iterations | Final parameter selection |

### Walk-Forward Validation

The optimizer uses walk-forward analysis to prevent overfitting:
1. **In-Sample**: Optimize parameters on training data
2. **Out-of-Sample**: Test optimized parameters on unseen data
3. **Roll Forward**: Repeat across multiple time windows

### Output Location

```
logs/optimization_reports/
    {optimization_name}/
        {strategy}_{symbol}_optimization.xlsx
```

### Interpreting Results

- **In-Sample Performance**: How well parameters fit the training data
- **Out-of-Sample Performance**: How well parameters generalize (more important)
- **Sensitivity Analysis**: Parameter stability - avoid cliff edges
- **Recommended Parameters**: Use parameters that perform well both in-sample AND out-of-sample

---

## Step 5: Vulnerability Score Visualization & Adjustment

After backtesting with optimized parameters, analyze capital allocation efficiency in portfolio mode.

### Running the Vulnerability Modeler

```bash
python ctk_vulnerability_gui.py
```

### Vulnerability Scoring Workflow

```
Vulnerability Modeler
        |
        v
+------------------------+
| 1. Load Portfolio      |  Load results from portfolio backtest
|    Backtest Results    |
+------------------------+
        |
        v
+------------------------+
| 2. View Current        |  See capital contention periods
|    Vulnerability Scores|
+------------------------+
        |
        v
+------------------------+
| 3. Adjust Scoring      |  Tune vulnerability parameters
|    Parameters          |
+------------------------+
        |
        v
+------------------------+
| 4. Visualize Impact    |  See how changes affect allocation
+------------------------+
        |
        v
+------------------------+
| 5. Apply New Settings  |  Use adjusted scores in backtesting
+------------------------+
```

### What Vulnerability Scoring Does

Vulnerability scoring helps manage capital allocation when multiple securities signal simultaneously:
- **High Score**: Security gets priority for capital
- **Low Score**: Security deferred when capital is limited
- **Factors**: Based on recent performance, drawdown, signal strength

### Tunable Parameters

- Score calculation weights
- Lookback periods
- Threshold adjustments
- Priority rules during capital contention

---

## Step 6: Statistical Evaluation

> **Status: Not Yet Implemented**

This step will involve analyzing correlations between trade outcomes and market factors:

- **Fundamental Data Analysis**: Examine whether successful trades correlate with specific fundamental indicators (P/E ratios, earnings surprises, etc.)
- **Insider Activity Correlation**: Determine if insider buying/selling precedes trade success/failure
- **Options Data Analysis**: Analyze put/call ratios, unusual options activity, and implied volatility around trade entries/exits
- **Statistical Significance Testing**: Validate whether observed correlations are statistically meaningful

The goal is to identify potential strategy improvements based on these market factors.

---

## Step 7: "What If Scenario" Analysis

> **Status: Not Yet Implemented**

This analysis examines alternative trade outcomes by exploring counterfactual scenarios:

- **Extended Holding Periods**: What if trades were held 30, 60, or 90 days longer?
- **Alternative Exit Timing**: How would different exit rules have affected outcomes?
- **Scenario Comparison**: Side-by-side view of actual vs. hypothetical results

This helps identify whether exit rules are too aggressive or too conservative.

---

## Step 8: Per-Trade Analysis

> **Status: Not Yet Implemented**

Deep-dive analysis of individual trades to understand success/failure patterns:

- **Trade Selection**: Select representative trades (winners, losers, breakeven)
- **Multi-Factor Examination**: Review all available data around each trade:
  - Price action and technicals before/during/after
  - Fundamental data at entry/exit
  - Insider activity timeline
  - Options market sentiment
  - Correlated security behavior
- **Pattern Identification**: Find common characteristics in successful vs. unsuccessful trades

---

## Iteration Cycle

After completing steps 1-8 (or 1-5 with current implementation), repeat the cycle:

```
     +-----> Strategy Adjustment <-----+
     |              |                  |
     |              v                  |
     |       Backtesting              |
     |              |                  |
     |              v                  |
     |       Optimization             |
     |              |                  |
     |              v                  |
     |       Analysis                 |
     |              |                  |
     +----- Results Review -----------+

     Repeat until desired performance achieved
```

### When to Iterate

- **Poor Win Rate**: Adjust entry conditions
- **Low Profit Factor**: Improve exit timing
- **High Drawdown**: Tighten stop losses or position sizing
- **Inconsistent Results**: Re-evaluate parameter stability
- **Overfitting Signs**: Simplify strategy or use more out-of-sample testing

---

## Quick Reference: Commands

| Task | Command |
|------|---------|
| **Data Collection** | `python apps/data_collection_gui.py` |
| **Run Backtest** | `python ctk_backtest_gui.py` |
| **Run Optimization** | `python ctk_optimization_gui.py` |
| **Vulnerability Modeler** | `python ctk_vulnerability_gui.py` |
| **AlphaTrend Explorer** | `streamlit run apps/alphatrend_explorer.py` |

---

## Directory Structure Reference

```
BackTestingFramework/
    |
    +-- apps/                          # GUI applications
    |   +-- data_collection_gui.py     # Data fetching interface
    |   +-- alphatrend_explorer.py     # Strategy visualization
    |   +-- backtest_gui.py            # Alternative backtest GUI
    |   +-- optimization_gui.py        # Alternative optimization GUI
    |   +-- vulnerability_gui.py       # Alternative vulnerability GUI
    |
    +-- strategies/                    # Trading strategies
    |   +-- alphatrend_strategy.py     # Included AlphaTrend strategy
    |
    +-- raw_data/                      # Historical data storage
    |   +-- daily/                     # Daily price data
    |   +-- weekly/                    # Weekly price data
    |   +-- fundamentals/              # Financial statements
    |   +-- insider_transactions/      # Insider activity
    |   +-- options/                   # Options chains
    |   +-- forex/                     # Currency rates
    |
    +-- logs/                          # Output & results
    |   +-- backtests/                 # Backtest results
    |   +-- optimization_reports/      # Optimization results
    |   +-- data_collection/           # Collection logs
    |
    +-- config/                        # Configuration files
    |   +-- data_collection/           # Ticker presets, API settings
    |   +-- baskets/                   # Portfolio definitions
    |   +-- strategy_presets/          # Saved strategy configs
    |   +-- vulnerability_presets/     # Scoring presets
    |
    +-- Classes/                       # Framework core modules
    +-- tools/                         # CLI utilities
    +-- docs/                          # Documentation
```

---

## Troubleshooting

### Data Collection Issues

| Problem | Solution |
|---------|----------|
| API rate limit errors | Wait for rate limit reset; collector handles automatically |
| Missing data columns | Verify API subscription includes required data types |
| Empty files | Check ticker symbol validity; some may be delisted |

### Backtest Issues

| Problem | Solution |
|---------|----------|
| "Required columns missing" | Ensure your CSV has all columns the strategy needs |
| "No trades generated" | Loosen strategy conditions or check date range |
| Results don't match TradingView | Export indicators from TradingView; check commission settings |

### GUI Issues

| Problem | Solution |
|---------|----------|
| GUI doesn't start | Install tkinter: `sudo apt-get install python3-tk` |
| CustomTkinter errors | `pip install customtkinter --upgrade` |
| Streamlit issues | `pip install streamlit --upgrade` |

---

## Next Steps

After familiarizing yourself with this workflow:

- **[TOOLS.md](TOOLS.md)** - Complete reference for all CLI tools and utilities
- **[STRATEGIES.md](STRATEGIES.md)** - Detailed guide for creating custom strategies
- **[CONFIGURATION.md](CONFIGURATION.md)** - All configuration options explained
- **[PORTFOLIO_MODE.md](PORTFOLIO_MODE.md)** - Advanced multi-security backtesting
- **[technical/ARCHITECTURE.md](technical/ARCHITECTURE.md)** - Framework internals
