# Portfolio Mode Guide

This guide explains how to backtest strategies across multiple securities with shared capital.

## Single Security vs Portfolio Mode

| Aspect | Single Security | Portfolio |
|--------|-----------------|-----------|
| Capital | Isolated per security | Shared across all |
| Positions | One at a time | Multiple concurrent |
| Competition | None | Securities compete for capital |
| Results | Per-security | Aggregated portfolio |

## When to Use Portfolio Mode

Use portfolio mode when you want to:

1. **Test diversification** - Does the strategy work across multiple stocks?
2. **Simulate real trading** - With limited capital, you can't trade everything
3. **Manage position limits** - Cap exposure to any single security
4. **Test capital allocation** - How should capital be split?

---

## Using Portfolio Mode (GUI)

### Starting

```bash
python run_gui.py
```

### Step 1: Select Portfolio Mode

Click the "Portfolio Mode" toggle at the top.

### Step 2: Select Securities

Select multiple securities from the list:
- Hold Ctrl (or Cmd) and click to select multiple
- Or use "Select All" for all available securities

### Step 3: Configure Portfolio Settings

| Setting | Description | Example |
|---------|-------------|---------|
| **Max Positions** | Maximum concurrent open trades | 3 |
| **Position Size Limit** | Max capital per position (fraction) | 0.3 (30%) |
| **Total Allocation Limit** | Max total capital in positions | 0.9 (90%) |

### Step 4: Set Strategy and Commission

Same as single-security mode.

### Step 5: Run

Click "Run Backtest" and view aggregated results.

---

## Using Portfolio Mode (Python)

```python
from pathlib import Path
from Classes.Config.config import PortfolioConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.portfolio_engine import PortfolioEngine
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure portfolio
config = PortfolioConfig(
    initial_capital=100000.0,
    max_positions=3,
    position_size_limit=0.3,      # Max 30% per position
    total_allocation_limit=0.9,   # Max 90% total
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001)
)

# Load data for multiple securities
loader = DataLoader(Path('raw_data'))
data_dict = {
    'AAPL': loader.load_csv('AAPL'),
    'MSFT': loader.load_csv('MSFT'),
    'GOOGL': loader.load_csv('GOOGL'),
    'AMZN': loader.load_csv('AMZN'),
}

# Run portfolio backtest
strategy = AlphaTrendStrategy()
engine = PortfolioEngine(config)
results = engine.run(data_dict, strategy)

# View results
print("Portfolio Results:")
for symbol, result in results.symbol_results.items():
    print(f"  {symbol}: {result.total_return_pct:.2f}%")

print(f"\nTotal Portfolio Return: {results.portfolio_return_pct:.2f}%")
```

---

## How Capital Allocation Works

### Position Size Limits

When a BUY signal occurs, the framework calculates:

```
Position Value = min(
    Strategy's requested size × Available Capital,
    Position Size Limit × Initial Capital,
    Available Capital
)
```

**Example:**
- Initial capital: £100,000
- Position size limit: 30%
- Available capital: £70,000
- Strategy requests: 50% of capital

Result: Position is £30,000 (limited by 30% of initial capital)

### Total Allocation Limit

Prevents going "all in":

```
Max Capital in Positions = Total Allocation Limit × Initial Capital
```

**Example:**
- Initial capital: £100,000
- Total allocation limit: 90%
- Max in positions: £90,000
- Cash reserve: £10,000 minimum

### Max Positions

Limits concurrent open positions. If at max:
- New signals are rejected (DEFAULT mode)
- Or weak positions are closed (VULNERABILITY_SCORE mode)

---

## Capital Contention

When multiple securities signal at once, there may not be enough capital for all. This is called **capital contention**.

### DEFAULT Mode

First-come-first-served based on processing order.

```python
config = PortfolioConfig(
    capital_contention=CapitalContentionConfig(
        mode=CapitalContentionMode.DEFAULT
    )
)
```

Signals are processed in order. Later signals are rejected if capital runs out.

### VULNERABILITY_SCORE Mode

Evaluates existing positions and may close weak ones to make room for new signals.

```python
from Classes.Config.config import CapitalContentionConfig, CapitalContentionMode

config = PortfolioConfig(
    capital_contention=CapitalContentionConfig(
        mode=CapitalContentionMode.VULNERABILITY_SCORE,
        swap_threshold=50.0
    )
)
```

Each open position gets a vulnerability score (0-100):
- 100 = Strong, protected
- 0 = Weak, can be swapped

If a new signal arrives and capital is insufficient:
1. Calculate vulnerability scores for all open positions
2. Find the weakest position
3. If weakest score < threshold, close it and open new position
4. Otherwise, reject the new signal

### What Makes a Position Vulnerable?

Factors that increase vulnerability (lower score):
- **Days held** - Older positions decay over time
- **Low P/L** - Stagnant or losing positions
- **Negative momentum** - Declining recently
- **High volatility** - Unstable price action

Factors that decrease vulnerability (higher score):
- **New positions** - Protected by immunity period
- **High P/L** - Profitable positions
- **Positive momentum** - Rising recently

### Configuring Vulnerability Scoring

Use preset files in `config/vulnerability_presets/`:

| Preset | Behavior |
|--------|----------|
| `default.json` | Balanced approach |
| `aggressive.json` | Quick to swap weak positions |
| `conservative.json` | Protect positions longer |
| `momentum_focused.json` | Emphasize recent momentum |

---

## Multi-Currency Support

Portfolio mode supports securities in different currencies.

### Setup

1. **Configure security currencies** in `config/security_metadata.json`:
```json
{
  "AAPL": { "currency": "USD" },
  "VOD.L": { "currency": "GBP" }
}
```

2. **Provide exchange rates** in `currency_rates/`:
```
currency_rates/
├── USDGBP.csv
├── EURGBP.csv
```

3. **Initialize converters**:
```python
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Data.security_registry import SecurityRegistry

registry = SecurityRegistry(Path('config/security_metadata.json'))
converter = CurrencyConverter(base_currency='GBP')
converter.load_rates_directory(Path('currency_rates/'))

engine = PortfolioEngine(
    config=config,
    currency_converter=converter,
    security_registry=registry
)
```

All P/L is converted to base currency (GBP) for aggregation.

---

## Portfolio Results

### Symbol Results

Per-security performance:
```python
for symbol, result in results.symbol_results.items():
    print(f"{symbol}:")
    print(f"  Trades: {result.num_trades}")
    print(f"  Return: {result.total_return_pct:.2f}%")
    print(f"  Win Rate: {result.win_rate * 100:.1f}%")
```

### Portfolio Equity Curve

Combined portfolio value over time:
```python
equity_curve = results.portfolio_equity_curve
# DataFrame with date and portfolio_value columns
```

### Signal Rejections

Signals that couldn't be executed due to capital constraints:
```python
for rejection in results.signal_rejections:
    print(f"{rejection['date']}: {rejection['symbol']} - {rejection['reason']}")
```

### Vulnerability Swaps

Positions that were closed to make room for new signals:
```python
for swap in results.vulnerability_swaps:
    print(f"{swap['date']}: Closed {swap['closed_symbol']} for {swap['new_symbol']}")
```

---

## Best Practices

### 1. Set Reasonable Position Limits

- **Position size limit**: 20-40% typically
- **Max positions**: 3-5 for focused portfolios
- **Total allocation**: 80-95% (keep some cash reserve)

### 2. Use Similar Securities

- Test on securities with similar characteristics
- E.g., all tech stocks, all large caps
- A strategy that works on AAPL may not work on penny stocks

### 3. Watch for Correlation

- Highly correlated securities = less diversification benefit
- If all your stocks move together, portfolio mode adds little value

### 4. Check Signal Rejections

- Many rejections = consider increasing capital or position limits
- Or reduce the number of securities traded

### 5. Compare to Single Security

Run both modes and compare:
- Does portfolio mode improve risk-adjusted returns?
- Is the diversification benefit worth the complexity?

---

## Example: Tech Stock Portfolio

```python
from pathlib import Path
from Classes.Config.config import PortfolioConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.portfolio_engine import PortfolioEngine
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure for 5 tech stocks with max 3 positions
config = PortfolioConfig(
    initial_capital=100000.0,
    max_positions=3,
    position_size_limit=0.35,
    total_allocation_limit=0.95,
    commission=CommissionConfig(CommissionMode.PERCENTAGE, 0.001)
)

# Load tech stocks
loader = DataLoader(Path('raw_data'))
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
data_dict = {symbol: loader.load_csv(symbol) for symbol in tech_stocks}

# Run
strategy = AlphaTrendStrategy()
engine = PortfolioEngine(config)
results = engine.run(data_dict, strategy)

# Summary
print("=" * 50)
print("PORTFOLIO BACKTEST RESULTS")
print("=" * 50)

for symbol, result in results.symbol_results.items():
    print(f"{symbol}: {result.num_trades} trades, {result.total_return_pct:.1f}% return")

print("-" * 50)
print(f"Portfolio Return: {results.portfolio_return_pct:.1f}%")
print(f"Signal Rejections: {len(results.signal_rejections)}")
```

---

## Analyzing Portfolio Performance

### Generating Reports

```python
from Classes.Analysis.portfolio_report_generator import PortfolioReportGenerator

report_gen = PortfolioReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000.0
)
report_path = report_gen.generate_report(results)
```

### Key Metrics to Check

| Metric | What It Shows |
|--------|---------------|
| **Portfolio Sharpe** | Overall risk-adjusted return |
| **Max Drawdown** | Worst peak-to-trough decline |
| **Win Rate per Symbol** | Does strategy work everywhere? |
| **Capital Utilization** | How much capital was typically invested |
| **Rejection Rate** | % of signals rejected due to capital |

### Comparing Strategies

Run the same portfolio with different strategies:
```python
strategies = [
    AlphaTrendStrategy(),
    MACrossover(fast_period=20, slow_period=50),
]

for strategy in strategies:
    engine = PortfolioEngine(config)
    results = engine.run(data_dict, strategy)
    print(f"{strategy.get_name()}: {results.portfolio_return_pct:.1f}%")
```

---

## Troubleshooting

### No trades on some securities

**Cause:** Strategy conditions not met, or capital already allocated.

**Check:**
- Run single-security backtest to verify strategy works on that security
- Check signal rejections log
- Lower max_positions or position_size_limit

### Poor portfolio performance despite good individual results

**Cause:** Capital contention causing missed opportunities.

**Solutions:**
- Increase initial capital
- Reduce position_size_limit to fit more positions
- Use vulnerability scoring to swap weak positions

### High rejection rate

**Cause:** More signals than capital can support.

**Solutions:**
- Increase initial capital
- Reduce number of securities
- Lower position_size_limit
- Use vulnerability scoring

---

## Related Guides

- **[User Guide](USER_GUIDE.md)** - Getting started
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options
- **[Strategies Guide](STRATEGIES.md)** - Strategy configuration
- **[Optimization Guide](OPTIMIZATION.md)** - Parameter optimization
