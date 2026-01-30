# Backtesting Concepts

This document explains the core concepts behind backtesting in this framework.

---

## What Is Backtesting?

Backtesting is the process of testing a trading strategy against historical market data to see how it would have performed. It answers the question: "If I had traded this strategy in the past, what would my results have been?"

The framework simulates:
- When you would have entered trades (based on your strategy's signals)
- When you would have exited (via signal, stop loss, or time limit)
- How much profit or loss each trade would have generated
- How your capital would have grown or shrunk over time

---

## Execution Model

### Trade Timing

All trades in this framework execute at **closing prices**. When your strategy generates a BUY signal on a given day, the trade opens at that day's closing price.

```
Day N:
├── Market opens
├── Price moves throughout day
├── Strategy evaluates conditions using Day N data
├── Signal generated (if conditions met)
└── Trade executes at Day N close price
```

This model:
- Matches how TradingView and many retail platforms operate
- Provides realistic execution assumptions
- Avoids issues with intraday price movement

### No Lookahead Bias

The framework enforces strict temporal boundaries. When evaluating Day N:
- The strategy can see: Days 1 through N
- The strategy cannot see: Days N+1 and beyond

This prevents the common backtesting error of using future information to make past decisions.

```
                    ┌─────────────────────────────────────────┐
                    │              CURRENT BAR (Day N)        │
                    └─────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               │                               ▼
┌───────────────────┐                   │                  ┌───────────────────┐
│  VISIBLE DATA     │                   │                  │  INVISIBLE DATA   │
│                   │                   │                  │                   │
│  Days 1 to N      │                   │                  │  Days N+1 onward  │
│  (Historical)     │                   │                  │  (Future)         │
│                   │                   │                  │                   │
│  Strategy CAN     │                   │                  │  Strategy CANNOT  │
│  use this data    │                   │                  │  use this data    │
└───────────────────┘                   │                  └───────────────────┘
```

---

## Trade Lifecycle

A trade moves through distinct phases:

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          TRADE LIFECYCLE                                │
    └─────────────────────────────────────────────────────────────────────────┘

    1. SIGNAL GENERATION
       │
       │  Strategy analyzes current bar
       │  Conditions met → BUY signal generated
       │
       ▼
    2. ENTRY
       │
       │  Position opened at close price
       │  Quantity calculated based on capital and risk
       │  Stop loss set (if strategy provides one)
       │  Commission deducted
       │
       ▼
    3. POSITION MANAGEMENT
       │
       │  Daily monitoring:
       │  ├── Check if stop loss triggered
       │  ├── Check for exit signals from strategy
       │  ├── Check time-based exit conditions
       │  └── Optionally adjust stop loss
       │
       ▼
    4. EXIT
       │
       │  Position closed when:
       │  ├── Strategy generates SELL signal
       │  ├── Stop loss price reached
       │  ├── Maximum holding period reached
       │  └── (Portfolio mode) Capital needed elsewhere
       │
       ▼
    5. TRADE RECORD
       │
       │  Trade object created with:
       │  ├── Entry/exit dates and prices
       │  ├── Profit/loss calculation
       │  ├── Exit reason
       │  └── Duration
       │
       ▼
    6. NEXT OPPORTUNITY
       │
       │  Capital available for new trade
       │  Cycle repeats
```

---

## Position Sizing

The framework supports different approaches to determining how much to buy:

### Risk-Based Sizing

The most common approach. Position size is calculated to risk a fixed percentage of capital:

```
Position Size = (Capital × Risk %) / (Entry Price - Stop Loss)
```

Example:
- Capital: $100,000
- Risk per trade: 2%
- Entry price: $100
- Stop loss: $95

Position Size = ($100,000 × 0.02) / ($100 - $95) = $2,000 / $5 = 400 shares

### Fixed Percentage Sizing

Allocate a fixed percentage of capital to each position:

```
Position Value = Capital × Allocation %
Shares = Position Value / Entry Price
```

### Full Capital

Use all available capital for each trade (typical for single security mode).

---

## Commission Handling

The framework accounts for trading costs in two modes:

### Percentage Commission

A percentage of the trade value is deducted:

```
Commission = Trade Value × Commission Rate

Example:
Trade Value = $10,000
Commission Rate = 0.1% (0.001)
Commission = $10,000 × 0.001 = $10
```

### Fixed Commission

A fixed dollar amount per trade:

```
Commission = Fixed Amount

Example:
Fixed Amount = $5
Commission = $5 per trade
```

Commissions are deducted on both entry and exit, affecting the net profit/loss.

---

## Single Security Mode

In single security mode, each security operates in isolation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SINGLE SECURITY MODE                                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         AAPL Backtest                               │  │
│   │                                                                     │  │
│   │   Capital: $100,000 (isolated)                                     │  │
│   │   Positions: 0 or 1 at a time                                      │  │
│   │   No interaction with other securities                             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         MSFT Backtest                               │  │
│   │                                                                     │  │
│   │   Capital: $100,000 (isolated)                                     │  │
│   │   Positions: 0 or 1 at a time                                      │  │
│   │   No interaction with other securities                             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Results: Separate metrics for each security                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Use Cases:**
- Testing strategy performance on specific securities
- Comparing performance across different securities
- Initial strategy development and debugging

---

## Portfolio Mode

Portfolio mode introduces shared capital across multiple securities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PORTFOLIO MODE                                    │
│                                                                             │
│                      ┌─────────────────────────┐                           │
│                      │    Shared Capital Pool  │                           │
│                      │       $100,000          │                           │
│                      └───────────┬─────────────┘                           │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                     │
│              │                   │                   │                     │
│              ▼                   ▼                   ▼                     │
│      ┌───────────────┐   ┌───────────────┐   ┌───────────────┐           │
│      │     AAPL      │   │     MSFT      │   │     GOOGL     │           │
│      │               │   │               │   │               │           │
│      │  Can request  │   │  Can request  │   │  Can request  │           │
│      │  capital when │   │  capital when │   │  capital when │           │
│      │  signal fires │   │  signal fires │   │  signal fires │           │
│      └───────────────┘   └───────────────┘   └───────────────┘           │
│                                                                             │
│   Constraints:                                                              │
│   • max_positions: Maximum simultaneous positions (e.g., 3)                │
│   • position_size_limit: Max % per position (e.g., 40%)                    │
│   • total_allocation_limit: Max % total allocated (e.g., 95%)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Differences from Single Security:**
- Securities compete for limited capital
- Position limits constrain simultaneous holdings
- Capital contention requires resolution (see [Vulnerability Scoring](VULNERABILITY_SCORING.md))

See [Portfolio Mode](PORTFOLIO_MODE.md) for detailed documentation.

---

## Results and Metrics

After a backtest completes, the framework calculates extensive performance metrics:

### Return Metrics
- Total Return ($ and %)
- Annualized Return
- Risk-Free Adjusted Return

### Risk Metrics
- Maximum Drawdown ($ and %)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Value at Risk (VaR)

### Trade Metrics
- Total Trades
- Win Rate
- Average Win / Average Loss
- Profit Factor
- Expectancy
- Average Trade Duration

### Consistency Metrics
- Monthly Win Rate
- Best/Worst Month
- Recovery Factor

See [Metrics Glossary](../reference/METRICS_GLOSSARY.md) for complete definitions.

---

## Output Files

Each backtest generates:

| File | Contents |
|------|----------|
| `{strategy}_{symbol}_trades.csv` | Complete trade log with all details |
| `{strategy}_{symbol}_parameters.json` | Parameters used for the backtest |
| `{strategy}_{symbol}_report.xlsx` | Excel report with charts and metrics |

Output location: `logs/backtests/single_security/{backtest_name}/` or `logs/backtests/portfolio/{backtest_name}/`

---

## Best Practices

### Data Quality

- Ensure data covers your intended test period
- Verify required columns exist (strategy-dependent)
- Check for gaps or anomalies in price data

### Realistic Expectations

- Include commission costs (they add up)
- Test across different market conditions
- Don't over-optimize on historical data

### Validation

- Compare results with known benchmarks
- Test the same strategy on different securities
- Use out-of-sample testing via walk-forward optimization

---

## Related Documentation

- [Strategies](STRATEGIES.md) — How strategies work
- [Portfolio Mode](PORTFOLIO_MODE.md) — Multi-security backtesting
- [Optimization](OPTIMIZATION.md) — Finding optimal parameters
- [Metrics Glossary](../reference/METRICS_GLOSSARY.md) — All metrics explained
