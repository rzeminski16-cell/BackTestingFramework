# Portfolio Mode

Portfolio mode enables backtesting across multiple securities with shared capital, simulating how a real trading account operates.

---

## Overview

In portfolio mode:
- Multiple securities share a single capital pool
- Positions are limited by configurable constraints
- The framework manages capital allocation and conflicts
- Results are aggregated across the entire portfolio

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PORTFOLIO MODE                                    │
│                                                                             │
│                         Total Capital: $100,000                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        ALLOCATION POOL                              │  │
│   │                                                                     │  │
│   │   Available: $60,000     Allocated: $40,000                        │  │
│   │                                                                     │  │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐                           │  │
│   │   │  AAPL   │  │  MSFT   │  │  Empty  │  max_positions: 3         │  │
│   │   │ $20,000 │  │ $20,000 │  │         │                           │  │
│   │   └─────────┘  └─────────┘  └─────────┘                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Constraints:                                                              │
│   • max_positions = 3                                                       │
│   • position_size_limit = 40% ($40,000 max per position)                   │
│   • total_allocation_limit = 95% ($95,000 max total)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use Portfolio Mode

| Use Case | Single Security | Portfolio Mode |
|----------|-----------------|----------------|
| Test strategy on one ticker | Best choice | Overkill |
| Test strategy across many tickers | Run separately | Best choice |
| Simulate real account behavior | Not realistic | Realistic |
| Study capital allocation effects | N/A | Required |
| Test position limit impact | N/A | Required |

---

## Configuration Options

### Capital Settings

| Setting | Description | Example |
|---------|-------------|---------|
| `initial_capital` | Starting portfolio value | 100,000 |
| `position_size_limit` | Maximum % of capital per position | 0.40 (40%) |
| `total_allocation_limit` | Maximum % of capital allocated | 0.95 (95%) |

### Position Constraints

| Setting | Description | Example |
|---------|-------------|---------|
| `max_positions` | Maximum simultaneous positions | 3 |
| `min_position_size` | Minimum position value | 1,000 |

### Commission

| Setting | Description | Example |
|---------|-------------|---------|
| `commission_mode` | "percentage" or "fixed" | "percentage" |
| `commission_value` | Rate or amount | 0.001 (0.1%) |

---

## Daily Processing Flow

Each trading day, the portfolio engine processes all securities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAILY PROCESSING                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    START OF DAY
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 1. UPDATE EXISTING POSITIONS                                           │
    │                                                                        │
    │    For each open position:                                            │
    │    • Check stop loss triggers                                         │
    │    • Process exit signals from strategy                               │
    │    • Update position value with current price                         │
    │    • Close positions that meet exit criteria                          │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 2. COLLECT NEW SIGNALS                                                 │
    │                                                                        │
    │    For each security without a position:                              │
    │    • Generate strategy signal                                         │
    │    • If BUY signal, add to pending queue                              │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 3. CHECK CAPITAL AVAILABILITY                                          │
    │                                                                        │
    │    Calculate:                                                          │
    │    • available_capital = cash + closed_position_proceeds              │
    │    • current_positions = count of open positions                      │
    │    • can_open_new = (current_positions < max_positions)               │
    │                     AND (available_capital > min_position_size)       │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
         │
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 4. RESOLVE CAPITAL CONTENTION (if needed)                              │
    │                                                                        │
    │    If more signals than available slots:                              │
    │                                                                        │
    │    DEFAULT mode:                                                       │
    │    • First-come-first-served                                          │
    │    • Excess signals skipped                                           │
    │                                                                        │
    │    VULNERABILITY_SCORE mode:                                           │
    │    • Score all positions and pending signals                          │
    │    • Close weakest positions if new signals score better              │
    │    • Open highest-scoring signals                                     │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 5. EXECUTE SIGNALS                                                     │
    │                                                                        │
    │    For each pending signal (until capital/slots exhausted):           │
    │    • Calculate position size                                          │
    │    • Verify against position_size_limit                               │
    │    • Open position at close price                                     │
    │    • Deduct capital and commission                                    │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 6. UPDATE PORTFOLIO VALUE                                              │
    │                                                                        │
    │    portfolio_value = cash + sum(position_market_values)               │
    │    equity_curve.append(portfolio_value)                               │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    END OF DAY → NEXT DAY
```

---

## Capital Contention

Capital contention occurs when multiple securities signal simultaneously but insufficient capital or position slots exist.

### Scenario Example

```
Situation:
├── max_positions: 3
├── Current positions: 2 (AAPL, MSFT)
├── Available slots: 1
└── New BUY signals: 3 (GOOGL, AMZN, NVDA)

Problem: 3 signals but only 1 slot available
```

### Resolution Modes

#### DEFAULT Mode

Simple first-come-first-served:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEFAULT MODE                                       │
│                                                                             │
│   Signals received: [GOOGL, AMZN, NVDA]                                    │
│                                                                             │
│   Processing order: As encountered in data                                 │
│                                                                             │
│   Result:                                                                   │
│   ├── GOOGL: Executed (filled available slot)                              │
│   ├── AMZN: Skipped (no slots)                                             │
│   └── NVDA: Skipped (no slots)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### VULNERABILITY_SCORE Mode

Score-based intelligent allocation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VULNERABILITY_SCORE MODE                               │
│                                                                             │
│   Step 1: Score existing positions                                         │
│   ├── AAPL: Score 0.65 (profitable, 3 days old)                           │
│   └── MSFT: Score 0.35 (small loss, 7 days old)                           │
│                                                                             │
│   Step 2: Score pending signals                                            │
│   ├── GOOGL: Score 0.70 (strong entry setup)                              │
│   ├── AMZN: Score 0.55 (moderate setup)                                    │
│   └── NVDA: Score 0.45 (weak setup)                                        │
│                                                                             │
│   Step 3: Compare and swap if beneficial                                   │
│   ├── GOOGL (0.70) > MSFT (0.35) → Close MSFT, Open GOOGL                 │
│   ├── AMZN (0.55) > next weakest? → No swap opportunity                   │
│   └── NVDA (0.45) → Skipped                                                │
│                                                                             │
│   Final positions: [AAPL, GOOGL]                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

See [Vulnerability Scoring](VULNERABILITY_SCORING.md) for detailed scoring mechanics.

---

## Position Sizing in Portfolio Mode

### Per-Position Limits

Each position is constrained by `position_size_limit`:

```
Requested size: $50,000
position_size_limit: 40%
Total capital: $100,000
Maximum allowed: $40,000

Result: Position capped at $40,000
```

### Total Allocation Limit

Combined positions cannot exceed `total_allocation_limit`:

```
Current allocations:
├── AAPL: $30,000
├── MSFT: $30,000
└── Total: $60,000

total_allocation_limit: 95%
Total capital: $100,000
Maximum total: $95,000
Remaining capacity: $35,000

New signal for GOOGL requests $40,000
Result: Capped at $35,000
```

### Cash Reserve

The difference between 100% and `total_allocation_limit` acts as a cash reserve:

```
total_allocation_limit: 95%
Cash reserve: 5% ($5,000 on $100,000 capital)

Purpose:
• Buffer for adverse price moves
• Available for emergency needs
• Reduces margin risk
```

---

## Security Baskets

Baskets define groups of securities for portfolio testing:

### Pre-Configured Baskets

Located in `config/baskets/`:

| Basket | Description | Count |
|--------|-------------|-------|
| `All_Stocks` | Full universe of available stocks | 165+ |
| `Large_Cap_15` | Top 15 large-cap stocks | 15 |
| `Balanced_Large_Cap` | Diversified large-cap selection | 20 |
| `Tech_Focused` | Technology sector focus | 25 |

### Basket Configuration

```
config/baskets/Large_Cap_15.json
┌─────────────────────────────────────────────────────────────────────────────┐
│  {                                                                          │
│    "name": "Large_Cap_15",                                                 │
│    "description": "Top 15 large-cap stocks by market cap",                │
│    "symbols": [                                                            │
│      "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",                             │
│      "META", "TSLA", "BRK.B", "UNH", "JNJ",                               │
│      "V", "JPM", "XOM", "PG", "HD"                                        │
│    ]                                                                        │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Portfolio Results

### Per-Symbol Results

Each security generates individual metrics:

```
Per-Symbol Breakdown:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Symbol  │  Trades  │  Win Rate  │  Return %  │  Sharpe  │  Contribution  │
├──────────┼──────────┼────────────┼────────────┼──────────┼────────────────┤
│  AAPL    │    12    │   58.3%    │   +15.2%   │   1.45   │    +$4,560    │
│  MSFT    │     8    │   62.5%    │   +12.8%   │   1.32   │    +$3,840    │
│  GOOGL   │    10    │   40.0%    │   -2.3%    │  -0.21   │     -$690     │
│  ...     │   ...    │    ...     │    ...     │   ...    │      ...      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Aggregated Portfolio Metrics

Combined performance across all positions:

```
Portfolio Summary:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Total Return:        +18.5% ($18,500)                                     │
│  Annualized Return:   +24.2%                                                │
│  Sharpe Ratio:        1.28                                                  │
│  Max Drawdown:        -8.3% ($8,300)                                        │
│                                                                             │
│  Total Trades:        85                                                    │
│  Win Rate:            52.9%                                                 │
│  Profit Factor:       1.65                                                  │
│                                                                             │
│  Capital Utilization: 78.4% average                                        │
│  Contention Events:   23 (signals skipped due to limits)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Output Files

Portfolio mode generates:

| File | Contents |
|------|----------|
| `portfolio_trades.csv` | All trades across all symbols |
| `portfolio_summary.json` | Aggregated metrics |
| `portfolio_report.xlsx` | Comprehensive Excel report |
| `per_symbol/` | Individual symbol results |

---

## Best Practices

### Choosing Position Limits

```
Conservative:
├── max_positions: 3-5
├── position_size_limit: 25-30%
└── Benefit: Lower risk, easier to manage

Moderate:
├── max_positions: 5-10
├── position_size_limit: 15-20%
└── Benefit: Balance of diversification and concentration

Aggressive:
├── max_positions: 10+
├── position_size_limit: 10-15%
└── Benefit: Maximum diversification
```

### Capital Allocation Strategy

Consider your strategy's characteristics:

| Strategy Type | Recommended Settings |
|--------------|---------------------|
| High win rate, low R | More positions, smaller size |
| Low win rate, high R | Fewer positions, larger size |
| Trend following | Moderate positions, trailing allocation |
| Mean reversion | More positions, quick turnover |

### Testing Capital Contention

Run backtests with different contention modes to understand impact:

1. Run with DEFAULT mode
2. Run with VULNERABILITY_SCORE mode
3. Compare:
   - Total returns
   - Drawdowns
   - Trade counts
   - Missed opportunities

---

## Related Documentation

- [Backtesting Concepts](BACKTESTING.md) — Core backtesting fundamentals
- [Vulnerability Scoring](VULNERABILITY_SCORING.md) — Position scoring for contention
- [Configuration Reference](../reference/CONFIGURATION.md) — All configuration options
- [Applications: Backtest GUI](../applications/BACKTEST_GUI.md) — Using the portfolio backtest interface
