# BackTesting Framework

## What Is This Framework?

The BackTesting Framework is a comprehensive system for developing, testing, and refining trading strategies using historical market data. It provides the tools needed to validate trading ideas before risking real capital.

At its core, the framework simulates how a trading strategy would have performed historically, accounting for realistic factors like commission costs, position sizing, and capital constraints.

---

## Who Is It For?

This framework is designed for:

- **Traders** who want to test trading ideas systematically before live deployment
- **Analysts** who need to evaluate strategy performance across different market conditions
- **Researchers** exploring the statistical properties of trading signals
- **Developers** building custom trading strategies with robust testing infrastructure

No programming experience is required for basic usage—the graphical interface provides access to all major features. For custom strategy development, Python knowledge is helpful.

---

## Key Capabilities

### Strategy Testing

Test trading strategies against years of historical data to understand how they would have performed:

| Capability | Description |
|------------|-------------|
| **Single Security Testing** | Test a strategy on one security with isolated capital |
| **Portfolio Testing** | Test across multiple securities with shared capital pool |
| **Realistic Execution** | Trades execute at closing prices matching real-world constraints |
| **Commission Modeling** | Account for percentage or fixed commission costs |

### Parameter Optimization

Find optimal strategy parameters systematically rather than guessing:

| Approach | Best For |
|----------|----------|
| **Univariate Optimization** | Understanding individual parameter impact |
| **Walk-Forward Optimization** | Finding robust parameters that avoid overfitting |
| **Bayesian Search** | Efficient exploration of large parameter spaces |

### Performance Analysis

Understand why a strategy succeeds or fails:

| Analysis Type | Purpose |
|---------------|---------|
| **Edge Analysis** | Validate entry quality with E-ratio calculations |
| **Factor Analysis** | Identify what conditions lead to good vs bad trades |
| **Vulnerability Scoring** | Optimize capital allocation during signal conflicts |

### Data Collection

Gather the market data needed for backtesting:

- Daily and weekly price data (OHLCV)
- Technical indicators (50+ pre-calculated)
- Fundamental data (financial statements, ratios)
- Insider transaction records
- Options chain data

---

## Framework Philosophy

### No Lookahead Bias

Strategies only see data that would have been available at each historical point. The framework enforces strict temporal boundaries to prevent unrealistic results.

### Execution Realism

Trades execute at closing prices, matching how most retail traders operate. This alignment with TradingView and other platforms makes results directly comparable.

### Long-Only Focus

The framework currently supports long-only trading. This simplifies position management while covering the most common retail trading approach.

### Iterative Refinement

The workflow encourages iteration:

```
    Strategy Idea
         │
         ▼
    ┌─────────┐
    │Backtest │◄────────────────────┐
    └────┬────┘                     │
         │                          │
         ▼                          │
    ┌──────────┐                    │
    │ Optimize │                    │
    └────┬─────┘                    │
         │                          │
         ▼                          │
    ┌──────────┐                    │
    │ Analyze  │────► Insights ─────┘
    └──────────┘         │
                         │
                    Good enough?
                         │
                    Yes  │
                         ▼
                    Deploy
```

---

## Getting Started

The recommended path through the framework:

1. **Collect Data** — Gather historical price data for your target securities
2. **Run Initial Backtest** — Test the included AlphaTrend strategy or your own
3. **Optimize Parameters** — Find parameter values that improve performance
4. **Analyze Results** — Use edge analysis and factor analysis to understand performance
5. **Iterate** — Refine strategy based on insights and re-test

See [QUICK_START.md](QUICK_START.md) for step-by-step instructions to run your first backtest.

---

## Documentation Structure

| Section | Contents |
|---------|----------|
| **Overview** | Introduction, architecture, quick start |
| **Concepts** | Core concepts explained (backtesting, strategies, optimization, analysis) |
| **Applications** | How to use each GUI application |
| **Strategy Development** | Creating custom trading strategies |
| **Alternative System** | The newer `backtesting/` module |
| **Reference** | Configuration files, metrics glossary, securities |

---

## Where to Find Code

The framework is organized into distinct areas:

| Location | Purpose |
|----------|---------|
| `Classes/` | Core framework components (engines, strategies, analysis) |
| `strategies/` | Production-ready trading strategies |
| `backtesting/` | Alternative modular backtesting system |
| `config/` | Configuration files (presets, parameters, baskets) |
| `raw_data/` | Historical market data storage |
| `logs/` | Backtest results and reports |

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).
