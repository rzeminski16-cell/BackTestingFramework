# Factor Analysis

Factor analysis identifies what conditions lead to successful versus unsuccessful trades, enabling data-driven strategy improvement.

---

## What Is Factor Analysis?

Factor analysis examines the relationship between various market conditions (factors) and trade outcomes. It answers questions like:
- "Do my trades perform better when RSI is low?"
- "Does insider buying activity predict trade success?"
- "What market conditions should I avoid trading in?"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FACTOR ANALYSIS CONCEPT                                 │
│                                                                             │
│                         FACTORS                      OUTCOME                │
│   ┌─────────────────────────────────────┐       ┌──────────────────┐       │
│   │  Technical:                         │       │                  │       │
│   │  • RSI at entry                     │       │   Trade Result   │       │
│   │  • Momentum                         │       │                  │       │
│   │  • Volatility                       │   →   │   Good (+2%+)    │       │
│   │                                     │       │   Bad (-1%-)     │       │
│   │  Fundamental:                       │       │   Indeterminate  │       │
│   │  • P/E ratio                        │       │                  │       │
│   │  • Earnings growth                  │       └──────────────────┘       │
│   │                                     │                                   │
│   │  Insider:                           │                                   │
│   │  • Recent insider buys              │       ┌──────────────────┐       │
│   │                                     │       │   INSIGHT:       │       │
│   │  Market Regime:                     │   →   │   Low RSI trades │       │
│   │  • Trend direction                  │       │   have 2.1x      │       │
│   │  • Volatility regime                │       │   better odds    │       │
│   │                                     │       └──────────────────┘       │
│   └─────────────────────────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Trade Classification

Before analysis, trades are classified into outcome categories:

| Classification | Threshold | Description |
|----------------|-----------|-------------|
| **Good** | P/L ≥ +2.0% | Profitable trades worth emulating |
| **Bad** | P/L ≤ -1.0% | Losing trades to avoid |
| **Indeterminate** | Between | Neutral outcomes, excluded from analysis |

Thresholds are configurable based on your strategy's characteristics.

```
Trade Distribution Example:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│       BAD              INDETERMINATE              GOOD                     │
│   ◄───────────►    ◄─────────────────────►    ◄───────────►                │
│                                                                             │
│       32%                    23%                    45%                     │
│      (48 trades)           (35 trades)           (67 trades)               │
│                                                                             │
│   ├─────────────┼─────────────┼─────────────────────────────┤              │
│  -10%          -1%            0%           +2%            +10%  (P/L %)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Factor Categories

### Technical Factors

Derived from price and volume data:

| Factor | Description | Source |
|--------|-------------|--------|
| RSI | Relative Strength Index at entry | Price data |
| MACD | Momentum indicator | Price data |
| Bollinger %B | Position within Bollinger Bands | Price data |
| Volume Ratio | Volume vs average | Volume data |
| Momentum | Rate of price change | Price data |
| Volatility | Recent price variation | Price data |

### Fundamental Factors

From financial statements and ratios:

| Factor | Description | Source |
|--------|-------------|--------|
| P/E Ratio | Price to earnings | Fundamentals |
| P/B Ratio | Price to book | Fundamentals |
| ROE | Return on equity | Fundamentals |
| Debt/Equity | Leverage ratio | Fundamentals |
| Revenue Growth | Sales growth rate | Fundamentals |
| EPS Surprise | Actual vs expected | Fundamentals |

### Insider Factors

From insider transaction data:

| Factor | Description | Source |
|--------|-------------|--------|
| Buy Count | Recent insider purchases | Insider data |
| Sell Count | Recent insider sales | Insider data |
| Net Shares | Net insider activity | Insider data |
| Sentiment | Computed insider sentiment | Insider data |

### Market Regime Factors

Market condition classifications:

| Factor | Description | Calculation |
|--------|-------------|-------------|
| Trend Regime | Uptrend/Downtrend/Sideways | Price vs moving averages |
| Volatility Regime | High/Normal/Low | ATR percentile |
| Momentum Regime | Strong/Weak | Rate of change |

---

## Three-Tier Analysis

The framework uses a progressive analysis approach:

### Tier 1: Exploratory Analysis

Quick scan to identify potential relationships:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIER 1: EXPLORATION                                 │
│                                                                             │
│   Point-Biserial Correlations with Trade Outcome:                          │
│                                                                             │
│   Factor              Correlation    p-value    Significant?               │
│   ──────────────────────────────────────────────────────────               │
│   RSI_14                -0.23         0.001        Yes ★                  │
│   Volume_Ratio          +0.18         0.008        Yes ★                  │
│   P/E_Ratio             -0.12         0.089        No                     │
│   Insider_Buys          +0.15         0.032        Yes ★                  │
│   Momentum_5d           +0.21         0.003        Yes ★                  │
│   ...                                                                       │
│                                                                             │
│   ★ Factors to investigate further in Tier 2                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 2: Hypothesis Testing

Statistical tests on significant factors:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TIER 2: HYPOTHESIS TESTING                            │
│                                                                             │
│   Logistic Regression Results:                                             │
│                                                                             │
│   Factor              Coefficient    Odds Ratio    p-value                 │
│   ──────────────────────────────────────────────────────────               │
│   RSI_14                 -0.034         0.97        0.002                  │
│   Volume_Ratio           +0.45          1.57        0.012                  │
│   Insider_Buys           +0.28          1.32        0.041                  │
│   Momentum_5d            +0.12          1.13        0.008                  │
│                                                                             │
│   Interpretation:                                                           │
│   • Each 1-point increase in RSI decreases odds of good trade by 3%       │
│   • High volume increases odds of good trade by 57%                       │
│   • Recent insider buying increases odds by 32%                           │
│                                                                             │
│   ANOVA Results (Categorical Factors):                                     │
│   • Trend regime significantly affects outcomes (F=12.3, p<0.001)         │
│   • Volatility regime effect is marginal (F=3.2, p=0.043)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 3: Machine Learning Analysis

Advanced feature importance using ML models:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TIER 3: ML FEATURE IMPORTANCE                          │
│                                                                             │
│   Random Forest Feature Importance:                                        │
│                                                                             │
│   RSI_14          ████████████████████████  0.18                          │
│   Momentum_5d     ██████████████████████    0.16                          │
│   Volume_Ratio    █████████████████         0.13                          │
│   Volatility      ████████████████          0.12                          │
│   Insider_Buys    ████████████              0.09                          │
│   P/E_Ratio       █████████                 0.07                          │
│   ...                                                                       │
│                                                                             │
│   SHAP Analysis (Feature Contribution):                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   Low RSI values push prediction toward "Good Trade"               │  │
│   │   High RSI values push prediction toward "Bad Trade"               │  │
│   │                                                                     │  │
│   │   High Volume Ratio → Strong positive impact                       │  │
│   │   Low Volume Ratio → Slight negative impact                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Scenario Detection

Identifies specific conditions that produce best/worst outcomes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCENARIO DETECTION                                   │
│                                                                             │
│   BEST SCENARIOS (highest win rate):                                       │
│                                                                             │
│   Scenario 1: "Low RSI + High Volume"                                      │
│   ├── Conditions: RSI < 35 AND Volume_Ratio > 1.5                          │
│   ├── Trades matching: 28                                                   │
│   ├── Win Rate: 71% (vs 45% baseline)                                      │
│   └── Lift: 1.58x better than average                                      │
│                                                                             │
│   Scenario 2: "Insider Buying + Uptrend"                                   │
│   ├── Conditions: Insider_Buys > 2 AND Trend = Uptrend                     │
│   ├── Trades matching: 19                                                   │
│   ├── Win Rate: 68% (vs 45% baseline)                                      │
│   └── Lift: 1.51x better than average                                      │
│                                                                             │
│   WORST SCENARIOS (highest loss rate):                                     │
│                                                                             │
│   Scenario 1: "High RSI + Low Volume"                                      │
│   ├── Conditions: RSI > 70 AND Volume_Ratio < 0.8                          │
│   ├── Trades matching: 15                                                   │
│   ├── Loss Rate: 67% (vs 32% baseline)                                     │
│   └── Risk: 2.1x worse than average                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Temporal Alignment

A critical feature preventing lookahead bias:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPORAL ALIGNMENT                                     │
│                                                                             │
│   Trade Entry: March 15, 2023                                              │
│                                                                             │
│   WRONG (Lookahead Bias):                                                  │
│   ├── Use Q1 2023 earnings (reported April 20)  ← Future data!            │
│   ├── Use March 20 insider filing              ← Future data!             │
│   └── Results would be unrealistically good                               │
│                                                                             │
│   CORRECT (As-Of Data):                                                    │
│   ├── Use Q4 2022 earnings (reported Jan 15)   ← Available on March 15    │
│   ├── Use March 10 insider filing              ← Available on March 15    │
│   └── Results reflect realistic trading                                   │
│                                                                             │
│   Configuration:                                                           │
│   ├── fundamental_delay: 0 days (use report_date)                         │
│   ├── insider_delay: 3 days (filing → public availability)               │
│   └── options_snapshot: as-of trade date                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Using Factor Analysis

### Step 1: Prepare Data

Ensure you have:
- Trade logs from backtests (`logs/backtests/*/trades.csv`)
- Daily price data (`raw_data/daily/`)
- Optional: Fundamental data (`raw_data/fundamentals/`)
- Optional: Insider data (`raw_data/insider_transactions/`)

### Step 2: Launch the GUI

```bash
python ctk_factor_analysis_gui.py
```

### Step 3: Configure Analysis

Use the Configuration Manager to set:
- Trade classification thresholds
- Factors to include
- Statistical test parameters
- Scenario detection options

### Step 4: Upload Data

Load your trade logs and supplementary data files.

### Step 5: Run Analysis

Execute the three-tier analysis pipeline.

### Step 6: Interpret Results

Review each tier's findings:
- Tier 1: Which factors correlate with outcomes?
- Tier 2: Are relationships statistically significant?
- Tier 3: What's the predictive importance?
- Scenarios: What conditions should I target/avoid?

### Step 7: Apply Insights

```
EXAMPLE: Applying Factor Analysis Findings

Finding: "RSI < 35 entries have 1.6x better odds"

Actions:
├── Add RSI filter to entry conditions
│   └── Only enter when RSI < 35
│
├── Or: Increase position size when RSI < 35
│   └── Risk 3% instead of 2%
│
└── Re-test strategy to verify improvement
```

---

## Output Reports

### Excel Report

Multi-sheet workbook containing:

| Sheet | Contents |
|-------|----------|
| Summary | Key findings, trade counts, top factors |
| Factor Statistics | Descriptive stats for all factors |
| Correlations | Point-biserial correlations |
| Hypothesis Tests | Regression and ANOVA results |
| ML Analysis | Feature importance, SHAP values |
| Scenarios | Best/worst conditions |

### JSON Payload

Machine-readable format for integration:

```
{
  "summary": {
    "trade_counts": {...},
    "top_factors": [...],
    "best_scenarios": [...]
  },
  "factor_analysis": {
    "tier1": {...},
    "tier2": {...},
    "tier3": {...}
  },
  "scenarios": {...}
}
```

---

## Best Practices

### Sample Size

Aim for sufficient trades:
- Minimum: 50 trades (basic analysis)
- Recommended: 100+ trades (robust statistics)
- Ideal: 200+ trades (scenario detection)

### Factor Selection

Don't test everything:
- Start with factors you believe matter
- Add factors based on domain knowledge
- Avoid "fishing" through hundreds of factors

### Multiple Testing Correction

With many factors, some will appear significant by chance:
- Framework applies FDR (False Discovery Rate) correction
- Focus on factors significant after correction
- Be skeptical of marginal results

### Validation

Don't overfit to analysis results:
- Apply insights to out-of-sample data
- Use walk-forward validation
- Start with conservative application

---

## Related Documentation

- [Applications: Factor Analysis GUI](../applications/FACTOR_ANALYSIS_GUI.md) — Using the interface
- [Edge Analysis](EDGE_ANALYSIS.md) — Complementary entry validation
- [Strategies](STRATEGIES.md) — Applying insights to strategies
- [Configuration: Factor Analysis](../reference/CONFIGURATION.md#factor-analysis) — All settings
