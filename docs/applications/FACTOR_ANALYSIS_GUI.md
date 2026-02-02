# Factor Analysis GUI

The Factor Analysis GUI provides an interface for analyzing what conditions lead to successful versus unsuccessful trades.

---

## Launching

```bash
# Full dashboard
python ctk_factor_analysis_gui.py

# Configuration manager only
python ctk_factor_analysis_gui.py --config

# Data upload interface only
python ctk_factor_analysis_gui.py --upload
```

---

## Dashboard Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FACTOR ANALYSIS DASHBOARD                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Summary] [Tier 1] [Tier 2] [Tier 3] [Scenarios] [Export]      ◄── Tabs    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │    SIDEBAR              │  │              MAIN PANEL                 │  │
│  │                         │  │                                         │  │
│  │  Data Files:            │  │  [Content changes based on tab]        │  │
│  │  ☑ trades.csv           │  │                                         │  │
│  │  ☑ AAPL_daily.csv       │  │                                         │  │
│  │  ☑ fundamentals/        │  │                                         │  │
│  │                         │  │                                         │  │
│  │  Configuration:         │  │                                         │  │
│  │  [Settings ▼]           │  │                                         │  │
│  │                         │  │                                         │  │
│  │  [Run Analysis]         │  │                                         │  │
│  │                         │  │                                         │  │
│  └─────────────────────────┘  └─────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Loading Data

### Required Files

| File Type | Location | Purpose |
|-----------|----------|---------|
| Trade log | `logs/backtests/*/trades.csv` | Trades to analyze |
| Price data | `raw_data/daily/*.csv` | Technical factors |

### Optional Files

| File Type | Location | Purpose |
|-----------|----------|---------|
| Fundamentals | `raw_data/fundamentals/*/` | Value/quality factors |
| Insider data | `raw_data/insider_transactions/*.csv` | Insider factors |
| Options data | `raw_data/options/*/` | Options factors |

### Data Upload Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA UPLOAD                                           │
│                                                                             │
│  Trade Logs                                                                │
│  ──────────                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │     Drag and drop trade log files here                             │   │
│  │                    or                                              │   │
│  │              [Browse Files]                                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Loaded: portfolio_trades.csv (99 trades)                                  │
│                                                                             │
│  Price Data Directory                                                      │
│  ────────────────────                                                      │
│  Path: [raw_data/daily/                 ] [Browse]                         │
│  Status: ✓ 165 securities found                                            │
│                                                                             │
│  Supplementary Data (Optional)                                             │
│  ─────────────────────────────                                             │
│  Fundamentals: [raw_data/fundamentals/  ] [Browse]  Status: ✓ 120 found   │
│  Insider Data: [raw_data/insider/       ] [Browse]  Status: ✓ 85 found    │
│  Options Data: [raw_data/options/       ] [Browse]  Status: ○ Not loaded  │
│                                                                             │
│  [Validate Data]  [Continue to Analysis]                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Manager

### Trade Classification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   TRADE CLASSIFICATION                                      │
│                                                                             │
│  Good Trade Threshold:     [  2.0 ] %    (P/L ≥ this = Good)              │
│  Bad Trade Threshold:      [ -1.0 ] %    (P/L ≤ this = Bad)               │
│                                                                             │
│  Preview:                                                                  │
│  ────────                                                                  │
│  99 total trades                                                           │
│  ├── Good:          45 (45.5%)                                            │
│  ├── Bad:           32 (32.3%)                                            │
│  └── Indeterminate: 22 (22.2%)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Factor Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FACTOR CATEGORIES                                         │
│                                                                             │
│  ☑ Technical Factors                                                       │
│     ☑ RSI_14                                                               │
│     ☑ MACD                                                                 │
│     ☑ Bollinger_%B                                                         │
│     ☑ Momentum_5d                                                          │
│     ☑ Volume_Ratio                                                         │
│     ☐ [Add custom...]                                                      │
│                                                                             │
│  ☑ Fundamental Factors                                                     │
│     ☑ P/E Ratio                                                            │
│     ☑ P/B Ratio                                                            │
│     ☑ ROE                                                                  │
│     ☐ Debt/Equity                                                          │
│                                                                             │
│  ☑ Insider Factors                                                         │
│     ☑ Buy_Count_30d                                                        │
│     ☑ Sell_Count_30d                                                       │
│     ☑ Net_Sentiment                                                        │
│                                                                             │
│  ☑ Market Regime Factors                                                   │
│     ☑ Trend_Regime                                                         │
│     ☑ Volatility_Regime                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Temporal Alignment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   TEMPORAL ALIGNMENT                                        │
│                                                                             │
│  Prevent look-ahead bias by using only data available at entry:            │
│                                                                             │
│  Fundamental Data Delay:   [ 0 ] days after report_date                    │
│  Insider Data Delay:       [ 3 ] days after filing_date                    │
│  Options Snapshot:         ● As-of entry date  ○ Previous close            │
│                                                                             │
│  ⚠ Important: Delays account for data availability in real trading        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Analysis Tabs

### Summary Tab

Overview of key findings:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS SUMMARY                                     │
│                                                                             │
│  DATA OVERVIEW                                                             │
│  ─────────────                                                             │
│  Total Trades: 99    Good: 45    Bad: 32    Indeterminate: 22             │
│  Factors Analyzed: 18                                                      │
│  Analysis Duration: 12.3 seconds                                           │
│                                                                             │
│  TOP PREDICTIVE FACTORS                                                    │
│  ─────────────────────                                                     │
│  1. RSI_14           r=-0.28    p=0.001    Strong negative correlation    │
│  2. Volume_Ratio     r=+0.22    p=0.008    Positive correlation           │
│  3. Insider_Buys     r=+0.18    p=0.024    Positive correlation           │
│  4. Momentum_5d      r=+0.16    p=0.041    Positive correlation           │
│                                                                             │
│  BEST TRADING CONDITIONS                                                   │
│  ───────────────────────                                                   │
│  • RSI < 35 AND Volume > 1.5x average (Win rate: 72%, Lift: 1.58x)        │
│  • Insider buying in past 30 days (Win rate: 65%, Lift: 1.43x)            │
│                                                                             │
│  WORST TRADING CONDITIONS                                                  │
│  ────────────────────────                                                  │
│  • RSI > 70 AND Low volume (Loss rate: 68%, Risk: 2.1x)                   │
│  • Downtrend regime AND High volatility (Loss rate: 58%, Risk: 1.8x)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1 Tab (Correlations)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TIER 1: CORRELATION ANALYSIS                            │
│                                                                             │
│  Point-Biserial Correlations with Trade Outcome                            │
│                                                                             │
│  Factor              Correlation    p-value    Significant                 │
│  ─────────────────────────────────────────────────────────────            │
│  RSI_14                 -0.28        0.001         ★★★                    │
│  Volume_Ratio           +0.22        0.008         ★★                     │
│  Insider_Buys           +0.18        0.024         ★                      │
│  Momentum_5d            +0.16        0.041         ★                      │
│  P/E_Ratio              -0.12        0.089                                │
│  Bollinger_%B           -0.09        0.152                                │
│  ...                                                                       │
│                                                                             │
│  ★★★ p < 0.01    ★★ p < 0.05    ★ p < 0.10                               │
│                                                                             │
│  [Chart: Correlation Bar Plot]                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 2 Tab (Hypothesis Testing)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIER 2: HYPOTHESIS TESTING                               │
│                                                                             │
│  LOGISTIC REGRESSION                                                       │
│  ───────────────────                                                       │
│  Dependent Variable: Good Trade (1) vs Bad Trade (0)                       │
│  Model Accuracy: 68.3%                                                     │
│                                                                             │
│  Coefficient Table:                                                        │
│  Variable         Coef      Std Err    Odds Ratio    p-value              │
│  ──────────────────────────────────────────────────────────               │
│  RSI_14          -0.034      0.011        0.97        0.002               │
│  Volume_Ratio    +0.452      0.178        1.57        0.011               │
│  Insider_Buys    +0.281      0.138        1.32        0.041               │
│  Momentum_5d     +0.125      0.062        1.13        0.045               │
│                                                                             │
│  ANOVA RESULTS (Categorical)                                               │
│  ──────────────────────────                                                │
│  Trend_Regime:      F=12.31, p<0.001  ★★★ Significant                     │
│  Volatility_Regime: F=3.24,  p=0.043  ★   Significant                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 3 Tab (Machine Learning)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIER 3: ML FEATURE IMPORTANCE                            │
│                                                                             │
│  RANDOM FOREST IMPORTANCE                                                  │
│  ────────────────────────                                                  │
│  RSI_14          ████████████████████████  0.18                           │
│  Momentum_5d     ██████████████████████    0.16                           │
│  Volume_Ratio    █████████████████         0.13                           │
│  Volatility      ████████████████          0.12                           │
│  Insider_Buys    ████████████              0.09                           │
│  P/E_Ratio       █████████                 0.07                           │
│  Trend_Regime    ████████                  0.06                           │
│  ...                                                                       │
│                                                                             │
│  SHAP SUMMARY                                                              │
│  ────────────                                                              │
│  • Low RSI values → Strong push toward "Good Trade"                       │
│  • High Volume Ratio → Moderate positive impact                           │
│  • High RSI values → Push toward "Bad Trade"                              │
│                                                                             │
│  CONSENSUS FEATURES (appear in top 5 across methods):                     │
│  RSI_14, Momentum_5d, Volume_Ratio                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scenarios Tab

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCENARIO DETECTION                                     │
│                                                                             │
│  BEST SCENARIOS (sorted by lift)                                           │
│  ──────────────────────────────                                            │
│                                                                             │
│  Scenario 1: Low RSI + High Volume                                         │
│  ├── Conditions: RSI < 35 AND Volume_Ratio > 1.5                          │
│  ├── Trades: 18 of 77                                                      │
│  ├── Win Rate: 72% (vs 58% baseline)                                      │
│  ├── Lift: 1.58x                                                          │
│  └── Statistical Significance: p = 0.003                                  │
│                                                                             │
│  Scenario 2: Insider Buying + Uptrend                                      │
│  ├── Conditions: Insider_Buys > 0 AND Trend = Up                          │
│  ├── Trades: 23 of 77                                                      │
│  ├── Win Rate: 65% (vs 58% baseline)                                      │
│  ├── Lift: 1.43x                                                          │
│  └── Statistical Significance: p = 0.018                                  │
│                                                                             │
│  WORST SCENARIOS (sorted by risk)                                          │
│  ─────────────────────────────                                             │
│                                                                             │
│  Scenario 1: High RSI + Low Volume                                         │
│  ├── Conditions: RSI > 70 AND Volume_Ratio < 0.8                          │
│  ├── Trades: 12 of 77                                                      │
│  ├── Loss Rate: 67% (vs 42% baseline)                                     │
│  ├── Risk: 2.1x worse                                                     │
│  └── Statistical Significance: p = 0.008                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Export Options

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPORT RESULTS                                        │
│                                                                             │
│  Export Format:                                                            │
│  ○ Excel (.xlsx)     Comprehensive workbook with all sheets               │
│  ○ JSON              Machine-readable structured data                      │
│  ○ CSV               Individual tables as CSV files                        │
│  ● HTML Report       Formatted report for sharing                          │
│                                                                             │
│  Include:                                                                  │
│  ☑ Summary statistics                                                      │
│  ☑ Correlation tables                                                      │
│  ☑ Hypothesis test results                                                 │
│  ☑ ML feature importance                                                   │
│  ☑ Scenario analysis                                                       │
│  ☐ Raw trade data                                                          │
│  ☑ Charts and visualizations                                               │
│                                                                             │
│  Output Path: [logs/factor_analysis/report_2024-01-15] [Browse]           │
│                                                                             │
│  [Export]                                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Applying Insights

After analysis, apply findings to your strategy:

| Finding | Action |
|---------|--------|
| "RSI > 70 increases loss rate" | Add RSI filter to block high-RSI entries |
| "Insider buying improves odds" | Prioritize signals with insider activity |
| "Low volume reduces edge" | Require minimum volume for entry |
| "Uptrend regime performs best" | Filter for trend direction |

---

## Related Documentation

- [Factor Analysis Concepts](../concepts/FACTOR_ANALYSIS.md) — Theory and methodology
- [Edge Analysis](ANALYSIS_TOOLS.md#edge-analysis) — Complementary entry validation
- [Backtest GUI](BACKTEST_GUI.md) — Testing strategy modifications
