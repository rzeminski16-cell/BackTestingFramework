# Factor Analysis System - GUI Specification

## Executive Summary

This document defines the complete user interface specification for the **Factor Analysis Module** - a professional-grade system for analyzing trade performance against market, technical, fundamental, insider, and options factors.

The system requires **three primary GUI applications**:

1. **Factor Analysis Dashboard** - Main analysis interface and visualization
2. **Configuration Manager** - Profile creation and parameter configuration
3. **Data Upload & Preparation** - Trade log and factor data ingestion

---

## Part 1: Factor Analysis Dashboard

### Purpose
Primary interface for running analyses, exploring results, and generating insights.

### 1.1 Main Interface Layout

#### Header Section
```
┌─────────────────────────────────────────────────────────────────┐
│ Factor Analysis Dashboard                     [Settings] [Help]  │
│ Analysis: Momentum Strategy | Period: Jan 2023 - Dec 2024       │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- App title with branding
- Current analysis name/profile dropdown
- Quick-access settings & help buttons
- Breadcrumb or analysis path indicator

#### Left Navigation Sidebar
```
┌──────────────────────────┐
│ ▶ Workspace              │
│   ├─ My Analyses        │
│   ├─ Recent             │
│   └─ Shared             │
│                         │
│ ▶ Upload Data           │
│                         │
│ ▶ Run Analysis          │
│                         │
│ ▶ View Results          │
│   ├─ Data Summary       │
│   ├─ Tier 1 (Exploratory) │
│   ├─ Tier 2 (Hypothesis) │
│   ├─ Tier 3 (ML)        │
│   ├─ Scenarios          │
│   └─ Interactions       │
│                         │
│ ▶ Export & Reports      │
│                         │
│ ▶ History & Audit       │
└──────────────────────────┘
```

**Navigation Items:**
- Workspace management (saved analyses, recent, shared)
- Data upload quick-link
- Analysis launcher
- Results explorer (expandable sections)
- Export options
- Audit trail/history

---

### 1.2 Data Summary View

**Trigger:** Click "Data Summary" in results explorer

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Data Summary Report                           [Refresh] [Export]│
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Trade Statistics                                                │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐    │
│  │ Total Trades│ Good Trades │  Bad Trades │Indeterminate│    │
│  │     245     │      89     │      72     │     84      │    │
│  │  (100%)     │  (36.3%)    │  (29.4%)    │  (34.3%)    │    │
│  └─────────────┴─────────────┴─────────────┴─────────────┘    │
│                                                                  │
│  Trade Characteristics                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Metric              │ Value                           │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │ Avg Trade Duration  │ 12.4 days (median: 8 days)     │    │
│  │ Avg Profit (Good)   │ +4.2% (${123,456} avg)         │    │
│  │ Avg Loss (Bad)      │ -2.1% (${-58,900} avg)         │    │
│  │ Win/Loss Ratio      │ 1.24 (good trades earn more)   │    │
│  │ Date Range          │ Jan 1, 2023 - Dec 31, 2024     │    │
│  │ Symbols Analyzed    │ 150 unique tickers             │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Data Quality Score: 92% ✓                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Price data coverage:    ████████░░░  98%             │    │
│  │ Fundamental coverage:   ██████████░░  85%             │    │
│  │ Insider data coverage:  ███████░░░░░░  58%            │    │
│  │ Options data coverage:  ████████░░░░░  72%            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Issues/Warnings                                                 │
│  ⚠ 12 trades missing exit price (will affect analysis)         │
│  ⚠ 8 symbols have < 30 days of price history                   │
│  ⓘ Fundamental data delayed by 1 day (as configured)            │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Summary statistics cards (total/good/bad/indeterminate trades)
- Detailed metrics table (duration, P&L, date range, symbols)
- Data quality gauge (visual progress bars by data source)
- Warnings and issues list with actionable messages

---

### 1.3 Tier 1 - Exploratory Analysis View

**Trigger:** Click "Tier 1 (Exploratory)" in results explorer

#### Layout - Two-Column Design
```
┌────────────────────────────────────────────────────────────────┐
│ Tier 1: Exploratory Analysis                 [PDF] [Excel]     │
├──────────────────────────────┬──────────────────────────────────┤
│  FACTOR SELECTION             │  CORRELATION RESULTS             │
│  ─────────────────────────    │  ──────────────────────────      │
│                                │                                  │
│  Category Filter:              │  Factor                    r    p-val  │
│  ☐ All Factors (50)           │  ─────────────────────────────  │
│  ☑ Technical (8)              │  ✓ insider_sentiment    0.34  <.001│
│  ☑ Fundamental (12)           │  ✓ tech_momentum_20     0.28  <.001│
│  ☑ Insider (6)                │  ✓ options_iv_pctl      0.21   .002  │
│  ☑ Options (5)                │  ✓ fund_roe_ttm         0.18   .008  │
│  ☑ Regime (4)                 │  ~ fund_pe_ratio        0.11   .065  │
│  ☐ Interaction (15)           │  ✗ tech_rsi_14         -0.05   .412  │
│                                │  ✗ tech_atr_pct        -0.12   .134  │
│  Sort By:                      │  ✗ regime_volatility   -0.15   .045  │
│  [▼ Correlation (desc)]        │  ...                             │
│                                │  [Load More] [Export Table]     │
│  Display:                      │                                  │
│  ◉ Significant Only (<.05)     │  Interpretation Guide:           │
│  ○ All Factors                 │  ✓ = Strong positive (helps)    │
│  ○ Top 20 Only                 │  ~ = Weak/neutral               │
│  ○ Top 50 Only                 │  ✗ = Weak negative              │
│                                │                                  │
│  [Apply Filters]               │  Histogram View:                 │
│                                │  ┌──────────────────────────┐   │
│                                │  │ insider_sentiment (r=.34) │   │
│                                │  │                          │   │
│                                │  │      ▄                   │   │
│                                │  │      █                   │   │
│                                │  │    ▄ █ ▄                │   │
│                                │  │ ▄▄ ██ ████             │   │
│                                │  │─────────────────────────│   │
│                                │  │ Bad    Indeterminate Good   │   │
│                                │  └──────────────────────────┘   │
│                                │                                  │
└──────────────────────────────────┴──────────────────────────────┘
```

**Left Panel - Factor Selection:**
- Expandable category filters (Technical, Fundamental, Insider, Options)
- Toggle individual factors on/off
- Sort by correlation, p-value, or name
- Display options: significant only, top N, all
- Apply/Reset buttons

**Right Panel - Results Table:**
- Factor name, correlation coefficient, p-value
- Visual indicators (✓✗~) for significance/direction
- Color coding: green (positive), red (negative), gray (neutral)
- Interactive histogram showing distribution by trade outcome
- Clickable factors to drill into details

**Additional Features:**
- Export correlation matrix as CSV/Excel
- Copy table to clipboard
- Download as PDF report
- Filter by significance threshold slider

---

### 1.4 Tier 2 - Hypothesis Testing View

**Trigger:** Click "Tier 2 (Hypothesis)" in results explorer

#### Sub-Tabs: Logistic Regression | ANOVA | Chi-Square | Interaction

##### Tab 1: Logistic Regression Results
```
┌────────────────────────────────────────────────────────────────┐
│ Tier 2: Hypothesis Testing - Logistic Regression              │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model Performance                                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Pseudo R² (McFadden)    │ 0.28  ████████░░░░░░░░░░ (Good)│ │
│  │ AIC                     │ 312.4                           │ │
│  │ BIC                     │ 378.2                           │ │
│  │ Samples Used            │ 215 / 245 (87.8%)              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Factor Coefficients (Predicting Good Trade)                    │
│  ┌──────────────┬────────┬──────────┬──────────┬────────┐     │
│  │ Factor       │Coefficient│ Odds Ratio│ Std Err│p-value│     │
│  ├──────────────┼────────┼──────────┼──────────┼────────┤     │
│  │insider_sent  │  0.847 │  2.33    │  0.198  │<.001  │     │
│  │tech_momentum │  0.564 │  1.76    │  0.152  │<.001  │     │
│  │options_iv_pct│ -0.342 │  0.71    │  0.118  │ .003  │     │
│  │fund_roe      │  0.193 │  1.21    │  0.096  │ .043  │     │
│  │regime_trend  │  0.254 │  1.29    │  0.171  │ .138  │     │
│  │... more      │  ...   │  ...     │  ...    │  ...  │     │
│  └──────────────┴────────┴──────────┴──────────┴────────┘     │
│                                                                  │
│  Interpretation:                                                 │
│  • Insider sentiment has STRONGEST positive effect              │
│    (1% increase → 2.33x odds of good trade)                    │
│  • Options IV percentile has INVERSE relationship               │
│    (higher vol → lower win probability)                        │
│                                                                  │
│  Prediction Accuracy                                             │
│  ┌────────────────────────┬────────────────────────┐          │
│  │ Sensitivity (True +)   │ 72.6%                  │          │
│  │ Specificity (True -)   │ 68.4%                  │          │
│  │ Overall Accuracy       │ 71.2%                  │          │
│  │ ROC-AUC                │ 0.78                   │          │
│  └────────────────────────┴────────────────────────┘          │
│                                                                  │
│  [Download Coefficients] [View Diagnostics] [Export Model]    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Model performance metrics (Pseudo R², AIC, BIC, sample count)
- Coefficient table with odds ratios and p-values
- Visual odds ratio bars (shows relative strength of effects)
- Interpretation guide with economic significance
- Prediction accuracy metrics (sensitivity, specificity, ROC-AUC)
- Diagnostic plots (residuals, calibration curve)

##### Tab 2: ANOVA Results
```
┌────────────────────────────────────────────────────────────────┐
│ Tier 2: ANOVA Analysis                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Select Factor:  [insider_sentiment ▼]                          │
│                                                                  │
│  Group Comparison (Good vs Bad trades)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Good Trades   Bad Trades   p-value       │  │
│  │ ─────────────────────────────────────────────────────────│  │
│  │ Mean              0.34          -0.12       <.001 ✓      │  │
│  │ Std Dev           0.28          0.31                     │  │
│  │ Min               -0.45         -0.78                    │  │
│  │ Max                0.89          0.45                    │  │
│  │ Median            0.41          -0.08                    │  │
│  │ Sample Size       89             72                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Distribution Visualization                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Good Trades  (─ ─ ─)   Bad Trades ( ─ ─ )     │  │
│  │      ▄▄                                                  │  │
│  │     ██                                                   │  │
│  │ ▄███████▄        ▄█▄                                    │  │
│  │███████████▄  ▄▄███████▄                                │  │
│  │────────────────────────────────                        │  │
│  │ -1.0   -0.5    0.0    0.5   1.0                        │  │
│  │            insider_sentiment                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Effect Size (Eta-Squared): 0.18 ★★★ (Large effect)           │
│  Cohen's d:                 1.24 (Very large difference)       │
│                                                                  │
│  F-Statistic Results:                                            │
│  F(1, 159) = 32.4, p < .001                                    │
│  Interpretation: insider_sentiment SIGNIFICANTLY differs        │
│  between good and bad trades.                                   │
│                                                                  │
│  [Next Factor ▶] [All Results] [Export Data]                   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Factor selector dropdown
- Group statistics table (mean, SD, min, max, median, n)
- Side-by-side distribution visualization
- Effect size metrics (Eta-squared, Cohen's d)
- F-statistic results and interpretation
- Navigation to next/previous factors

---

### 1.5 Tier 3 - Machine Learning Analysis View

**Trigger:** Click "Tier 3 (ML)" in results explorer

#### Sub-Tabs: Random Forest | SHAP | Mutual Information | Consensus

##### Tab 1: Random Forest Feature Importance
```
┌────────────────────────────────────────────────────────────────┐
│ Tier 3: Machine Learning - Random Forest                       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model Performance                                               │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Cross-Validation Accuracy:  76.2% (±3.4%)                 ││
│  │ Total Trees:                100                             ││
│  │ Max Depth:                  12                              ││
│  │ Min Samples per Leaf:       5                               ││
│  │ Training Samples:           215                             ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Top Features by Importance                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Feature                    Importance   Contribution       ││
│  │ ────────────────────────────────────────────────────────── ││
│  │ 1. insider_sentiment       █████████░░░  18.3%             ││
│  │ 2. tech_momentum_20        ████████░░░░░  16.8%            ││
│  │ 3. options_iv_percentile   ████████░░░░░  15.2%            ││
│  │ 4. fund_roe_ttm            ███████░░░░░░  14.1%            ││
│  │ 5. options_put_call_ratio  ███████░░░░░░  13.4%            ││
│  │ 6. regime_volatility       ████░░░░░░░░░   7.8%            ││
│  │ 7. tech_rsi_14             ███░░░░░░░░░░   7.1%            ││
│  │ 8. fund_price_to_book      ██░░░░░░░░░░░   4.2%            ││
│  │ 9. tech_close_vs_sma200    ██░░░░░░░░░░░   2.1%            ││
│  │ 10. insider_buy_count_30d  █░░░░░░░░░░░░   1.0%            ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Permutation Importance (Alternative View)                      │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Select Metric: [Accuracy ▼]                                ││
│  │                                                              ││
│  │ insider_sentiment       0.087  ████████░░░░░░░░            ││
│  │ tech_momentum_20        0.076  ███████░░░░░░░░░░           ││
│  │ options_iv_percentile   0.062  ██████░░░░░░░░░░░░          ││
│  │ ... (8 more)                                                ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Feature Interaction Strength                                   │
│  [Heatmap showing interaction pairs]                            │
│                                                                  │
│  [Download Model] [Export Predictions] [Load New Model]        │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Model performance summary (CV accuracy, tree count, depth)
- Top-N features by importance (bar chart with percentages)
- Permutation importance as alternative ranking
- Interactive feature list (click to drill into details)
- Feature interaction heatmap
- Model export/download

##### Tab 2: SHAP Analysis
```
┌────────────────────────────────────────────────────────────────┐
│ Tier 3: SHAP Explainability Analysis                           │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Feature Importance (Mean |SHAP| values)                        │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ insider_sentiment        ████████░░░░░░░░░  0.142          ││
│  │ tech_momentum_20         ███████░░░░░░░░░░░  0.128         ││
│  │ options_iv_percentile    ██████░░░░░░░░░░░░  0.114         ││
│  │ fund_roe_ttm             ██████░░░░░░░░░░░░  0.108         ││
│  │ options_put_call_ratio   █████░░░░░░░░░░░░░  0.098         ││
│  │ ... (more)                                                   ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Feature Impact Direction (SHAP Summary)                        │
│  ┌────────────────────────────────────────────────────────────┐│
│  │Feature                     Impact on Trade Outcome         ││
│  │────────────────────────────────────────────────────────────││
│  │insider_sentiment  ◄─ ─ ─ ─ ● ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ►    ││
│  │                    Low                      High            ││
│  │                    (Bad)                     (Good)         ││
│  │                                                              ││
│  │                    ● = Mean SHAP value                     ││
│  │                                                              ││
│  │tech_momentum_20   ◄─ ─ ─ ● ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ► ││
│  │                                                              ││
│  │ ... more features                                           ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Individual Prediction Explainer                                │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Select Trade: [Trade #47 ▼]                                ││
│  │ Actual Outcome: Good (+4.2%)  │  Predicted: Good (87%)    ││
│  │                                                              ││
│  │ Contribution to Prediction:                                 ││
│  │                                                              ││
│  │ Base prediction        0.50                                 ││
│  │ insider_sentiment     +0.18  ✓ (high positive impact)      ││
│  │ tech_momentum_20      +0.12  ✓ (positive)                  ││
│  │ options_iv_pctl       -0.08  ✗ (reduces confidence)        ││
│  │ fund_roe_ttm          +0.10  ✓ (positive)                  ││
│  │ regime_volatility     -0.03  ~ (negligible)               ││
│  │ ─────────────────────────────                              ││
│  │ Final prediction       0.79 (79% probability of good trade)││
│  │                                                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Export SHAP Values] [View Dependence Plots] [Scatter View]   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Mean absolute SHAP values (feature importance)
- SHAP summary plot (impact direction and magnitude)
- Individual trade prediction explanation (waterfall chart)
- Contribution breakdown per feature for selected trade
- Dependence plots (feature value vs SHAP value)
- Force plots for specific predictions

##### Tab 3: Consensus Features
```
┌────────────────────────────────────────────────────────────────┐
│ Tier 3: Consensus Analysis (Agreement Across Methods)         │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Top Features Ranked by ALL Methods                             │
│  ┌────────────────────────────────────────────────────────────┐│
│  │Rank│ Feature              │ Correlation  │ RF  │SHAP │Score││
│  │    │                      │ (Tier 1)     │Rank │Rank │ /100││
│  │────│──────────────────────│──────────────│─────│─────│──────│
│  │ 1  │ insider_sentiment    │ r=0.34 ✓✓✓  │ 1   │ 1   │ 100 ││
│  │ 2  │ tech_momentum_20     │ r=0.28 ✓✓✓  │ 2   │ 2   │  97 ││
│  │ 3  │ options_iv_pctl      │ r=0.21 ✓✓   │ 3   │ 3   │  94 ││
│  │ 4  │ fund_roe_ttm         │ r=0.18 ✓    │ 4   │ 4   │  91 ││
│  │ 5  │ options_put_call_rto │ r=0.14 ✓    │ 5   │ 5   │  88 ││
│  │ 6  │ regime_volatility    │ r=0.11 ~    │ 6   │ 9   │  74 ││
│  │ 7  │ tech_rsi_14          │ r=0.08 ~    │ 7   │ 12  │  68 ││
│  │ ... │                      │              │     │     │     ││
│  └────│──────────────────────│──────────────│─────│─────│──────┘│
│                                                                  │
│  Confidence Levels:                                              │
│  ★★★★★ (95-100) = Highly Confident - Use in production rules  │
│  ★★★★☆ (85-95)  = Confident - Validated across methods       │
│  ★★★☆☆ (70-85)  = Moderate - Consider in analysis             │
│  ★★☆☆☆ (55-70)  = Weak - Exploratory use only                │
│  ★☆☆☆☆ (<55)    = Not Validated - Ignore                     │
│                                                                  │
│  Method Agreement Matrix                                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           │Corr│ RF │SHAP│Logit│ANOVA│ Agree │         │ │
│  │───────────┼────┼────┼────┼─────┼─────┼───────┤         │ │
│  │insider_sentiment │ 1  │ 1  │ 1  │ 1   │ 1   │  5/5  │ │
│  │tech_momentum_20  │ 2  │ 2  │ 2  │ 2   │ 2   │  5/5  │ │
│  │options_iv_pctl   │ 3  │ 3  │ 3  │ 3   │ 3   │  5/5  │ │
│  │fund_roe_ttm      │ 4  │ 4  │ 4  │ 4   │ 4   │  5/5  │ │
│  │regime_volatility │ 6  │ 6  │ 9  │ 7   │ 8   │  4/5  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  [Download Consensus List] [Create Trading Rules] [Validate]   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Multi-method ranking table
- Consensus score calculation
- Method agreement matrix
- Confidence level stars
- Visual highlighting of high-agreement features

---

### 1.6 Scenario Analysis View

**Trigger:** Click "Scenarios" in results explorer

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Scenario Analysis: Best & Worst Conditions                     │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Filter by Lift: ▼ All] [Min Trades: 15▼] [Sort: Lift↓]      │
│                                                                  │
│  BEST Trading Scenarios (High Win Rate)                         │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Scenario: Insider Buying + Bullish Momentum                 ││
│  │ ────────────────────────────────────────────────────────────││
│  │ Conditions:                                                  ││
│  │  • insider_sentiment > 0.5  (strong buy signal)             ││
│  │  • tech_momentum_20 > 2.0   (positive momentum)             ││
│  │  • regime_trend = bullish   (bull market environment)       ││
│  │                                                              ││
│  │ Results:                                                     ││
│  │  Trades in Scenario:        47                              ││
│  │  Good Trade Rate:           79.1% (vs. 36.3% baseline)     ││
│  │  Lift:                      2.18x (2.18x better than avg)  ││
│  │  Avg Win:                   +5.8%                           ││
│  │  Avg Loss (when bad):       -1.2%                           ││
│  │  Confidence:                ████████░░ 82%                 ││
│  │                                                              ││
│  │ [View Details] [Add Rule] [Backtest Specific] [Export]     ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Scenario: Low Volatility + High ROE                        ││
│  │ [Similar structure...]                                      ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Scenario: Earnings Beat + Options Bullish                  ││
│  │ [Similar structure...]                                      ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  WORST Trading Scenarios (Low Win Rate)                        │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Scenario: Insider Selling + High Volatility                 ││
│  │ ────────────────────────────────────────────────────────────││
│  │ Conditions:                                                  ││
│  │  • insider_sentiment < -0.4 (selling signal)                ││
│  │  • regime_volatility = high (volatile market)               ││
│  │                                                              ││
│  │ Results:                                                     ││
│  │  Trades in Scenario:        31                              ││
│  │  Good Trade Rate:           16.1% (vs. 36.3% baseline)     ││
│  │  Lift:                      0.44x (0.44x of average)        ││
│  │  Avg Loss:                  -4.2%                           ││
│  │  Confidence:                ███████░░░ 71%                 ││
│  │                                                              ││
│  │ [View Details] [Avoid Rule] [Filter Specific] [Export]     ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Scenario Comparison Matrix                                    │
│  [Scatter plot: Win Rate (Y) vs Lift (X)]                      │
│  [Colored bubbles: size = # of trades, color = confidence]    │
│                                                                  │
│  [Create Custom Scenario] [Test Scenario] [Export All]        │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Best scenarios (highest win rate/lift)
- Worst scenarios (lowest win rate/lift)
- Condition details for each scenario
- Trade count, win rate, lift metrics
- Confidence gauge
- Quick action buttons (add rule, backtest, export)
- Scenario comparison matrix/scatter plot
- Custom scenario builder
- Scenario validation/backtesting interface

---

### 1.7 Export & Reporting View

**Trigger:** Click "Export & Reports" in navigation

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Export & Reporting                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Report Templates                                               │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Report Type              │ Format │ Size    │ Action       ││
│  │ ──────────────────────── │────────│─────────│────────────  ││
│  │ Executive Summary         │ PDF    │ 2 MB    │ [Generate]   ││
│  │ Full Analysis Report      │ PDF    │ 45 MB   │ [Generate]   ││
│  │ Data Tables & Raw Results │ Excel  │ 12 MB   │ [Generate]   ││
│  │ Scenario Deep Dive        │ PDF    │ 8 MB    │ [Generate]   ││
│  │ JSON Payload (Web/API)    │ JSON   │ 3 MB    │ [Generate]   ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Custom Export                                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Select Sections:                                             ││
│  │ ☑ Data Summary         ☑ Tier 1 Results     ☑ Tier 2       ││
│  │ ☑ Tier 3 Results       ☑ Scenarios          ☑ Audit Trail  ││
│  │                                                              ││
│  │ Export Format:                                               ││
│  │ ◉ PDF (single doc)   ○ Excel (multiple sheets)             ││
│  │ ○ HTML (interactive)  ○ JSON (API format)                   ││
│  │                                                              ││
│  │ Include Details:                                             ││
│  │ ☑ Visualizations (charts, plots)                           ││
│  │ ☑ Statistical Tables                                       ││
│  │ ☑ Interpretation Guides                                    ││
│  │ ☑ Individual Trade Details                                 ││
│  │ ☑ Audit Log & Metadata                                     ││
│  │                                                              ││
│  │ [Export] [Preview] [Schedule]                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Recent Exports                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Momentum_Strategy_Full_2024-12-29.pdf   12/29 @ 2:45 PM   ││
│  │ │ Download │ View │ Resend │ Delete                        ││
│  │                                                              ││
│  │ Momentum_Strategy_Tables_2024-12-29.xlsx 12/29 @ 2:30 PM  ││
│  │ │ Download │ View │ Resend │ Delete                        ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Scheduled Reports                                               │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ [ + Add Schedule ]                                           ││
│  │                                                              ││
│  │ Weekly Summary (Mondays 6:00 AM)                            ││
│  │ Format: PDF │ Recipients: you@example.com │ [Edit] [Delete]││
│  │                                                              ││
│  │ Monthly Full Analysis (1st of month 9:00 AM)                ││
│  │ Format: Excel │ Recipients: you@..., boss@... │[Edit] [Del] ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Pre-built report templates (Executive Summary, Full Analysis, etc.)
- Custom export builder (select sections, format, details)
- Export format options (PDF, Excel, HTML, JSON)
- Recent exports download/view/resend/delete
- Scheduled report automation
- Email delivery configuration
- Preview before download

---

### 1.8 Audit Trail & History View

**Trigger:** Click "History & Audit" in navigation

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Analysis History & Audit Trail                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Analysis Runs                                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Date/Time         │Profile │Status │Good│Bad │Duration │...││
│  │──────────────────┼────────┼───────┼───┼────┼─────────┼──││
│  │12/29 2:45 PM    │momentum│✓      │89 │72  │ 3.2 min │[▼]││
│  │12/29 11:20 AM   │mean_rev│✓      │156│48  │ 5.8 min │[▼]││
│  │12/28 3:15 PM    │momentum│✓      │89 │72  │ 3.2 min │[▼]││
│  │                                                              ││
│  │ [Details] [Rerun] [Compare] [Export]                        ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Expand Details (click row):                                    │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Analysis ID:          FA-20241229-0245-89ab                 ││
│  │ Profile Used:         momentum                              ││
│  │ Data Files:           momentum_trades.csv, prices.csv       ││
│  │ Entry Time:           2024-12-29 14:45:23 GMT              ││
│  │ Completion Time:      2024-12-29 14:48:32 GMT              ││
│  │ Status:               SUCCESS                               ││
│  │                                                              ││
│  │ Configuration Applied:                                       ││
│  │ • Good Threshold:     2.0% (ABSOLUTE)                      ││
│  │ • Bad Threshold:      -1.0%                                ││
│  │ • Fundamentals Delay: 1 day                                ││
│  │ • Insiders Delay:     3 days                               ││
│  │                                                              ││
│  │ Input Summary:                                               ││
│  │ • Trades Analyzed:    245 total, 215 with complete data   ││
│  │ • Date Range:         Jan 1 - Dec 31, 2024                ││
│  │ • Symbols:           150 unique tickers                    ││
│  │ • Price Data Points:  ~30,000 daily records                ││
│  │ • Fundamental Data:   89 stocks with full coverage          ││
│  │                                                              ││
│  │ Output Summary:                                              ││
│  │ • Good Trades:        89 (36.3%)                            ││
│  │ • Bad Trades:         72 (29.4%)                            ││
│  │ • Indeterminate:      84 (34.3%)                            ││
│  │ • Significant Factors: 12 (p<.05)                           ││
│  │                                                              ││
│  │ [Show Full Audit Log] [Rerun Analysis] [Download JSON]     ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Audit Log Viewer                                                │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Filter: [All ▼] Timestamps: [Auto ▼]                       ││
│  │                                                              ││
│  │ 14:45:23  INFO  Analysis started with config: momentum    ││
│  │ 14:45:25  INFO  Loading trades from momentum_trades.csv   ││
│  │ 14:45:28  INFO  245 trades loaded successfully             ││
│  │ 14:45:30  INFO  Loading price data (150 symbols)           ││
│  │ 14:45:45  INFO  Price data loaded: 30,124 records          ││
│  │ 14:45:47  INFO  Classifying trades...                      ││
│  │ 14:45:50  INFO  Good: 89, Bad: 72, Indeterminate: 84      ││
│  │ 14:45:52  INFO  Starting Tier 1 (exploratory) analysis     ││
│  │ 14:46:12  INFO  Tier 1 complete: 12 factors significant    ││
│  │ 14:46:14  INFO  Starting Tier 2 (hypothesis) analysis      ││
│  │ 14:46:45  INFO  Logistic regression complete (Pseudo R²=.28)││
│  │ 14:46:47  INFO  ANOVA tests complete (5 significant)       ││
│  │ 14:47:15  INFO  Starting Tier 3 (ML) analysis              ││
│  │ 14:47:50  INFO  Random Forest trained (76.2% CV accuracy)  ││
│  │ 14:48:10  INFO  SHAP analysis complete                     ││
│  │ 14:48:20  INFO  Scenario detection running...               ││
│  │ 14:48:30  INFO  Found 8 best scenarios, 6 worst scenarios  ││
│  │ 14:48:32  INFO  Analysis completed successfully            ││
│  │                                                              ││
│  │ [Save Log] [Export as CSV] [Copy to Clipboard]             ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Clear History] [Compare Two Runs] [Revert Changes]          │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Chronological list of analysis runs
- Quick status, trade counts, duration
- Expandable details for each run
- Full audit log viewer with timestamps
- Configuration snapshot
- Input/output summaries
- Save/export audit logs
- Comparison between runs
- Rerun capability

---

## Part 2: Configuration Manager

### Purpose
Create, edit, and manage analysis configurations/profiles.

### 2.1 Main Configuration Interface

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Configuration Manager - Analysis Profiles                      │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [+ New Profile]  [Import]  [Export]  [Duplicate]             │
│                                                                  │
│  My Profiles                                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Profile Name         │Updated  │Config │Use │Action        ││
│  │ ─────────────────── │─────────│───────┤────┤─────────────  ││
│  │ momentum             │12/29    │ view  │ 12 │ [Edit][Delete]││
│  │ mean_reversion       │12/27    │ view  │  8 │ [Edit][Delete]││
│  │ earnings_driven      │12/15    │ view  │  3 │ [Edit][Delete]││
│  │ conservative         │11/30    │ view  │  1 │ [Edit][Delete]││
│  │ _default             │12/01    │ view  │  - │ (read-only)   ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- List of saved profiles with usage count
- Create new profile
- Edit/duplicate existing profile
- Import/export profiles
- Quick access to profile configuration viewer

### 2.2 Edit Profile - Trade Classification

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Edit Profile: momentum                   [Save] [Cancel]       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TAB SELECT: [Trade Class] Data Align Factors Statistics      │
│                          Scenarios Output                       │
│                                                                  │
│  TRADE CLASSIFICATION SETTINGS                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Good Trade Threshold (%)                                   ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Threshold Type: ◉ Absolute  ○ Relative             │ ││
│  │  │                                                       │ ││
│  │  │ Minimum Return:   [2.0]%  ←────────→  [slider]     │ ││
│  │  │                   Min: 0% Max: 20%                  │ ││
│  │  │                                                       │ ││
│  │  │ Description: Trades returning ≥ 2.0% are classified │ ││
│  │  │             as \"good trades\" in the analysis.        │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Bad Trade Threshold (%)                                    ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Maximum Return:   [-1.0]%  ←────────→  [slider]    │ ││
│  │  │                   Min: -20% Max: 0%                 │ ││
│  │  │                                                       │ ││
│  │  │ Description: Trades returning ≤ -1.0% are classified│ ││
│  │  │             as \"bad trades\" in the analysis.        │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Indeterminate Trade Handling                              ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Max Duration for Indeterminate:  [15] days          │ ││
│  │  │                                  [slider: 0-30]     │ ││
│  │  │                                                       │ ││
│  │  │ Min Duration Before 'Bad' Label: [20] days          │ ││
│  │  │                                  [slider: 0-60]     │ ││
│  │  │                                                       │ ││
│  │  │ Description: Trades between good and bad thresholds │ ││
│  │  │             AND lasting < 15 days are labeled       │ ││
│  │  │             \"indeterminate\" (noise/short signals). │ ││
│  │  │             Trades lasting >= 20 days become \"bad\"  │ ││
│  │  │             if still negative, even if small loss.   │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Preview Impact                                             ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Estimated classification breakdown:                  │ ││
│  │  │                                                       │ ││
│  │  │ Good:            89 trades (36.3%)                  │ ││
│  │  │ Bad:             72 trades (29.4%)                  │ ││
│  │  │ Indeterminate:   84 trades (34.3%)                  │ ││
│  │  │                                                       │ ││
│  │  │ [Based on current sample data]                       │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Help] [Load Template] [Reset to Default] [Save] [Cancel]    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Threshold sliders with live preview
- Absolute vs Relative threshold mode toggle
- Descriptive text for each parameter
- Classification impact preview
- Template loader for quick presets
- Reset to defaults option

### 2.3 Edit Profile - Data Alignment

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Edit Profile: momentum                   [Save] [Cancel]       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TAB SELECT: Trade Class [Data Align] Factors Statistics      │
│                          Scenarios Output                       │
│                                                                  │
│  DATA ALIGNMENT & REPORTING DELAY SETTINGS                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Fundamental Data Reporting Delay                           ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Days After Report Release: [1] day                  │ ││
│  │  │                            [slider: 0-14 days]      │ ││
│  │  │                                                       │ ││
│  │  │ Description: Fundamental metrics (PE, ROE, etc.)    │ ││
│  │  │             become available N days after official   │ ││
│  │  │             public release. Default: 1 day for      │ ││
│  │  │             consensus data; 2-3 days for raw data.  │ ││
│  │  │             Set to 0 if using real-time feeds.     │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Insider Transaction Reporting Delay                        ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Days After Filing Date: [3] days                    │ ││
│  │  │                         [slider: 0-14 days]         │ ││
│  │  │                                                       │ ││
│  │  │ Description: SEC insider transactions are officially │ ││
│  │  │             filed N days after the actual trade.    │ ││
│  │  │             Standard SEC filing: 2 business days    │ ││
│  │  │             (3-5 calendar days). Set to actual lag   │ ││
│  │  │             between transaction and public filing.   │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Options Data Lookback Period                               ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Historical Days to Include: [60] days               │ ││
│  │  │                             [slider: 20-252 days]   │ ││
│  │  │                                                       │ ││
│  │  │ Description: How far back to look for options        │ ││
│  │  │             metrics (IV percentile calculations).    │ ││
│  │  │             Standard: 60-252 days (1 year).          │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Price Data Handling                                        ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ ☑ Allow Forward-Fill for Missing Dates              │ ││
│  │  │   (Will use last known price for weekends/holidays) │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Include Pre-Market/After-Hours (when available)   │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Adjust for Stock Splits/Dividends                 │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Timezone Settings                                          ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Analysis Timezone: [UTC ▼]                          │ ││
│  │  │ Convert All Times to: [UTC ▼]                       │ ││
│  │  │                                                       │ ││
│  │  │ (Important for international trading / accurate     │ ││
│  │  │  temporal alignment across time zones)              │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Help] [Load Template] [Reset to Default] [Save] [Cancel]    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Reporting delay sliders for fundamentals, insiders, options
- Lookback period for IV calculations
- Price data handling options
- Timezone configuration
- Forward-fill and adjustment toggles

### 2.4 Edit Profile - Factor Settings

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Edit Profile: momentum                   [Save] [Cancel]       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TAB SELECT: Trade Class Data Align [Factors] Statistics      │
│                          Scenarios Output                       │
│                                                                  │
│  FACTOR ENGINEERING SETTINGS                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Factor Categories (Enable/Disable)                         ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ ☑ Technical Factors (8 factors)                     │ ││
│  │  │    RSI, MACD, Bollinger Bands, Momentum, etc.       │ ││
│  │  │    [Customize Individual Factors ▼]                 │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Fundamental Factors (12 factors)                  │ ││
│  │  │    PE, Price/Book, ROE, Revenue Growth, etc.        │ ││
│  │  │    [Customize Individual Factors ▼]                 │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Insider Factors (6 factors)                       │ ││
│  │  │    Buy/sell counts, sentiment, net shares, etc.     │ ││
│  │  │    [Customize Individual Factors ▼]                 │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Options Factors (5 factors)                       │ ││
│  │  │    IV, Put/Call ratio, Skew, Sentiment, etc.        │ ││
│  │  │    [Customize Individual Factors ▼]                 │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Market Regime Factors (4 factors)                 │ ││
│  │  │    Volatility regime, Trend, Momentum, etc.         │ ││
│  │  │    [Customize Individual Factors ▼]                 │ ││
│  │  │                                                       │ ││
│  │  │ ☐ Interaction Factors (auto-detected)               │ ││
│  │  │    High-order interactions (15+ auto-generated)      │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Regime Factors (auto-classified)                  │ ││
│  │  │    Market regime classification                      │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Outlier Handling                                           ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Method: [Winsorize ▼]                               │ ││
│  │  │ (Options: Winsorize, Drop, Separate, MICE)          │ ││
│  │  │                                                       │ ││
│  │  │ Threshold (Std Devs): [3.0] ← ────────→ [slider]   │ ││
│  │  │ Values > 3 STD from mean are treated as outliers     │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Missing Data Handling                                      ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Strategy: [Drop ▼]                                  │ ││
│  │  │ (Options: Drop rows, Forward-fill, Backward-fill)   │ ││
│  │  │                                                       │ ││
│  │  │ Max Missing %:  [50]% ← ────────→ [slider]         │ ││
│  │  │ Drop factors with > 50% missing values               │ ││
│  │  │                                                       │ ││
│  │  │ Impact: This may reduce factors from ~50 to ~40     │ ││
│  │  │ depending on your data sources.                      │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Normalization                                              ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Method: [Z-Score ▼]                                 │ ││
│  │  │ (Options: Z-Score, Min-Max, Percentile, None)       │ ││
│  │  │                                                       │ ││
│  │  │ Apply to: ◉ All factors  ○ Per-factor  ○ None      │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Help] [Load Template] [Reset to Default] [Save] [Cancel]    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Category toggles with factor counts
- Customize individual factor selection
- Outlier handling method and threshold
- Missing data strategy
- Normalization method selection
- Impact preview

### 2.5 Edit Profile - Statistical Analysis

#### Layout (condensed)
```
┌────────────────────────────────────────────────────────────────┐
│ Edit Profile: momentum                   [Save] [Cancel]       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TAB SELECT: Trade Class Data Align Factors [Statistics]      │
│                          Scenarios Output                       │
│                                                                  │
│  STATISTICAL ANALYSIS SETTINGS                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  TIER 1: Exploratory Analysis                              ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ ☑ Enabled                                            │ ││
│  │  │   • Point-biserial correlations (factor vs outcome)  │ ││
│  │  │   • Descriptive statistics                           │ ││
│  │  │                                                       │ ││
│  │  │ Significance Level (alpha): [0.05] ◄─────►          │ ││
│  │  │ Min Samples for Tier 1: [30] ◄─────►                │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  TIER 2: Hypothesis Testing                                ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ ☑ Enabled                                            │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Logistic Regression                                │ ││
│  │  │   (Predicting good vs bad trades)                    │ ││
│  │  │                                                       │ ││
│  │  │ ☑ ANOVA Tests                                        │ ││
│  │  │   (Group-wise factor differences)                    │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Chi-Square Tests                                   │ ││
│  │  │   (Categorical factor associations)                  │ ││
│  │  │                                                       │ ││
│  │  │ Significance Level (alpha): [0.05] ◄─────►          │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  TIER 3: Machine Learning                                  ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ ☑ Enabled                                            │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Random Forest Classifier                           │ ││
│  │  │   Number of Trees: [100] ◄─────►                    │ ││
│  │  │   Max Depth: [12] ◄─────►                            │ ││
│  │  │   Min Samples per Leaf: [5] ◄─────►                  │ ││
│  │  │                                                       │ ││
│  │  │ ☑ SHAP Explainability                                │ ││
│  │  │   (Model-agnostic feature importance)                │ ││
│  │  │                                                       │ ││
│  │  │ ☑ Mutual Information Analysis                        │ ││
│  │  │   (Non-linear factor relationships)                  │ ││
│  │  │                                                       │ ││
│  │  │ Minimum Samples for Tier 3: [50] ◄─────►            │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Multiple Testing Correction                               ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Method: [FDR (Benjamini-Hochberg) ▼]                │ ││
│  │  │ (Options: None, Bonferroni, FDR, Sidak)             │ ││
│  │  │                                                       │ ││
│  │  │ Significance Level: [0.05] ◄─────►                   │ ││
│  │  │                                                       │ ││
│  │  │ Description: FDR controls false discovery rate,      │ ││
│  │  │ balancing sensitivity with specificity for           │ ││
│  │  │ multiple comparisons.                                │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Help] [Load Template] [Reset to Default] [Save] [Cancel]    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 2.6 Edit Profile - Scenario Analysis

```
┌────────────────────────────────────────────────────────────────┐
│  TAB SELECT: Trade Class Data Align Factors Statistics        │
│                          [Scenarios] Output                     │
│                                                                  │
│  SCENARIO ANALYSIS SETTINGS                                    │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  Scenario Detection Mode                                    ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ ◉ Binary (Top vs Bottom - Best for simple rules)    │ ││
│  │  │ ○ Clustering (K-means - Find natural groups)         │ ││
│  │  │ ○ Auto (Both - Let system choose optimal)            │ ││
│  │  │                                                       │ ││
│  │  │ For Clustering Mode:                                 │ ││
│  │  │ Number of Clusters: [5] ◄─────►                      │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  │  Scenario Filtering                                         ││
│  │  ┌──────────────────────────────────────────────────────┐ ││
│  │  │ Min Trades Per Scenario: [20] ◄─────►               │ ││
│  │  │ (Only scenarios with 20+ trades are valid)           │ ││
│  │  │                                                       │ ││
│  │  │ Min Lift Threshold: [1.2] ◄─────►                    │ ││
│  │  │ (Only report scenarios 1.2x better/worse than avg)  │ ││
│  │  │                                                       │ ││
│  │  │ Estimated scenarios to find: ~8-15 best/worst        │ ││
│  │  └──────────────────────────────────────────────────────┘ ││
│  │                                                              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Help] [Load Template] [Reset to Default] [Save] [Cancel]    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Data Upload & Preparation Interface

### Purpose
Load trade logs and factor data, validate data quality, and prepare for analysis.

### 3.1 Data Upload Main View

#### Layout
```
┌────────────────────────────────────────────────────────────────┐
│ Data Upload & Preparation                                      │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [ + Upload Trade Log ] [ + Upload Price Data ] [ + Fundamentals]
│  [ + Insider Data ]    [ + Options Data ]                      │
│                                                                  │
│  CURRENTLY LOADED DATA                                          │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Data Source            │ Records  │ Date Range  │ Status   ││
│  │ ───────────────────────┼──────────┼─────────────┼────────  ││
│  │ Trade Log              │  245     │ Jan - Dec   │  ✓ Ready ││
│  │ (momentum_trades.csv)  │ 2024     │             │           ││
│  │                                                              ││
│  │ Price Data             │ 30,124   │ Jan - Dec   │  ✓ Ready ││
│  │ (daily_prices.csv)     │          │ 2024        │           ││
│  │                                                              ││
│  │ Fundamental Data       │ 1,456    │ Jan - Dec   │  ✓ Ready ││
│  │ (fundamentals.csv)     │          │ 2024        │           ││
│  │                                                              ││
│  │ Insider Data           │ 892      │ Jan - Dec   │  ✓ Ready ││
│  │ (insider_transactions)│          │ 2024        │           ││
│  │                                                              ││
│  │ Options Data           │ 18,547   │ Jan - Dec   │  ✓ Ready ││
│  │ (options_quotes.csv)   │          │ 2024        │           ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  DATA QUALITY ASSESSMENT                                        │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Overall Data Quality Score: 92%  ████████░░                 ││
│  │                                                              ││
│  │ Issues Found:                                                ││
│  │ ⚠ [12 trades] Missing exit_price (will use exit_date close)││
│  │ ⚠ [8 symbols] Have < 30 days of price history              ││
│  │ ⓘ [4 symbols] Missing fundamental data (will skip)          ││
│  │ ⓘ [Insider data] Delayed by 3 days (as expected)           ││
│  │                                                              ││
│  │ [View Details] [Auto-Fix] [Manual Review]                  ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ACTIONS                                                         │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ [Clear All Data]  [Review/Edit]  [Ready to Analyze ✓]    ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Upload buttons for each data type
- Loaded data summary table
- File names and record counts
- Date range coverage
- Status indicators
- Data quality score with breakdowns
- Issue warnings with action buttons

### 3.2 Upload Dialog - Trade Log

#### Layout (Modal)
```
┌────────────────────────────────────────────────────────────────┐
│ Upload Trade Log                             [?] [×]          │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Upload File                                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Drag & drop CSV file here, or [Click to Browse]            ││
│  │                                                              ││
│  │ [Selected: momentum_trades.csv]                             ││
│  │ File Size: 145 KB  |  Preview: 245 trades                 ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Column Mapping (Auto-detected)                                │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Required Columns:                                            ││
│  │                                                              ││
│  │ Symbol         [ column_0: symbol ✓ ]                      ││
│  │ Entry Date     [ column_1: entry_date ✓ ]                  ││
│  │ Exit Date      [ column_2: exit_date ✓ ]                   ││
│  │ P&L ($)        [ column_3: pl ✓ ]                          ││
│  │ P&L (%)        [ column_4: pl_pct ✓ ]                      ││
│  │                                                              ││
│  │ Optional Columns:                                            ││
│  │ Entry Price    [ column_5: entry_price ✓ ]                 ││
│  │ Exit Price     [ column_6: exit_price ○ (not found)]       ││
│  │                                                              ││
│  │ [Need to map? Click here to manually adjust]                ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Data Preview                                                    │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ symbol │entry_date │exit_date  │pl_pct │entry_price│exit_p││
│  │────────┼───────────┼───────────┼───────┼───────────┼──────││
│  │AAPL    │2024-01-05 │2024-01-12 │+3.24% │ 180.50   │186.35││
│  │MSFT    │2024-01-08 │2024-01-22 │-1.87% │ 375.20   │368.20││
│  │TSLA    │2024-01-15 │2024-01-18 │+0.94% │ 245.60   │247.90││
│  │ ... (242 more rows)                                         ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Cancel]  [Reset]  [Next >]                                   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Drag-and-drop or file browser
- Auto column detection
- Column mapping interface
- Data preview table
- File validation

### 3.3 Data Quality Review

#### Layout (Full Screen)
```
┌────────────────────────────────────────────────────────────────┐
│ Data Quality Review & Fixes                [Back] [Continue]  │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Overall Data Quality: 92% ████████░░                           │
│                                                                  │
│  ISSUES & WARNINGS                                              │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ TYPE │SEVERITY│COUNT │AFFECTED RECORDS      │ACTION         ││
│  │──────┼────────┼──────┼──────────────────────┼──────────────  ││
│  │Missing│Warning │ 12  │exit_price (trades)   │[Auto-Fix][▼]  ││
│  │Data   │        │     │Will use exit close   │              ││
│  │       │        │     │                      │              ││
│  │Short  │Info    │  8  │8 symbols < 30 days   │[Auto-Fix][▼]  ││
│  │History│        │     │May skip tech factors  │              ││
│  │       │        │     │                      │              ││
│  │Missing│Warning │  4  │Fundamental data gaps │[Drop][Keep]   ││
│  │Funds  │        │     │(4 small-cap stocks)  │              ││
│  │       │        │     │                      │              ││
│  │Date   │Info    │  0  │All trades in range   │[OK]           ││
│  │Overlap│        │     │                      │              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  IMPACT ANALYSIS                                                │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Action: [Current Action] → [Expected Outcome]              ││
│  │                                                              ││
│  │ Auto-Fix Missing exit_price                                ││
│  │  Affected Trades: 12 / 245 (4.9%)                          ││
│  │  Method: Use closing price from exit_date                  ││
│  │  Impact: ✓ No trades will be lost; accuracy improved      ││
│  │                                                              ││
│  │ Auto-Fix Short History Symbols                             ││
│  │  Affected Symbols: 8 / 150 (5.3%)                          ││
│  │  Method: Calculate available technical factors              ││
│  │  Impact: ⚠ Some technical factors may be unavailable       ││
│  │          (e.g., 200-day SMA unavailable)                   ││
│  │                                                              ││
│  │ Keep Symbols Missing Fundamental Data                      ││
│  │  Affected Symbols: 4 / 150 (2.7%)                          ││
│  │  Method: Skip fundamental factors for these 4              ││
│  │  Impact: ✓ Trades kept; limited factor availability       ││
│  │                                                              ││
│  │ Overall Impact: 245 → 245 trades (100% retention)         ││
│  │                 Coverage: 50 factors → 48 factors          ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  CONFIDENCE & READINESS                                         │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Data Quality Score After Fixes: 95% ████████░░░            ││
│  │ Minimum Tier 3 ML Requirements: ✓ MET (215+ trades)        ││
│  │ All Core Factors Available: ✓ YES (40+ of 50)              ││
│  │ Ready for Analysis: ✓ YES                                  ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Back]  [Reset]  [Continue to Analysis >]                    │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

**Features:**
- Data quality score gauge
- Issues/warnings table with severity
- Impact analysis for each fix
- Before/after comparison
- Readiness checklist

---

## Part 4: Key UI Components & Interactions

### 4.1 Reusable Components

#### 1. Factor Comparison Chart
```
Interactive visualization comparing two factors' distributions
across good/bad trade groups.
```

#### 2. Scenario Condition Builder
```
Visual builder for creating custom scenarios:
[Factor] [Operator] [Threshold]
insider_sentiment [>] [0.5]
AND tech_momentum_20 [>] [2.0]
AND regime_trend [=] [bullish]
```

#### 3. Statistical Test Results Card
```
Compact card showing:
- Test name and statistic
- P-value with significance indicator
- Effect size
- Sample counts
- Interpretation text
```

#### 4. Feature Importance Bar Chart
```
Horizontal bars showing:
- Feature name
- Importance score
- Percentage contribution
- Color-coded significance
```

### 4.2 Interactive Features

#### Drill-Down Capability
- Click any factor → See detailed analysis (histogram, ANOVA, SHAP)
- Click any scenario → See all trades in that scenario
- Click any trade → See factor breakdown for that trade

#### Export Functions
- **PDF**: Professional formatted reports with charts
- **Excel**: Tabular data with worksheets per section
- **JSON**: Machine-readable format for integrations
- **CSV**: Individual data tables

#### Real-Time Validation
- As user adjusts parameters → Impact preview updates
- As data loads → Quality assessment updates
- As analysis runs → Progress bar with ETA

#### Search & Filter
- Global search across all factors
- Filter scenarios by lift, confidence, trade count
- Filter trades by outcome, date range, symbol
- Filter factors by p-value, correlation, importance

---

## Part 5: Navigation & Information Architecture

### 5.1 Main Flow Diagram
```
1. START
   ├─→ Upload Data
   │   ├─→ Trade Log
   │   ├─→ Price Data
   │   ├─→ Fundamentals
   │   ├─→ Insiders
   │   └─→ Options
   │
   ├─→ Configure Analysis
   │   ├─→ Create/Edit Profile
   │   ├─→ Set Thresholds
   │   ├─→ Set Delays
   │   ├─→ Select Factors
   │   └─→ Statistical Settings
   │
   ├─→ Run Analysis
   │   ├─→ Data Quality Check
   │   ├─→ Tier 1 (Exploratory)
   │   ├─→ Tier 2 (Hypothesis)
   │   ├─→ Tier 3 (ML)
   │   └─→ Scenarios
   │
   ├─→ View Results
   │   ├─→ Data Summary
   │   ├─→ Tier 1 Results
   │   ├─→ Tier 2 Results
   │   ├─→ Tier 3 Results
   │   ├─→ Scenarios
   │   └─→ Interactions
   │
   └─→ Export & Share
       ├─→ PDF Report
       ├─→ Excel Tables
       ├─→ JSON Payload
       └─→ Schedule Delivery
```

### 5.2 Information Hierarchy

**Level 1 (Dashboard Overview)**
- Summary statistics, KPIs, status
- Quick navigation to key analyses
- Recent activities

**Level 2 (Analysis by Tier)**
- Tier-specific results, tables, charts
- Filtering and sorting options
- Drill-down entry points

**Level 3 (Detailed View)**
- Individual factor/scenario breakdown
- Full statistical output
- Trade-level details

---

## Part 6: Mobile & Responsive Design

### 6.1 Responsive Breakpoints
- **Desktop** (1200px+): Full sidebar, multi-column layouts
- **Tablet** (768-1199px): Collapsible sidebar, stacked columns
- **Mobile** (< 768px): Hamburger menu, single column, mobile-optimized charts

### 6.2 Mobile Views
- Trade log preview with swipe-to-see-columns
- Results summary cards (stackable)
- Scenario list with expandable details
- Factor selector with touch-optimized buttons

---

## Summary

This specification defines **three fully-featured GUI applications** that work together to provide a complete interface to the Factor Analysis system:

1. **Factor Analysis Dashboard** - Visualize results, explore factors, define scenarios
2. **Configuration Manager** - Create and manage analysis profiles
3. **Data Upload Interface** - Ingest and validate input data

Each interface is optimized for the analyst workflow: upload → configure → analyze → explore → export.

All GUIs include:
- Real-time validation and feedback
- Comprehensive help and tooltips
- Export in multiple formats
- Professional data visualization
- Audit and reproducibility features
- Responsive, mobile-friendly design

