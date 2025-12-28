### Goal
The idea behind this analysis is to identify the characteristics of the periods and securities that my strategy performs at its best and worst. 
I will be looking at fundamentals for the security at the given time, both in isolation and in relation to the sector where appropriate. I will also be examining the price action on at a high level to help identify these characteristics that will help me differentiate.

Definition of 'best' performers is periods and securities where the win rate is above 45% and bad trades exit early while good trades are left to ride (I will be selecting them manually). Anything worse than this is considered a 'worst' period or security.

**Deliverable:**
What I will be gaining from this project is a better understanding of when a security is in a good period to execute my strategy and a set of rules to follow ,e.g. the 14W RIS smoothed is moving up, Free Cash Flow has been steady and growing, Debt-to-Equity ratio is above the sector average, etc.
### Portfolio of chosen securities

|           | Consumer Discretionary | Materials & Manufacturing | Healthcare | Services | Electronic Technology | Technology Services | Communications | Distribution Services | Energy | Finance | Utilities |
| --------- | ---------------------- | ------------------------- | ---------- | -------- | --------------------- | ------------------- | -------------- | --------------------- | ------ | ------- | --------- |
| **Large** | AMZN                   | CAT                       | LLY        | SPGI     | NVDA                  | GOOG                | TMUS           | MCK                   | XOM    | JPM     | NEE       |
| **Mid**   | ARMK                   | GTLS                      | QGEN       | BZ       | AMKR                  | KVYO                | FYBR           | AIT                   | WCP    | SSB     | AES       |
| **Small** | BLBD                   | SLVM                      | ATRC       | SBET     | PLAB                  | AI                  | IRDM           | CURY                  | PARR   | SYBT    | NWN       |

---

## Phase 1: Backtest Analysis & Data Collection

## 1.1 Backtest Setup & Trade Segmentation

**Objective**: Isolate periods of high vs. low strategy performance with sufficient granularity.

**Steps**:
1. **Run single-security backtests** (existing plan)
    - Daily timeframe, 16-year history per security
2. **Identify performance periods with statistical rigor**:
    - **High-performance threshold**: Calmar Ratio > 0.5 and positive expectancy
    - **Low-performance threshold**: Calmar Ratio < 0.5 and negative expectancy
    - **Indeterminate-performance**: Does not meet high or low performance conditions
    - **Minimum sample size per period**: If a security has <=3 trades in a period, exclude that period (insufficient data for reliable statistics)
    - **Track both outcomes**:
        - Early exits on bad trades (intended behaviour)
        - Trades allowed to ride (intended behaviour)
        - Trades that stop out prematurely
        - Trades that miss the full move
3. **Segment by period for granular analysis**:
    - Option A: Calendar year (e.g., 2015, 2016, etc.) – simpler, identifies secular trends
    - Option B: Trade-by-trade with rolling 12-month windows (more granular but complex)
    - **Recommendation**: Start with calendar year, then drill into anomalous years with rolling windows
4. **Document metadata for each trade**:
    `Date, Security, Win/Loss, Entry Price, Exit Price, Hold Duration,  Weekly RSI (14-SMA), BB Band Position, 52-Week Distance,  Entry Month/Quarter, Market Regime (Bull/Bear/Choppy)`
    
## 1.2 Fundamental Data Collection Framework

**Objective**: Capture the fundamental state of each security at the time of trade initiation (no forward bias).

**Data Timeline Requirement**: All metrics must represent information available to a trader **before or at market open on the trade date**.

**Metrics to Collect** (Organized by Category):

## **A. Earnings & Growth Momentum** 

| Metric                          | Definition                                 | How to Calculate/Find                                       | Sector Comparison                        | Interpretation                                                         |
| ------------------------------- | ------------------------------------------ | ----------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| **EPS (TTM)**                   | Earnings Per Share, trailing twelve months | Most recent 4 quarterly earnings / diluted shares           | Compare to sector median EPS (Damodaran) | Higher EPS relative to sector = stronger quality; negative = distress  |
| **EPS Growth Rate (YoY %)**     | % change in EPS from prior year            | (EPS_current - EPS_prior_year) / \|EPS_prior_year\|         | Sector median EPS growth                 | Positive acceleration = momentum; negative = deceleration              |
| **EPS Surprise/Revision Trend** | Direction of recent earnings surprises     | Track last 2-4 quarters: Beat vs. Miss                      | Peer comparison                          | Consistent beats = positive surprise momentum; misses = reversion risk |
| **Revenue Growth (TTM YoY %)**  | % change in revenue                        | (Revenue_current - Revenue_prior_year) / Revenue_prior_year | Sector median revenue growth             | Quality of earnings driver; fast growth + high margins = strength      |
| **Operating Margin (TTM %)**    | Operating Income / Revenue                 | Operating Income (TTM) / Revenue (TTM)                      | Sector median operating margin           | Higher = better execution; trending up = positive signal               |

**Data Sources**:
- **Free**: Yahoo Finance (Summary tab → Earnings section), Google Finance, MarketWatch
- **More reliable**: SEC Edgar (10-K, 10-Q filings) for official numbers
- **Sector benchmarks**: Use **Aswath Damodaran's free industry data** ([https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html)) – updated January each year
    - Search by industry group; find "Earnings per share" and "Sales/Market Cap" rows for sector medians

**Interpretation Guidance for EPS**:
- **vs. Sector**: If security EPS < sector median, it's underperforming peers → momentum may be negative
- **Trend**: If EPS trending down but price rising = potential dead cat bounce (negative signal for alpha trend)
- **Growth rate**: If YoY EPS growth decelerating (e.g., 20% → 10% → 5%), expect momentum fade

## **B. Valuation Metrics**

| Metric                  | Definition                                  | How to Calculate                      | Sector Comparison         | Interpretation                                                |
| ----------------------- | ------------------------------------------- | ------------------------------------- | ------------------------- | ------------------------------------------------------------- |
| **P/E Ratio (Forward)** | Stock Price / Expected EPS (next 12M)       | Stock price / analyst consensus EPS   | Sector median forward P/E | High P/E = expensive; momentum often precedes P/E contraction |
| **PEG Ratio**           | P/E Ratio / Earnings Growth Rate            | Forward P/E / EPS growth rate (%)     | Sector median PEG         | PEG < 1 = "cheap" relative to growth; > 2 = expensive         |
| **Price/Book**          | Stock Price / Book Value per Share          | Market cap / Shareholders' equity     | Sector median P/B         | Higher = growth stock; momentum favors growth, not distressed |
| **Price/Cash Flow**     | Stock Price / Operating Cash Flow per Share | Stock price / (Operating CF / shares) | Sector median P/CF        | Lower = healthier; high P/CF can signal overvaluation         |

**Data Sources**:
- Yahoo Finance (Statistics tab) – quick reference
- Damodaran data (for sector medians)
- Seeking Alpha (detailed multiples)

**Interpretation Guidance**:
- **For momentum strategies**: Don't exclude high P/E stocks – momentum strategies often ride hot stocks with elevated valuations
- **Red flag**: P/E expanding above sector average while earnings decelerate = unsustainable momentum (avoid entry)
- **Green flag**: P/E at sector average with accelerating earnings = sustainable momentum

## **C. Cash Flow & Financial Health**

| Metric                       | Definition                          | How to Calculate                            | Sector Comparison                 | Interpretation                                                   |
| ---------------------------- | ----------------------------------- | ------------------------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| **Free Cash Flow (TTM, $M)** | Operating CF – Capital Expenditures | From cash flow statement (10-K)             | Total FCF / market cap for sector | Positive FCF = sustainable; negative = funding via debt/dilution |
| **FCF Trend (YoY % change)** | % change in FCF year-over-year      | (FCF_current - FCF_prior) / FCF_prior       | Sector median FCF growth          | Positive trend = strength; flat/declining = concern              |
| **FCF Yield (%)**            | Free Cash Flow / Market Cap         | (FCF / Market cap) × 100                    | Sector median FCF yield           | Higher = more cash generated; compare to sector                  |
| **Debt-to-Equity Ratio**     | Total Debt / Shareholders' Equity   | (Short-term debt + Long-term debt) / Equity | Sector median D/E                 | Elevated D/E + weak earnings = default risk                      |

**Data Sources**:
- Yahoo Finance (Cash Flow tab for FCF)
- Company 10-K filings
- Damodaran (sector medians for D/E, FCF yield)

**Interpretation Guidance for Cash Flow**:
- **Negative FCF + high growth**: May be acceptable (e.g., Tesla in early years), but is riskier
- **FCF declining while EPS rising**: Red flag – earnings quality deteriorating
- **Strong FCF + rising D/E**: Company reinvesting aggressively (potentially positive) vs. over-leveraging (negative)

---

## Phase 2: Technical Analysis & Market Regime

## 2.1 Technical Metrics Collection

**Existing Plan Enhancements**:

|Metric|Lookback|Calculation|Why It Matters for Momentum|
|---|---|---|---|
|**14-Week RSI (SMA smoothed)**|14 weeks|Standard RSI(14), then apply 5-period SMA|Existing plan – keep as-is|
|**52-Week High/Low Distance**|52 weeks|(Current Price - 52W Low) / (52W High - 52W Low)|0.5 = midpoint; >0.75 = upper half (bullish); <0.25 = lower half (bearish)|
|**Bollinger Bands Position**|20 periods, 2 std dev|(Price - SMA20) / (2 × StdDev20)|-1 to +1 scale; >0.5 = upper band approach; touch/exceed = mean reversion risk|
|**BB Band Width (%)**|20 periods, 2 std dev|(Upper_BB - Lower_BB) / Middle_SMA × 100|Narrow <10% = low volatility (quiet before storm); Wide >30% = high volatility|
|**Weekly Volume Trend**|50-week MA|Compare current weekly volume to 50-week MA|Volume surge on breakouts = confirmation; declining volume = weak move|

**NEW METRICS – Relative Strength & Regime Detection**:

|Metric|Definition|How to Calculate|Interpretation|
|---|---|---|---|
|**Relative Strength (Stock vs. Sector)**|Stock price / Sector ETF price|Normalize both to same start date, plot ratio|Ratio rising = stock outperforming sector; falling = underperforming|
|**RS Trend (50-week MA)**|50-period SMA of RS ratio|Plot 50-week SMA of (Stock / Sector ETF)|Uptrend = momentum positive; downtrend = momentum fading|
|**Market Regime (Bull/Bear/Choppy)**|Macro assessment|SPY 200-week MA: Price > 200MA = Bull; < 200MA = Bear|Momentum strategies work best in Bull/choppy; weakest in Bear markets|
|**ATR-based Volatility (14-week)**|Average True Range, 14 weeks|Standard ATR(14) calculation|High ATR = high volatility; low ATR = quiet market. Time entries in low ATR (lower noise)|

**Data Sources**:

- TradingView (your platform – has RSI, BB, ATR built-in)
- Yahoo Finance (historical OHLCV for manual calculation)
- Sector ETFs: Use appropriate ETF for comparison
    - Consumer Discretionary: XLY
    - Technology: XLK
    - Finance: XLF
    - Healthcare: XLV
    - Energy: XLE
    - Industrials: XLI
    - Materials: XLB
    - Utilities: XLU
    - Communications: XLC
    - Consumer Staples: XLP (if applicable to your portfolio)

## 2.2 Volatility Timing Framework

**Key Insight from Research**: Momentum strategy alpha is driven ~2/3 by market timing, ~1/3 by volatility timing.

**Collect**:
- **VIX Level (at trade entry)**:
    - < 12 = Unusually calm (not ideal for breakouts; tight ranges)
    - 12-18 = Normal (ideal for momentum)
    - 18-25 = Elevated volatility (can be opportunities if directional)
    - > 25 = Panic (momentum strategies struggle; reversals common)

- **Weekly ATR as % of price**:
    - Identify if volatility is contracting or expanding at entry
    - Contraction before breakout = ideal entry (volatility about to expand)
    - Expansion at entry = already in momentum (late entry, more risk)

---

## Phase 3: Comparative Analysis & Pattern Identification

## 3.1 Fundamental Pattern Analysis

**Step 1: Group High vs. Low Performance Periods**
For each security:
- Create **two groups** of trades/periods
    - Group A: High-performance periods (Win rate > 45%)
    - Group B: Low-performance periods (Win rate < 45%)

**Step 2: Statistical Comparison**

For each fundamental metric, calculate and compare:

|Statistic|Definition|How to Use|
|---|---|---|
|**Mean**|Average value in group|Identifies central tendency|
|**Median**|Middle value|Less sensitive to outliers|
|**Std Dev**|Spread of values|Identifies consistency vs. volatility|
|**25th/75th Percentile**|Quartile range|Shows dispersion|

**Step 3: Identify Significant Differences**

- **Simple method**: If Group A median is 2x Group B median, it's likely meaningful
- **Statistical method** (more rigorous): Run t-tests or Mann-Whitney U tests
    - p-value < 0.05 = statistically significant difference
    - Example: "EPS YoY growth is significantly higher in high-performing periods (p=0.03)"

**Example Output Table**:
`Metric: EPS YoY Growth (%) --- High-Performance Periods:   Mean: 18.5%, Median: 16.2%, Std Dev: 8.3% Low-Performance Periods:   Mean: 3.2%, Median: 1.8%, Std Dev: 11.2% Conclusion: High-performance periods have 5x higher EPS growth.  Soft rule: Avoid entry if EPS growth < 5% YoY.`

## 3.2 Technical Pattern Analysis

**Repeat Phase 3.1 methodology for technical metrics**:
- 52-week distance (where in the cycle?)
- RSI levels (overbought entries vs. oversold bounces?)
- BB band position (mean-reversion vs. breakout?)
- Volume patterns (confirmation or divergence?)
- Relative strength trend (stock vs. sector trending up/down?)

---

## Phase 4: Statistical Correlation & Predictive Metric Ranking

## 4.1 Identify Which Metrics Actually Predict Success

**Objective**: Narrow 10+ metrics down to 2-3 that most strongly predict high win rates.

**Methodology**:
1. **Create a dataset** with all trades:
    - Rows = Each trade across all securities
    - Column 1: Outcome (Win=1, Loss=0) or Win_Rate_Period (as %)
    - Columns 2-20: Each fundamental/technical metric at trade entry
2. **Calculate correlation coefficients** (Pearson or Spearman):
  `For each metric: Correlation with Win Rate Example: - EPS_YoY_Growth ↔ Win Rate: r = +0.42 (strong positive) - PE_Ratio ↔ Win Rate: r = -0.08 (weak negative) - RSI_14 ↔ Win Rate: r = +0.15 (weak positive)`
    
3. **Run multiple regression** (if comfortable with stats):
    - Dependent variable: Win_Rate (%)
    - Independent variables: Top 6-8 fundamental metrics
    - Output: Coefficient (importance weight) + p-value (significance)
    - **Interpret**: Metrics with p < 0.05 are statistically significant
        
4. **Check for multicollinearity**:
    - Calculate Variance Inflation Factor (VIF) for each metric
    - If VIF > 5, two metrics are too correlated; keep the one with higher correlation to win rate

5. **Rank metrics by predictive power**:
    `Rank | Metric | Correlation to Win Rate | Significance (p-value) 1    | EPS_YoY_Growth | +0.42 | p=0.008 ✓ 2    | FCF_Trend | +0.38 | p=0.015 ✓ 3    | RS_Trend_50MA | +0.35 | p=0.022 ✓ 4    | OpMargin_TTM | +0.22 | p=0.089 (weak) 5    | PE_Forward | -0.08 | p=0.41 (not significant)`
    

**Recommendation**: Focus hard rules on top 3 metrics; use others for context/soft rules.

---

## Phase 5: Final Synthesis & Rule Development

## 5.1 Hard Rules (Backtestable, Mechanical)

Based on Phase 4 ranking, develop rules using top 3-5 predictive metrics.

**Example Output**:
`HARD RULES FOR ENTRY CONSIDERATION: 1. EPS YoY Growth > 10% 2. EPS_Surprise_Trend = 2+ consecutive beats (last 2 quarters) 3. Relative_Strength_50MA = Uptrend (RS ratio above 50-week MA) 4. VIX < 25 (avoid panic environments) 5. 52-Week Distance > 0.3 (avoid prices at 52-week lows) If all 5 conditions met: Green light for entry If 3-4 conditions met: Yellow light (entry OK if technicals strong) If <3 conditions met: Red light (pass)`

## 5.2 Soft Rules (Contextual, Decision-Making)

Metrics that add confidence but don't trigger mechanical filters.

**Example Output**:
`SOFT RULES FOR CONFIDENCE BOOST: - EPS growth accelerating (e.g., 12% → 15% → 18%): Add 5% confidence - D/E below sector median: Add 3% confidence - FCF positive and growing: Add 3% confidence - Stock outperforming sector by >15% (RS ratio > 1.15): Add 5% confidence - BB band in upper half (not touching upper band): Add 2% confidence Use these to calibrate position size or risk management.`

## 5.3 Avoid Rules (When NOT to trade)

Identify conditions when your strategy historically failed.

**Example Output** (from your historical data):
`AVOID ENTRY WHEN: - EPS growth < 0% (earnings contraction) - Company just reported earnings miss (surprise negative) - Sector underperforming market by >20% for 6+ months - Stock at 52-week low (bottom-fishing often reverses) - Volatility regime shifting (ATR spiking, price gap reversals) - Financial distress signals: FCF negative for 2+ quarters, D/E rising >30%`

---

## Phase 6: Sector & Size-Specific Insights

## 6.1 Sector Variation Analysis

**After completing Phase 1-5, drill down by sector**:
- Does your strategy work better in certain sectors?
- Example: Tech momentum > Energy momentum?
- Why? (Sector characteristics: beta, liquidity, analyst coverage, cyclicality)

## 6.2 Cap-Size Effects

**Does performance vary by market cap?**
- Large-cap: More analyst coverage, more efficient pricing, slower momentum?
- Mid-cap: Sweet spot for alpha?
- Small-cap: Higher volatility, wider spreads, slower fill execution?

---

## Key Data Source Summary

|Data|Primary Source|Secondary|Frequency|
|---|---|---|---|
|**Fundamental Metrics**|SEC Edgar (10-K, 10-Q)|Yahoo Finance|Quarterly|
|**Sector Benchmarks**|Damodaran (NYU Stern)|Manually calculated from top peers|Annual (updated Jan)|
|**Technical Data**|TradingView / Yahoo Finance|OHLCV|Daily|
|**Sector ETF Prices**|Yahoo Finance / TradingView|IEX Cloud|Daily|
|**VIX / Volatility**|CBOE (via Yahoo/TradingView)|FRED|Daily|

---

## Critical Success Factors

1. **Time-series consistency**: All metrics captured at trade initiation, not retrospectively
    
2. **Sufficient sample size**: Minimum 30-50 trades per security for reliable statistics
    
3. **Sector benchmark accuracy**: Use official Damodaran data when possible; validate manual calculations
    
4. **Backtesting rigor**: Exclude look-ahead bias, account for data availability at trade time
    
5. **Statistical literacy**: Don't over-fit to noise; focus on metrics with p < 0.05 significance
    
6. **Sector variation awareness**: Rules may need adjustment by sector; document exceptions
    

---
