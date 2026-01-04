# Per-Trade Analysis Feature - Detailed Specification
## Backtesting Framework Enhancement

**Date Created:** January 4, 2026  
**Target Framework:** Python-based backtesting system  
**Integration Point:** Post-backtest analysis tool  
**Primary User:** Quantitative trader performing deep-dive trade forensics  

---

## 1. EXECUTIVE SUMMARY

This feature enables granular analysis of individual trades to understand success/failure patterns. The system will:

1. **Load multiple trade log CSV files** from different backtests
2. **Automatically select representative trade samples** across outcome distributions
3. **Aggregate multi-source data** (price action, indicators, fundamentals, insider activity, options, correlations)
4. **Generate interactive GUI** with tabbed analysis views
5. **Export comprehensive Excel reports** with detailed trade forensics
6. **Identify patterns** in winners vs losers with statistical flagging

---

## 2. DATA INPUT SPECIFICATION

### 2.1 Trade Log CSV Format (Required)

**Source:** Output from backtesting engine

**Required Columns:**
```
trade_id, symbol, entry_date, entry_price, exit_date, exit_price, quantity, 
side, initial_stop_loss, final_stop_loss, take_profit, pl, pl_pct, 
security_pl, fx_pl, entry_fx_rate, exit_fx_rate, security_currency, 
duration_days, entry_equity, entry_capital_available, entry_capital_required, 
concurrent_positions, competing_signals, entry_reason, exit_reason, 
commission_paid, partial_exits
```

**Data Types:**
- Dates: YYYY-MM-DD format
- Prices/Rates: Float (4+ decimal places)
- PL values: Float (supports negative)
- Reasons: String/categorical
- Side: "LONG" or "SHORT"

**Validation Rules:**
- No null values in core columns (trade_id through duration_days)
- exit_date >= entry_date
- exit_price and pl must be consistent with entry_price and side
- All dates must be within historical data range

### 2.2 Raw Data Structure (Required)

**Location:** `raw_data/` directory

**Daily Price Data:**
```
raw_data/daily/{SYMBOL}_daily.csv
Columns: date, open, high, low, close, volume, [indicator_values...]
Format: YYYY-MM-DD, float values
Indicators: ATR, RSI, BB_Upper, BB_Lower, BB_Middle, SMA_20, SMA_50, SMA_200, 
            MACD, MACD_Signal, MACD_Histogram, [any custom indicators]
```

**Weekly Price Data:**
```
raw_data/weekly/{SYMBOL}_weekly.csv
Columns: date, open, high, low, close, volume, [indicator_values...]
Format: Same structure, but weekly aggregation
```

**Sector/Index Data:**
```
raw_data/daily/{SECTOR_NAME}_daily.csv
Example: raw_data/daily/XLK_daily.csv, raw_data/daily/SPX_daily.csv
Purpose: Calculate correlation to broader market
Columns: date, open, high, low, close, volume
```

**Optional Data Files:**
```
raw_data/fundamentals/{SYMBOL}_fundamentals.csv
Columns: date, pe_ratio, eps_growth_pct, revenue_growth_pct, 
         debt_to_equity, free_cash_flow, roa, roe, current_ratio
Format: One row per quarter/annual update, most recent value carried forward

raw_data/insider/{SYMBOL}_insider.csv
Columns: transaction_date, insider_name, transaction_type, shares_traded, 
         transaction_value, form4_date, insider_role
Format: YYYY-MM-DD, transaction_type = BUY/SELL

raw_data/options/{SYMBOL}_options.csv
Columns: date, implied_volatility, put_call_ratio, option_flow_signal, 
         days_to_earnings
Format: Daily snapshot, flow_signal = -1 (bearish)/0 (neutral)/1 (bullish)
```

### 2.3 Metadata Configuration File (Required)

**File:** `config/trade_analysis_metadata.json`

**Purpose:** Define available data and validation

**Structure:**
```json
{
  "available_symbols": {
    "AAPL": {
      "has_daily_prices": true,
      "has_weekly_prices": true,
      "has_fundamentals": true,
      "has_insider_data": true,
      "has_options_data": false,
      "sector_ticker": "XLK",
      "index_ticker": "SPX",
      "primary_currency": "USD",
      "data_start_date": "2009-01-01",
      "data_end_date": "2026-01-04"
    },
    "MSFT": {
      "has_daily_prices": true,
      "has_weekly_prices": true,
      "has_fundamentals": true,
      "has_insider_data": true,
      "has_options_data": true,
      "sector_ticker": "XLK",
      "index_ticker": "SPX",
      "primary_currency": "USD",
      "data_start_date": "2009-01-01",
      "data_end_date": "2026-01-04"
    }
  },
  "thresholds": {
    "insider_activity_min_value_usd": 50000,
    "insider_buying_lookback_days": 30,
    "insider_selling_lookback_days": 30,
    "correlation_threshold_significant": 0.70,
    "market_regime_sma_period_short": 20,
    "market_regime_sma_period_long": 200,
    "iv_percentile_threshold": 75,
    "extreme_trade_pl_percentile": 95
  },
  "fundamental_metrics": [
    "pe_ratio", "eps_growth_pct", "revenue_growth_pct", 
    "debt_to_equity", "free_cash_flow", "roa", "roe"
  ],
  "indicator_columns": [
    "ATR", "RSI", "BB_Upper", "BB_Lower", "BB_Middle", 
    "SMA_20", "SMA_50", "SMA_200", "MACD", "MACD_Signal", "MACD_Histogram"
  ]
}
```

---

## 3. CORE FUNCTIONALITY

### 3.1 Trade Log Ingestion & Validation

**Input:** User selects one or more trade log CSV files via file dialog

**Processing:**
1. Load each CSV file
2. Validate column presence and data types
3. Concatenate multiple files (with source file tracking)
4. Remove duplicates by trade_id
5. Filter out "extreme good trades" (trades in top 5% by pl_pct)
6. Sort by exit_date descending

**Output:** Validated DataFrame with metadata

**Error Handling:**
- Alert user if columns missing
- Alert user if date range inconsistencies
- Suggest metadata.json updates if symbols not in config

---

### 3.2 Representative Trade Sampling

**Objective:** Automatically select statistically representative sample across outcome distributions

**Algorithm:**

**Step 1: Outcome Distribution Analysis**
- Calculate trade outcome percentiles (0, 25, 50, 75, 100)
- Define outcome buckets:
  - **Losers (Bottom 25%):** pl_pct < 25th percentile
  - **Breakeven (25-75%):** 25th percentile <= pl_pct <= 75th percentile
  - **Winners (Top 25%):** pl_pct > 75th percentile

**Step 2: Exit Type Stratification**
- Within each outcome bucket, further stratify by exit_reason:
  - Target Hit
  - Stop Loss Hit
  - Manual Exit
  - Time-Based Exit
  - Other (as needed)

**Step 3: Sample Selection**
- For each outcome bucket + exit type combination:
  - If count >= 3: Select 1 trade (proportional random selection favoring median performers)
  - If count = 2: Select 1 trade
  - If count = 1: Select that trade
- **Target total:** 8-15 representative trades

**Step 4: Diversity Enforcement**
- Ensure sample spans at least 2 different symbols (if available)
- Ensure sample spans at least 2 different time periods (earliest to latest)
- Ensure sample includes both LONG and SHORT sides (if available)

**Output:** List of selected trade_ids with rationale for each selection

**UI Component:**
- Display selected trades before analysis
- Allow user to manually add/remove trades
- Show why each was selected

---

### 3.3 Data Aggregation Pipeline

**For each selected trade, aggregate:**

#### A. Price Action Data
- **Pre-entry window:** 1 year or 3 months (user selectable) before entry_date
- **Entry day:** entry_date ± 2 trading days
- **Post-entry to exit:** Full trade duration
- **Post-exit window:** 30 calendar days after exit_date
- **Output:** DataFrame with daily and weekly OHLCV + all indicators

#### B. Fundamental Data
- **At entry_date:** Most recent fundamental snapshot before/on entry date
- **At exit_date:** Most recent fundamental snapshot before/on exit date
- **Comparison:** Calculate deltas for each metric
- **Output:** Dict of entry_fundamentals, exit_fundamentals, delta_fundamentals

#### C. Insider Activity
- **Lookback window:** 1 year before entry_date to 30 days after exit_date
- **Aggregation:** All transactions for symbol, with insider details
- **Flagging Rules:**
  - Flag if insider BUYING >= $50k in 30 days before entry (bullish signal)
  - Flag if insider SELLING >= $50k in 30 days before exit (bearish signal)
  - Flag if insider BUYING immediately after exit (suggest timing validation)
  - Flag large concentrated transactions (>5% of insider's holdings)
  - Flag coordinated activity (multiple insiders in same period)
- **Output:** List of transactions with flags, timeline mapping to trade dates

#### D. Options Data (if available)
- **At entry_date:** IV level, put/call ratio, flow signal, days to earnings
- **During trade:** IV trend (increasing/decreasing), major flow events
- **At exit_date:** IV level, put/call ratio
- **Analysis Flags:**
  - High IV at entry (>75th percentile) → interpret explosive reversals?
  - Low IV at entry (<25th percentile) → interpret missed moves?
  - Earnings announcement during trade → was catalyst handled?
- **Output:** Dict with IV history, flow events, earnings dates

#### E. Sector/Market Correlation
- **Sector:** Fetch sector ticker (from metadata) daily prices
- **Index:** Fetch index ticker (SPX) daily prices
- **Calculation:** Pearson correlation(symbol_returns, sector_returns) and correlation(symbol_returns, index_returns) for:
  - Pre-entry period (same window as price action)
  - Trade duration
  - Post-exit period
- **Analysis Flags:**
  - Trade direction aligned with sector? (both up/down = correlated)
  - Trade direction aligned with index?
  - Leading/lagging? (did stock move first or follow sector?)
- **Output:** Correlation matrix, trend analysis

#### F. Entry Signal Strength Analysis
- **Calculate metrics at entry_date:**
  - Distance from SMA_20, SMA_50, SMA_200 (% off each)
  - RSI value (overbought/oversold/neutral)
  - ATR value (volatile/stable relative to history)
  - Bollinger Band position (upper/middle/lower third)
  - Volume profile (above/below average)
  - MACD status (positive/negative, crossover?)
  - Trend alignment (daily trend up/down/ranging, weekly trend, 200-SMA direction)
- **Signal Strength Scoring:** 0-100 scale based on confluence
  - Perfect confluence (all factors aligned) = 80-100
  - Good confluence (4-5 factors) = 60-79
  - Moderate (2-3 factors) = 40-59
  - Weak (<2 factors) = 0-39
- **Output:** Signal strength score + breakdown

#### G. MAE/MFE Analysis
- **MAE (Maximum Adverse Excursion):** Worst price during trade vs entry
  - Calculate: |(worst_price - entry_price) / entry_price * 100|
  - For LONG: Use low prices
  - For SHORT: Use high prices
  - Identify: Date when MAE occurred
- **MFE (Maximum Favorable Excursion):** Best price during trade vs entry
  - Calculate: |(best_price - entry_price) / entry_price * 100|
  - For LONG: Use high prices
  - For SHORT: Use low prices
  - Identify: Date when MFE occurred
- **Output:** MAE %, MFE %, dates of each, comparison to actual exit price

#### H. Market Regime Detection
- **For pre-entry and entry date:**
  - Compare SMA_20 > SMA_50 > SMA_200 → Uptrend
  - Compare SMA_20 < SMA_50 < SMA_200 → Downtrend
  - Range-bound if SMA_20 oscillates around SMA_50
  - Calculate daily volatility (std dev of returns) vs 60-day average
  - Flag: High volatility (>1.5x avg) or Low volatility (<0.66x avg)
- **Output:** Market regime label + volatility characterization

**Data Aggregation Output Structure:**
```python
{
    'trade_id': str,
    'symbol': str,
    'price_data': DataFrame,  # OHLCV + indicators, pre/during/post
    'fundamentals': {
        'entry': dict,
        'exit': dict,
        'deltas': dict
    },
    'insider_activity': DataFrame,
    'insider_flags': list,
    'options_data': dict,  # if available
    'correlation': {
        'sector_correlation': float,
        'index_correlation': float,
        'analysis': str
    },
    'entry_signal_strength': {
        'score': float,  # 0-100
        'breakdown': dict
    },
    'mae_mfe': {
        'mae_pct': float,
        'mfe_pct': float,
        'mae_date': date,
        'mfe_date': date
    },
    'market_regime': str,
    'volatility_regime': str,
    'data_quality': {
        'missing_fundamentals': bool,
        'missing_insider_data': bool,
        'missing_options_data': bool,
        'warnings': list
    }
}
```

---

## 4. GUI SPECIFICATION

### 4.1 Technology Choice: Streamlit

**Rationale:**
- Web-based (no installation complexity)
- Rapid iteration
- Built-in data display (tables, metrics)
- Interactive multiselect and filters
- Export-friendly

**Alternative considered (rejected):** PyQt (more complex, slower iteration, overkill for analysis tool)

### 4.2 Layout & Tabs

**Main Screen: Trade Selection Panel**
- **Multi-file upload:** Accept multiple CSV files
- **Sidebar Controls:**
  - Symbol filter (multi-select)
  - Side filter (LONG/SHORT)
  - Outcome filter (Winners/Losers/Breakeven)
  - Date range slider
  - "Auto-Select Representative Sample" button
  - "Refresh Data" button
- **Main area:** 
  - Summary stats (total trades, filtered count, outcome distribution)
  - Table showing all filtered trades (sortable)
  - Selected trades highlighted
  - "Add/Remove manually" button for each row

**Tab 1: Trade Summary**
- Key metrics displayed as large cards:
  - P&L (absolute and %)
  - Duration (days)
  - Entry/Exit prices
  - Side and quantity
  - Max Adverse Excursion (MAE) %
  - Max Favorable Excursion (MFE) %
  - Entry signal strength score (0-100 gauge)
- Exit reason displayed prominently
- Entry reason displayed
- Position sizing context (capital required, equity at entry)
- FX impact (if applicable)

**Tab 2: Price Action & Technicals**
- **Sections:**
  - Pre-entry technical snapshot (table)
    - Columns: ATR, RSI, BB position, SMA distances, MACD, Volume
    - Show 1-month, 3-month, 1-year pre-entry snapshot
  - Price action summary
    - Entry price vs SMA_20/50/200 (% distance)
    - Trend alignment (200-SMA direction, weekly trend)
    - Volume profile (entry bar volume vs 20-day avg)
  - Market regime at entry
    - Uptrend/Downtrend/Ranging indicator
    - Volatility regime (High/Normal/Low)
- **Warnings:** Flag unusual technicals (RSI extreme, ATR spike, etc.)

**Tab 3: Fundamentals**
- **Side-by-side comparison table:**
  - Columns: Metric | Entry Value | Exit Value | Change | Health Status
  - Rows: P/E, EPS Growth %, Revenue Growth %, Debt/Equity, FCF, ROA, ROE
  - Color code: Red for deterioration, Green for improvement, Gray for unknown
- **Narrative summary:**
  - "Company fundamentals at entry: [description]"
  - "Company fundamentals at exit: [description]"
  - "Overall assessment: [improved/stable/deteriorated]"
- **Missing data alert:** If fundamentals not available, flag prominently

**Tab 4: Insider Activity**
- **Timeline visualization:**
  - Horizontal line representing trade duration
  - Markers for insider transactions (BUY = green up arrow, SELL = red down arrow)
  - Marker size = transaction size (larger = bigger)
  - Hover shows: insider_name, role, amount, date
  - Trade entry/exit marked on timeline
- **Transaction table:**
  - Columns: Date | Insider Name | Role | Type | Shares | Value ($) | Form 4 Date | Flags
  - Sortable by column
  - Color code rows: Green for BUY, Red for SELL
- **Pattern analysis:**
  - "Insider buying in 30 days pre-entry: YES/NO ($ amount)"
  - "Insider selling in 30 days post-exit: YES/NO ($ amount)"
  - "Coordinated activity (multiple insiders): YES/NO"
  - "Recommendation: [interpretation]"
- **Missing data alert:** If no insider data available

**Tab 5: Options Market**
- **IV Analysis:**
  - IV at entry, at exit, at peak/trough during trade
  - IV percentile (is 45th, 75th, etc. percentile?)
  - Put/Call ratio at entry
  - Earnings date (if within trade window) with days_to_earnings
- **Flow signals:**
  - Timeline of major option flow events during trade
  - Bullish/neutral/bearish signals
- **Interpretation:**
  - "High IV at entry: Expected volatility decline helped/hurt?"
  - "Earnings during trade: Were you holding through catalyst?"
- **Missing data alert:** If no options data available (very prominent)

**Tab 6: Sector & Market Correlation**
- **Correlation summary cards:**
  - Sector correlation: [coefficient] with interpretation (strong/moderate/weak, positive/negative)
  - Index (SPX) correlation: [coefficient] with interpretation
- **Price comparison table:**
  - 3 columns: Symbol | Sector | Index
  - Rows: Pre-entry return %, Trade period return %, Post-exit return %
- **Analysis:**
  - "Your trade was [aligned/counter] to sector movement"
  - "Sector led your stock by [X days]" or "Your stock led sector"
  - "Could you have traded the sector instead for better risk/reward?"
- **Visualization:** 3-line chart (if Streamlit supports, else table)

**Tab 7: MAE/MFE Analysis**
- **Cards displaying:**
  - MAE: [%] on [date], [days into trade]
  - MFE: [%] on [date], [days into trade]
  - Actual P&L: [%]
  - Ratio: MFE/MAE (did market reward you proportionally?)
  - Ratio: Actual PL / MFE (what % of max opportunity did you capture?)
- **Interpretation:**
  - "You allowed [X%] adverse move before hitting target"
  - "You captured [Y%] of the maximum favorable excursion"
  - "Could tighter stops have improved risk/reward?"

**Tab 8: Pattern Analysis (Summary)**
- **Aggregated across all selected trades:**
  - Win rate by entry signal strength (bucket: 0-39, 40-59, 60-79, 80-100)
  - Win rate by market regime (Uptrend/Downtrend/Ranging)
  - Win rate by insider activity (With buying signal / Without)
  - Win rate by correlation alignment (With sector / Against sector)
  - Average MAE by outcome (winners vs losers)
  - Average duration by outcome
  - "These are the characteristics of your successful trades..."

---

### 4.3 Data Quality Warnings

**On any tab, if data missing, display:**
```
⚠️  MISSING DATA: Fundamentals not available for this symbol
    The above analysis cannot be completed. Check config/trade_analysis_metadata.json
```

**Location:** Streamlit warning box at top of affected tab

**Severity levels:**
- 🔴 **Critical:** Can't proceed (e.g., price data missing) → disable tab
- 🟡 **Warning:** Optional data missing (e.g., options) → show warning, allow viewing other data
- 🔵 **Info:** Minor limitation (e.g., insider data sparse) → note but don't block

---

## 5. EXCEL EXPORT SPECIFICATION

### 5.1 Report Structure

**File naming:** `{symbol}_{trade_id}_analysis_{export_date}.xlsx`

**Sheets (tabs):**

#### Sheet 1: Trade Summary
**Single-page overview**
```
TRADE ANALYSIS REPORT
Symbol: AAPL | Trade ID: T_001234
Exit Date: 2024-12-15 | Duration: 18 days | Side: LONG

POSITION METRICS                         TRADE OUTCOME
Entry Price: $150.25                     P&L: $2,450 (+3.25%)
Exit Price: $155.10                      Entry Signal Strength: 72/100
Quantity: 1,000 shares                   Exit Reason: Target Hit
Entry Equity: $500,000                   Exit Type: Take Profit

RISK METRICS                             TECHNICAL CONTEXT
Initial Stop: $148.50                    MAE: 1.2% on 2024-12-01
Final Stop: $149.00                      MFE: 4.8% on 2024-12-10
Take Profit: $160.00                     Market Regime: Uptrend
Max Adverse Excursion: 1.2%              Volatility: Normal

KEY INSIGHTS
Entry: Pullback to 50-SMA in uptrend, RSI 55, volume 15% above average, insider buying noted 
$250k in past 30 days.
Exit: Target reached after insider selling ($180k 5 days post-exit).
Correlation: +0.75 with sector (XLK up 2.1%, your trade up 3.25%).

RECOMMENDATION: Strong technical setup + insider bullish signal = predictive pattern identified.
Consider: Can you replicate this setup?
```

#### Sheet 2: Price Action
**Daily prices with technical indicators (pre-entry to post-exit)**
```
Columns: Date | Open | High | Low | Close | Volume | ATR | RSI | 
         BB_Upper | BB_Middle | BB_Lower | SMA_20 | SMA_50 | SMA_200 | 
         MACD | MACD_Signal | Entry_Marker | Exit_Marker

Rows: 1 year (or 3 months) before entry through 30 days after exit
Conditional formatting: Green for entry row, Red for exit row
Entry price highlighted, stop loss levels marked, take profit level marked
```

#### Sheet 3: Weekly Prices
**Same structure as Sheet 2, but weekly aggregation**

#### Sheet 4: Fundamentals
```
Columns: Metric | Entry Date Value | Exit Date Value | Change ($) | Change (%) | 
         At_Entry_Status | At_Exit_Status

Rows: P/E Ratio, EPS Growth %, Revenue Growth %, Debt/Equity, 
      Free Cash Flow ($M), ROA (%), ROE (%)

Color coding: Green (improving), Red (declining), Gray (unavailable)
Add note rows: "Earnings reported between entry and exit: Yes/No"
               "Dividend ex-date during trade: Yes/No"
```

#### Sheet 5: Insider Activity
```
Columns: Date | Insider Name | Role | Type (BUY/SELL) | Shares Traded | 
         Transaction Value ($) | Form 4 Filing Date | 
         Days Before Entry | Days After Exit | Flag

Rows: All transactions in 1-year lookback + 30 days post-exit
Color: Green for BUY, Red for SELL
Summary rows at top:
  - "Insider buying 30 days pre-entry: $X (Y transactions)"
  - "Insider selling 30 days post-exit: $X (Y transactions)"
  - "Coordinated activity detected: Yes/No"
```

#### Sheet 6: Options Market (if available)
```
Columns: Date | IV | IV Percentile | Put/Call Ratio | Flow Signal | 
         Days to Earnings | Notes

Rows: Entry date, exit date, key dates with flow events
Summary:
  - "IV at entry: 35 (45th percentile) → Low"
  - "IV at exit: 38 (62nd percentile) → Normal"
  - "Earnings during trade: Yes/No (days away)"
```

#### Sheet 7: Sector/Market Correlation
```
Columns: Period | Symbol Return % | Sector Return % | Index Return % | 
         Symbol-Sector Correlation | Symbol-Index Correlation

Rows: Pre-entry, During Trade, Post-exit
Summary analysis:
  - "Alignment: Your trade moved WITH sector (both +2.1%)"
  - "Correlation strength: 0.75 (strong positive)"
  - "Leading/lagging: Sector led by 3 days"
```

#### Sheet 8: MAE/MFE Details
```
Maximum Adverse Excursion (MAE)
  Date Occurred: 2024-12-01
  Worst Price: $148.40
  MAE %: -1.23%
  Days into Trade: 2

Maximum Favorable Excursion (MFE)
  Date Occurred: 2024-12-10
  Best Price: $160.50
  MFE %: +6.72%
  Days into Trade: 13

Actual Exit
  Exit Price: $155.10
  Actual PL %: +3.25%
  Captured % of MFE: 48.4% (actual / max opportunity)

Analysis:
  - You weathered a 1.23% drawdown before hitting your target
  - You exited with 48.4% of the available upside in that move
  - Consider: Could you have held longer? Risk was 1.23%, reward was 6.72%
```

#### Sheet 9: Market Regime & Signal Strength
```
ENTRY TECHNICAL SETUP SNAPSHOT

One Month Pre-Entry:
  - Trend (SMA alignment): Uptrend (SMA_20 > SMA_50 > SMA_200)
  - Price to SMA_20: +0.5%
  - Price to SMA_50: +1.2%
  - Price to SMA_200: +3.4%
  - RSI: 55 (neutral)
  - ATR (20-day): $1.80 (normal)
  - Bollinger Bands: Trading middle third (balanced)
  - Volume: 120% of 20-day average
  - MACD: Positive, above signal line (bullish)

Signal Strength Scoring: 72/100
Breakdown:
  ☑ Trend aligned (uptrend): +25 pts
  ☑ Volume above average: +15 pts
  ☑ MACD bullish: +15 pts
  ☑ Price in middle of Bollingers: +12 pts
  ☐ RSI extreme (overbought/oversold): 0 pts
  ☐ Moving average crossover: 0 pts
  
Overall Assessment: GOOD SETUP - Multiple confluences, reasonable entry
```

#### Sheet 10: Pattern Summary (if comparing multiple trades)
```
AGGREGATE ANALYSIS ACROSS ALL SELECTED TRADES

Win Rates by Signal Strength:
  80-100 (Excellent): 67% win rate (4/6 trades)
  60-79 (Good): 50% win rate (3/6 trades)
  40-59 (Moderate): 33% win rate (2/6 trades)
  0-39 (Weak): 25% win rate (1/4 trades)

Win Rates by Market Regime:
  Uptrend: 60% win rate (6/10 trades)
  Downtrend: 40% win rate (2/5 trades)
  Ranging: 30% win rate (1/3 trades)

Win Rates by Insider Activity:
  With insider buying signal: 75% win rate (3/4 trades)
  Without insider activity: 40% win rate (6/15 trades)

Win Rates by Sector Correlation:
  Aligned with sector: 55% win rate (6/11 trades)
  Counter to sector: 35% win rate (3/9 trades)

Average MAE by Outcome:
  Winners: 0.8%
  Losers: 2.1%
  
Average MFE by Outcome:
  Winners: 3.5%
  Losers: 2.2%

Average Trade Duration:
  Winners: 16 days
  Losers: 8 days
```

### 5.2 Formatting Standards

- **Font:** Calibri 11pt, headers 12pt bold
- **Colors:**
  - Header row: Dark blue background, white text
  - Positive values: Green
  - Negative values: Red
  - Neutral/warnings: Yellow
  - Missing data: Light gray
- **Number formatting:**
  - Prices: 4 decimal places
  - Percentages: 2 decimal places
  - Dollars: 2 decimal places with $ symbol
  - Counts: 0 decimal places
- **Column widths:** Auto-adjusted, minimum 15 characters
- **Freeze panes:** Header rows frozen where applicable
- **Page setup:** Landscape where tables exceed 10 columns

---

## 6. PATTERN IDENTIFICATION LOGIC

### 6.1 Entry Signal Strength Scoring

**Scoring formula:** 0-100 scale based on technical confluence

```
Base score: 0

Technical Factors (add points if met):
  1. Trend Aligned: SMA_20 > SMA_50 > SMA_200 (LONG) or reverse (SHORT)
     → +25 points
  2. Price Position: Within middle third of Bollinger Bands
     → +12 points
  3. Volume Confirmation: Entry bar volume > 110% of 20-day avg
     → +15 points
  4. MACD Bullish: MACD > Signal line (LONG) or reverse (SHORT)
     → +15 points
  5. Moving Average Distance: Price within 2% of SMA_50
     → +10 points
  6. RSI Momentum: RSI 40-60 (neutral, not extreme)
     → +10 points
  7. ATR Volatility: ATR within 0.8-1.2x of 60-day average
     → +8 points
  8. Price Momentum: Close > Open on entry day (LONG) or reverse (SHORT)
     → +5 points

Total possible: 100 points
Scoring tiers:
  80-100: Excellent confluence (very strong setup)
  60-79: Good confluence (strong setup)
  40-59: Moderate confluence (reasonable setup)
  0-39: Weak confluence (questionable setup)
```

### 6.2 Pattern Flagging Rules

**Applied to each trade, flagged in GUI and Excel:**

#### Insider Activity Patterns
```
🚩 FLAG_INSIDER_BUYING_PREENTRY: Insider buying >= $50k in 30 days pre-entry
   Interpretation: Potential bullish insider knowledge
   
🚩 FLAG_INSIDER_SELLING_POSTENTRY: Insider selling >= $50k in 30 days post-entry
   Interpretation: May indicate insider knows of headwinds
   
🚩 FLAG_INSIDER_BUYING_POSTEXIT: Insider buying >= $50k in 7 days after exit
   Interpretation: You exited before insiders were buying (validate timing)
   
🚩 FLAG_COORDINATED_INSIDER_ACTIVITY: Multiple insiders (2+) active in same week
   Interpretation: Suggests coordinated decision (stronger signal)
   
🚩 FLAG_LARGE_INSIDER_TRANSACTION: Single transaction > 5% of insider's holdings
   Interpretation: Unusually concentrated position change
```

#### Technical Entry Patterns
```
🚩 FLAG_RSI_EXTREME: RSI < 30 (oversold) or RSI > 70 (overbought) at entry
   Interpretation: Possible mean reversion entry (risky if trend continuation)
   
🚩 FLAG_ATR_SPIKE: ATR at entry > 1.5x of 60-day average
   Interpretation: High volatility environment (stop losses may be wider needed)
   
🚩 FLAG_VOLUME_SPIKE: Entry volume > 200% of 20-day average
   Interpretation: Potential exhaustion or capitulation (watch for reversals)
   
🚩 FLAG_BREAKOUT_SETUP: Price > 52-week high (LONG) or < 52-week low (SHORT)
   Interpretation: Continuation potential but reversal risk
```

#### Market Regime Patterns
```
🚩 FLAG_COUNTER_TREND: Entry against the direction of 200-SMA
   Interpretation: Mean reversion play (more likely to fail)
   
🚩 FLAG_LOW_VOLATILITY_ENTRY: ATR < 0.66x of 60-day average
   Interpretation: Market sleepy; moves may be slow or choppy
   
🚩 FLAG_HIGH_VOLATILITY_ENTRY: ATR > 1.5x of 60-day average
   Interpretation: Whipsaws likely; stops may be hit easily
```

#### Correlation Patterns
```
🚩 FLAG_SECTOR_MISALIGNMENT: Trade direction opposite to sector movement
   Interpretation: Idiosyncratic move; if it fails, likely correlates to sector drop
   
🚩 FLAG_WEAK_CORRELATION: Sector correlation < 0.30
   Interpretation: Move driven by company-specific (idiosyncratic) factors
   
🚩 FLAG_STRONG_CORRELATION: Sector correlation > 0.85
   Interpretation: Could you have just traded the sector instead?
```

#### Options Market Patterns (if available)
```
🚩 FLAG_HIGH_IV_ENTRY: IV in top 25th percentile
   Interpretation: High implied volatility priced in; reversals likely
   
🚩 FLAG_LOW_IV_ENTRY: IV in bottom 25th percentile
   Interpretation: Low volatility priced in; moves may be muted
   
🚩 FLAG_EARNINGS_DURING_TRADE: Earnings within trade window
   Interpretation: Catalyst risk; volatility likely to spike
   
🚩 FLAG_BEARISH_OPTIONS_FLOW: Put/Call ratio spike, bearish flow signals
   Interpretation: Market makers/large traders positioning for downside
```

### 6.3 Success Pattern Analysis

**Aggregate analysis across selected trades:**

**For Winners:**
- What are the common characteristics? (by signal strength, regime, insider activity, correlation)
- What technical setups appear most frequently in winners?
- Do winners occur more often in uptrends or downtrends?
- Do winners have insider buying precursor?
- Do winners align with sector correlation?

**For Losers:**
- What are the distinguishing characteristics?
- Which flags appear more frequently?
- Did losers consistently enter in counter-trend?
- Did losers have negative insider signals?
- Did losers have extremely high/low IV?

**Output: Summary statement**
```
"Your winning trades exhibit: [list 3-5 characteristics]
 Your losing trades exhibit: [list 3-5 characteristics]
 Common mistake pattern: [if identifiable]
 Opportunity to exploit: [if pattern identified]"
```

---

## 7. DATA VALIDATION & ERROR HANDLING

### 7.1 Trade Log Validation
- ✅ All required columns present
- ✅ Date format consistency (YYYY-MM-DD)
- ✅ No null values in core columns
- ✅ exit_date >= entry_date
- ✅ Consistent PL calculation: pl = quantity * (exit_price - entry_price) * (1 if side==LONG else -1)
- ✅ Price data exists for all entry/exit dates
- ✅ Symbol in metadata config

**Error Handling:**
- Missing columns: Show detailed list, stop import
- Invalid dates: Highlight rows, ask user to fix
- Missing price data: Warn user, exclude trades from analysis

### 7.2 Data Completeness Warnings
- Track which data is missing for each trade
- Flag in "Data Quality" section of GUI
- Mark Excel sheets as "NOT AVAILABLE" if entire category missing
- Still allow analysis to proceed with available data

### 7.3 Metadata Validation
- ✅ config/trade_analysis_metadata.json exists
- ✅ All symbols in trade logs are in metadata
- ✅ All referenced files (daily, fundamentals, insider, options) exist
- ✅ Date ranges in metadata match actual data

**Error Handling:**
- Suggest metadata updates
- Allow user to specify missing files
- Provide template metadata.json if missing

---

## 8. PERFORMANCE REQUIREMENTS

- **GUI responsiveness:** All interactions complete within 2 seconds
- **Data loading:** Up to 100 trades, 5 CSV files → load in < 10 seconds
- **Excel generation:** Complete 10-sheet workbook generated in < 5 seconds
- **Streaming/on-demand:** No batch processing needed (tool is interactive)

---

## 9. IMPLEMENTATION CHECKLIST

### Phase 1: Core Infrastructure
- [ ] Trade log CSV ingestion + validation
- [ ] Metadata JSON loader + validation
- [ ] Data aggregation pipeline (price, fundamentals, insider, options, correlation)
- [ ] Representative trade sampling algorithm

### Phase 2: GUI Development
- [ ] Streamlit app scaffold
- [ ] Trade selection panel
- [ ] Tab 1: Trade Summary
- [ ] Tab 2: Price Action & Technicals
- [ ] Tab 3: Fundamentals
- [ ] Tab 4: Insider Activity
- [ ] Tab 5: Options Market
- [ ] Tab 6: Sector/Market Correlation
- [ ] Tab 7: MAE/MFE Analysis
- [ ] Tab 8: Pattern Analysis
- [ ] Data quality warnings

### Phase 3: Excel Export
- [ ] Excel workbook generation
- [ ] Sheet 1-10 implementation
- [ ] Formatting (colors, number formats, column widths)
- [ ] Formula-based calculations
- [ ] Conditional formatting

### Phase 4: Pattern Analysis
- [ ] Signal strength scoring algorithm
- [ ] Entry pattern flagging rules
- [ ] Insider pattern flagging
- [ ] Technical pattern flagging
- [ ] Market regime pattern flagging
- [ ] Correlation pattern flagging
- [ ] Options pattern flagging
- [ ] Aggregate pattern analysis

### Phase 5: Testing & Documentation
- [ ] Unit tests for calculations
- [ ] Integration tests with sample data
- [ ] User documentation
- [ ] Example trade analysis walkthrough

---

## 10. CONFIGURATION TEMPLATES

### 10.1 Sample metadata.json

See Section 2.3 for full template. Example:
```json
{
  "available_symbols": {
    "AAPL": {
      "has_daily_prices": true,
      "has_weekly_prices": true,
      "has_fundamentals": true,
      "has_insider_data": true,
      "has_options_data": true,
      "sector_ticker": "XLK",
      "index_ticker": "SPX",
      "primary_currency": "USD",
      "data_start_date": "2009-01-01",
      "data_end_date": "2026-01-04"
    }
  },
  "thresholds": {
    "insider_activity_min_value_usd": 50000,
    "insider_buying_lookback_days": 30,
    "extreme_trade_pl_percentile": 95
  }
}
```

### 10.2 Expected Directory Structure

```
project_root/
├── config/
│   └── trade_analysis_metadata.json
├── raw_data/
│   ├── daily/
│   │   ├── AAPL_daily.csv
│   │   ├── MSFT_daily.csv
│   │   ├── XLK_daily.csv
│   │   └── SPX_daily.csv
│   ├── weekly/
│   │   ├── AAPL_weekly.csv
│   │   └── MSFT_weekly.csv
│   ├── fundamentals/
│   │   ├── AAPL_fundamentals.csv
│   │   └── MSFT_fundamentals.csv
│   ├── insider/
│   │   ├── AAPL_insider.csv
│   │   └── MSFT_insider.csv
│   └── options/
│       ├── AAPL_options.csv
│       └── MSFT_options.csv
├── trade_logs/
│   ├── backtest_20240101_200trades.csv
│   └── backtest_20241215_150trades.csv
├── outputs/
│   └── [generated Excel reports]
└── per_trade_analysis.py
```

---

## 11. USAGE WORKFLOW

1. **Prepare data:**
   - Ensure trade logs in `trade_logs/` folder
   - Ensure raw data in `raw_data/` subfolders
   - Create/validate `config/trade_analysis_metadata.json`

2. **Launch tool:**
   ```bash
   streamlit run per_trade_analysis.py
   ```

3. **Load trades:**
   - Use file upload to select 1+ trade log CSV files
   - System auto-validates

4. **Filter trades:**
   - Apply sidebar filters (symbol, side, outcome, date range)
   - Review summary statistics

5. **Auto-select sample:**
   - Click "Auto-Select Representative Sample" button
   - System suggests 8-15 representative trades
   - Review rationale for each selection

6. **Analyze trade:**
   - Click on a trade row
   - View detailed tabs (Summary → Technicals → Fundamentals → ... → Correlation)
   - Review data quality warnings

7. **Export report:**
   - Click "Export to Excel" button
   - System generates multi-sheet workbook
   - File saved to `outputs/` folder

8. **Identify patterns:**
   - Review "Pattern Analysis" tab
   - Look for common characteristics in winners
   - Note differences between winners and losers
   - Decide: Are there exploitable patterns?

---

## 12. DELIVERABLES

**From Claude AI:**

1. **per_trade_analysis.py** - Complete Streamlit application
   - Trade selection and filtering UI
   - All 8 tabs with visualizations/tables
   - Data aggregation pipeline
   - Excel export functionality
   - Pattern flagging logic
   - Error handling and validation

2. **data_aggregation.py** - Data processing module
   - Trade log ingestion
   - Price/indicator aggregation
   - Fundamentals retrieval
   - Insider activity processing
   - Options data handling
   - Correlation calculation
   - Signal strength scoring
   - MAE/MFE calculation

3. **reporting.py** - Excel generation module
   - Multi-sheet workbook creation
   - Formatting and styling
   - Formula-based calculations
   - Conditional formatting
   - Summary statistics

4. **config/trade_analysis_metadata.json** - Configuration template

5. **README.md** - User guide with:
   - Setup instructions
   - File structure requirements
   - Workflow walkthrough
   - Example analyses
   - Troubleshooting

---

## 13. ASSUMPTIONS & CONSTRAINTS

**Assumptions:**
- Trade logs are accurate and internally consistent
- Price data covers all entry/exit dates
- Metadata JSON is maintained and up-to-date
- One backtest strategy analyzed at a time
- User has ~50 trades per security over 15-year period

**Constraints:**
- No real-time data (analysis is post-hoc)
- No machine learning prediction layer (flagging only)
- Limited to single strategy per analysis session
- Performance optimized for <200 trades per analysis
- Desktop/laptop execution (not mobile)

---

## 14. FUTURE ENHANCEMENTS

- **Machine learning:** Train classifier on winners vs losers, predict future trade success
- **Multi-strategy analysis:** Compare patterns across different strategies
- **Real-time monitoring:** Alert on setup similarities during live trading
- **Sentiment analysis:** Integrate news/social sentiment at entry/exit dates
- **Walk-forward analysis:** Update patterns as new trades accumulate
- **Monte Carlo simulation:** Test pattern robustness with parameter variation

---

**END OF SPECIFICATION**

---

**Version:** 1.0  
**Last Updated:** January 4, 2026  
**Next Review:** After Phase 1 implementation completion
