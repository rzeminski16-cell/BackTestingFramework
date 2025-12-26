# Raw Data Collection System - Refined System Prompt

## EXECUTIVE SUMMARY

This prompt defines a comprehensive **unified raw data collection system** powered by the Alpha Vantage API. The system provides a professional-grade GUI that orchestrates data fetching, validation, preparation, and logging for all data types required by the backtesting framework. It replaces all manual data collection processes with an automated, monitored, and reproducible pipeline.

---

## 1. SYSTEM ARCHITECTURE

### 1.1 Overall Design Philosophy

**Principles:**
- **Unified Source of Truth**: All raw data flows through Alpha Vantage API (except where unavailable)
- **Data Integrity First**: Extensive validation prevents corrupted/incomplete data from entering the pipeline
- **Reproducibility**: Comprehensive logging enables audit trails and debugging
- **User Transparency**: Interactive guidance and clear feedback for all decisions
- **Resilience**: Automatic retries, caching, and graceful degradation on errors
- **Performance**: Batch processing, smart caching, and rate-limit awareness

### 1.2 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUI ORCHESTRATOR                         │
│  (Tab selection, Input collection, Progress monitoring)         │
└─────────────┬───────────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬────────────┬──────────────┬────────────┐
    │         │         │            │              │            │
    ▼         ▼         ▼            ▼              ▼            ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Daily │ │Weekly│ │Fund. │ │ Insider  │ │  Forex   │ │ Options  │
│Data  │ │Data  │ │Data  │ │ Trans.   │ │  Weekly  │ │  Data    │
└──────┘ └──────┘ └──────┘ └──────────┘ └──────────┘ └──────────┘
    │         │         │            │              │            │
    └─────────┼─────────┼────────────┼──────────────┼────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │    Alpha Vantage API Client     │
    │  (Rate limiting, Caching,       │
    │   Retry logic, Error handling)  │
    └─────────────────────────────────┘
              │
    ┌─────────┼──────────────┐
    ▼         ▼              ▼
┌─────────┐ ┌──────────┐ ┌──────────┐
│Validation│ │Processing│ │  File I/O│
│  Engine  │ │  Engine  │ │  Engine  │
└─────────┘ └──────────┘ └──────────┘
    │         │              │
    └─────────┼──────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │   Prepared Raw Data Files       │
    │   (CSV with correct metadata,   │
    │    data types, folder structure)│
    └─────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │      Comprehensive Logs         │
    │  (Session, decision, error,     │
    │   validation, metric logs)      │
    └─────────────────────────────────┘
```

---

## 2. DATA COLLECTION TABS

### 2.1 Tab 1: Daily Prices & Technical Indicators

**Purpose**: Collect OHLCV data and pre-calculated technical indicators for single-security backtests.

**Input Parameters**:
- **Tickers**: Multi-select or comma-separated list
- **Date Range**: From date (YYYY-MM-DD), To date (YYYY-MM-DD), or "All available"
- **Technical Indicators**: Multi-select from:
	- |SMA|
	- |EMA|
	- |WMA|
	- |DEMA|
	- |TEMA|
	- |TRIMA|
	- |KAMA|
	- |MAMA|
	- |VWAP|
	- |T3|
	- |MACD|
	- |MACDEXT|
	- |STOCH|
	- |STOCHF|
	- |RSI|
	- |STOCHRSI|
	- |WILLR|
	- |ADX|
	- |ADXR|
	- |APO|
	- |PPO|
	- |MOM|
	- |BOP|
	- |CCI|
	- |CMO|
	- |ROC|
	- |ROCR|
	- |AROON|
	- |AROONOSC|
	- |MFI|
	- |TRIX|
	- |ULTOSC|
	- |DX|
	- |MINUS_DI|
	- |PLUS_DI|
	- |MINUS_DM|
	- |PLUS_DM|
	- |BBANDS|
	- |MIDPOINT|
	- |MIDPRICE|
	- |SAR|
	- |TRANGE|
	- |ATR|
	- |NATR|
	- |AD|
	- |ADOSC|
	- |OBV|
	- |HT_TRENDLINE|
	- |HT_SINE|
	- |HT_TRENDMODE|
	- |HT_DCPERIOD|
	- |HT_DCPHASE|
	- |HT_PHASOR|
  - All (select all available indicators)
- **Output Granularity**: Daily (only option for this tab)
- **Adjust for Splits/Dividends**: Yes/No toggle
- **Handle Missing Data**: Skip/Interpolate/Forward-fill

**Output Format**:
```
raw_data/
├── AAPL_daily.csv
├── MSFT_daily.csv
└── ...
```

**CSV Columns** (auto-ordered):
- `date` (datetime)
- `open` (float)
- `high` (float)
- `low` (float)
- `close` (float)
- `volume` (int)
- `[indicators...]` (float)

**Validation Rules**:
- Close price > 0
- High ≥ Low
- High ≥ Open, Close; Low ≤ Open, Close
- Volume ≥ 0
- No duplicate dates
- Chronological order
- No gaps > 5 trading days (configurable)
- Enough data points for SMA/EMA/etc. (min 200 bars recommended)

---

### 2.2 Tab 2: Weekly Prices & Technical Indicators

**Purpose**: Collect weekly OHLCV and indicators for longer-term analysis or lower-frequency strategies.

**Input Parameters**:
- **Tickers**: Multi-select
- **Date Range**: From/To or "All available"
- **Technical Indicators**: Same as daily (but applied to weekly bars)
- **Handle Missing Weeks**: Skip/Interpolate/Forward-fill
- **Same Additional Options**: Adjust for splits/dividends, handle missing data

**Output Format**:
```
raw_data/
├── AAPL_weekly.csv
├── MSFT_weekly.csv
└── ...
```

**CSV Columns**: Same as daily (date, OHLCV, indicators)

**Validation Rules**: Same as daily

---

### 2.3 Tab 3: Fundamental Data

**Purpose**: Collect quarterly fundamental metrics from Alpha Vantage (point-in-time data where possible).

**Input Parameters**:
- **Tickers**: Multi-select
- **Date Range**: Start year (YYYY), End year (YYYY), or "All available"
- **Data Categories**: Multi-select from:
  - Valuation Ratios (P/E, P/B, P/FCF, PEG)
  - Profitability (EPS, Margins, ROE, ROA)
  - Growth Rates (Revenue growth, EPS growth, FCF growth)
  - Financial Health (Debt/Equity, Interest Coverage, Current Ratio)
  - Cash Flow (OCF, CapEx, FCF, FCF Yield)
  - All Categories
- **Include Analyst Estimates**: Yes/No (Forward P/E, Price Targets, etc.)
- **API Configuration**: 
  - API Key (hidden input, validated)
  - Rate limit (inferred from subscription tier or manual entry)
  - Cache settings
  - Timeout values
- **Quarterly vs Annual**: Both / Quarterly only / Annual only
- **Handle Missing Data**: Skip / Use estimates / Interpolate

**Output Format**:
```
raw_data/fundamentals/
├── AAPL_fundamental.csv
├── MSFT_fundamental.csv
└── ...
```

**CSV Columns** (quarterly rows):
- `date` or `quarter_end_date` (datetime)
- `symbol` (string)
- `year` (int)
- `quarter` (int)
- `eps_ttm` (float)
- `eps_growth_yoy_pct` (float)
- `revenue_ttm` (float)
- `revenue_growth_yoy_pct` (float)
- `pe_ratio_trailing` (float)
- `pe_ratio_forward` (float)
- `[...other metrics...]`

**Validation Rules**:
- Required columns non-null
- All numerical columns are actual numbers (not 'NaN' strings)
- No duplicate date/symbol combinations
- Chronological order per symbol
- Reasonable value ranges (no -999 or 999999 placeholders)
- Quarter = 1-4, Year valid

---

### 2.4 Tab 4: Insider Transactions

**Purpose**: Collect insider buying/selling activity from Alpha Vantage.

**Input Parameters**:
- **Tickers**: Multi-select
- **Date Range**: From/To or "Last N days"
- **Transaction Types**: 
  - All transactions
  - Buys only
  - Sells only
- **Relationship Types**: 
  - All (Directors, Officers, Major shareholders, etc.)
  - Select specific: Director, Officer, 10% shareholder, etc.
- **Position Types**: Officer title, Sector, or All
- **Sort/Filter**: 
  - Sort by date, volume, or type
  - Min transaction value (e.g., $10,000)

**Output Format**:
```
raw_data/insider_transactions/
├── AAPL_insider.csv
├── MSFT_insider.csv
└── ...
```

**CSV Columns**:
- `date` (datetime)
- `symbol` (string)
- `insider_name` (string)
- `insider_title` (string)
- `transaction_type` (string: BUY/SELL)
- `shares` (int)
- `price` (float)
- `value` (float) = shares × price
- `ownership_after` (float) - % ownership post-transaction
- `relationship` (string)

**Validation Rules**:
- Valid transaction types (BUY, SELL)
- Shares > 0
- Price > 0
- Date <= today
- No duplicate transactions
- Reasonable price ranges (not outliers)

---

### 2.5 Tab 5: Forex Weekly (GBP/USD, GBP/EUR)

**Purpose**: Collect weekly forex closing prices for currency conversion in multi-currency backtests.

**Input Parameters**:
- **Currency Pairs**: 
  - GBP/USD (required checkbox)
  - GBP/EUR (required checkbox)
  - Both always selected (fixed)
- **Date Range**: From/To or "All available"
- **Data Type**: Close prices only (fixed)
- **Week Definition**: 
  - End of week = Friday close (UTC)
  - End of week = Monday open
- **Handle Missing Weeks**: Skip / Forward-fill / Interpolate

**Output Format**:
```
raw_data/forex/
├── GBPUSD_weekly.csv
├── GBPEUR_weekly.csv
└── ...
```

**CSV Columns**:
- `date` (datetime)
- `symbol` (string: GBPUSD, GBPEUR)
- `close` (float) - 1 GBP = X USD/EUR

**Validation Rules**:
- Close > 0
- Close in realistic range (0.5 - 1.5 typically)
- No duplicate dates
- Chronological order
- Continuous weekly data (no gaps > 1 month)

---

### 2.6 Tab 6: Options Data

**Purpose**: Collect options chain data (calls/puts, strikes, expiration, implied volatility, Greeks).

**Input Parameters**:
- **Tickers**: Multi-select
- **Date Range**: From/To or "Most recent"
- **Options Type**: 
  - Calls only
  - Puts only
  - Both
- **Expiration Dates**: 
  - All available expirations
  - Within N days (e.g., 30, 60, 90)
  - Specific expirations (user selects)
- **Strike Selection**:
  - All available strikes
  - ATM ± N strikes (e.g., ATM ± 5)
  - Custom range (from/to)
- **Data Fields**: Multi-select from:
  - Strike Price
  - Bid/Ask
  - Last Price
  - Volume
  - Open Interest
  - Implied Volatility (IV)
  - Delta, Gamma, Theta, Vega (Greeks)
  - Probability ITM/OTM
  - Time Value
  - Intrinsic Value
- **Snapshot Mode**: 
  - Single snapshot (most recent)
  - Daily snapshots
  - Weekly snapshots
- **Adjust for Stock Splits**: Yes/No

**Output Format**:
```
raw_data/options/
├── AAPL_20250326_calls.csv    # 26-Mar-2025 expiry
├── AAPL_20250326_puts.csv
├── AAPL_20250418_calls.csv    # 18-Apr-2025 expiry
├── AAPL_20250418_puts.csv
└── ...
```

**CSV Columns** (depends on data fields selected):
- `snapshot_date` (datetime)
- `symbol` (string)
- `option_type` (string: CALL/PUT)
- `expiration_date` (datetime)
- `strike` (float)
- `days_to_expiration` (int)
- `bid` (float)
- `ask` (float)
- `last_price` (float)
- `volume` (int)
- `open_interest` (int)
- `implied_volatility` (float)
- `delta` (float)
- `gamma` (float)
- `theta` (float)
- `vega` (float)
- `prob_itm` (float) - 0.0 to 1.0

**Validation Rules**:
- Valid option types (CALL, PUT)
- Strike > 0
- Expiration date > snapshot date
- Days to expiration ≥ 0
- Bid < Ask (if both present)
- IV > 0
- Greeks in reasonable ranges (delta: -1 to 1, gamma > 0, etc.)
- Volume/OI ≥ 0
- No duplicate (symbol, expiration, strike, type, snapshot_date) combinations

---

## 3. GUI DESIGN SPECIFICATIONS

### 3.1 Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│  RAW DATA COLLECTION SYSTEM                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Tabs]                                                 │
│  ┌─ Daily    ─ Weekly    ─ Fundamental ─ Insider ─     │
│  │                 ─ Forex     ─ Options               │
│  │                                                     │
│  ├─────────────────────────────────────────────────────┤
│  │                                                     │
│  │  [TAB CONTENT AREA]                                │
│  │                                                     │
│  ├─────────────────────────────────────────────────────┤
│  │ [Settings Button] [Clear Cache] [View Logs]        │
│  │              [Run Collection] [Exit]                │
│  ├─────────────────────────────────────────────────────┤
│  │                                                     │
│  │ ▶ Collection Progress:                             │
│  │   Processing AAPL... [███████░░░░░░░░] 35%         │
│  │                                                     │
│  ├─────────────────────────────────────────────────────┤
│  │ Messages:                                           │
│  │ [✓] API Key validated successfully                 │
│  │ [→] Fetching AAPL daily data (2024-01-01 to now)  │
│  │ [⚠] MSFT: 3 missing dates, using forward-fill      │
│  │                                                     │
│  └─────────────────────────────────────────────────────┘
```

### 3.2 Tab UI Patterns

**Pattern 1: Ticker Selection**
```
┌──────────────────────────────────────┐
│ Tickers                              │
│ ┌──────────────────────────────────┐ │
│ │ ☑ AAPL                           │ │
│ │ ☐ MSFT                           │ │
│ │ ☐ GOOGL                          │ │
│ │ ... (scrollable)                 │ │
│ └──────────────────────────────────┘ │
│ [Select All] [Clear All]             │
│ Selected: 5 / 100                    │
└──────────────────────────────────────┘
```

**Pattern 2: Date Range**
```
┌──────────────────────────────────────┐
│ Date Range                           │
│ ○ All available data                 │
│ ○ Custom range:                      │
│   From: [2024-01-01] 📅 ▼            │
│   To:   [2025-01-01] 📅 ▼            │
│ ○ Last N days: [365] days            │
└──────────────────────────────────────┘
```

**Pattern 3: Multi-Select Features**
```
┌──────────────────────────────────────┐
│ Technical Indicators                 │
│ ☑ SMA   Periods: [20, 50, 200]       │
│ ☑ EMA   Periods: [12, 26]            │
│ ☑ RSI   Period:  [14]                │
│ ☐ MACD                               │
│ ☐ Bollinger Bands                    │
│ ☐ ATR                                │
│ [All] [None]                         │
│                                      │
│ Selected: 3 indicators               │
└──────────────────────────────────────┘
```

**Pattern 4: Settings Panel**
```
┌──────────────────────────────────────┐
│ API Configuration                    │
│ API Key:        [••••••••••••••] 🔒   │
│ Rate Limit:     [75 calls/min] ▼     │
│ Cache Dir:      [cache/alpha_v...] 📁 │
│ Cache Expiry:   [1 day] ▼            │
│ Timeout:        [30 sec] ▼           │
│ Max Retries:    [3] ▼                │
│                                      │
│ [Test Connection]                    │
└──────────────────────────────────────┘
```

### 3.3 User Interaction Patterns

**Pattern 1: Validation Feedback**
```
✓ Tickers: 5 selected
✓ Date range: 2020-01-01 to 2024-12-31 (1825 days)
✓ Indicators: 3 selected
✗ Warning: Insufficient data for SMA(200) on some dates
? Continue anyway? [Yes] [No] [Cancel]
```

**Pattern 2: Progress Reporting**
```
Collecting Daily Data for 5 tickers...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

AAPL:    ✓ 1258 rows, 8 columns
MSFT:    ✓ 1258 rows, 8 columns
GOOGL:   ⚠ 1255 rows (3 missing dates), 8 columns
AMZN:    ✓ 1258 rows, 8 columns
META:    ✓ 1258 rows, 8 columns

Summary:
  Files created: 5
  Total rows: 6287
  Missing data: 3 rows (handled: forward-fill)
  Processing time: 47s
```

**Pattern 3: Decision Prompts**
```
┌──────────────────────────────────────────────┐
│ DATA DECISION REQUIRED                       │
├──────────────────────────────────────────────┤
│                                              │
│ For AAPL Fundamental Data:                   │
│ Found 2 possible EPS fields:                 │
│  1. reportedEPS (from actual earnings)       │
│  2. estimatedEPS (analyst forecast)          │
│                                              │
│ Which should be used?                        │
│                                              │
│ [Use reportedEPS]  [Use estimatedEPS]        │
│ [Skip]                                       │
│                                              │
│ ☑ Remember this choice for all symbols       │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 4. COMPREHENSIVE LOGGING SYSTEM

### 4.1 Log Architecture

```
logs/data_collection/
├── {session_id}_session.log         # High-level session log
├── {session_id}_api.log             # All API calls, responses, errors
├── {session_id}_validation.log      # Data validation results
├── {session_id}_decisions.log       # User decisions made
├── {session_id}_metrics.log         # Performance metrics (timing, counts)
├── {session_id}_errors.log          # All errors, exceptions, warnings
├── {session_id}_data_issues.log     # Data quality issues
└── {session_id}_summary.txt         # Human-readable session summary

config/
└── {session_id}_remembered_answers.json  # Decision memory for this session
```

### 4.2 Session Log Example

```
[2025-01-15 14:32:10] SESSION STARTED
  Session ID: 20250115_143210_a7f2
  User: analyst_001
  Data Type: Daily Prices & Indicators
  Configuration: 5 tickers, 1000 trading days, 7 indicators

[2025-01-15 14:32:15] API VALIDATION
  [✓] API Key valid
  [✓] Rate limit: 75 calls/min
  [✓] Cache directory: cache/alpha_vantage (1.2GB available)

[2025-01-15 14:32:18] STARTING COLLECTION
  Tickers: AAPL, MSFT, GOOGL, AMZN, META
  Date range: 2022-01-03 to 2025-01-15
  Indicators: SMA(20,50,200), EMA(12,26), RSI(14), ATR(14), MACD

[2025-01-15 14:32:19] FETCHING AAPL
  API call: TIME_SERIES_DAILY_ADJUSTED (from cache)
  Rows fetched: 753 bars
  Columns: OHLCV + adjustment factors
  
[2025-01-15 14:32:35] CALCULATING INDICATORS FOR AAPL
  SMA(20): [✓] Calculated, 733 values (20 NaN at start)
  SMA(50): [✓] Calculated, 703 values (50 NaN at start)
  SMA(200): [⚠] Calculated, 553 values (200 NaN at start)
  EMA(12): [✓] Calculated, 753 values
  EMA(26): [✓] Calculated, 753 values
  RSI(14): [✓] Calculated, 739 values (14 NaN at start)
  ATR(14): [✓] Calculated, 739 values (14 NaN at start)
  MACD: [✓] Calculated, 728 values (26 NaN at start)

[2025-01-15 14:32:42] VALIDATION FOR AAPL
  [✓] No negative prices
  [✓] High >= Low on all dates
  [✓] Chronological order verified
  [✓] No duplicate dates
  [✓] Volume sanity check passed (avg 50M)
  [✓] All indicators are numeric
  [✓] NaN values documented (expected at indicator start)
  [✓] Data quality: 100% complete OHLCV, 97% complete indicators

[2025-01-15 14:32:45] PROCESSING
  File: raw_data/AAPL_daily.csv
  Rows: 753
  Columns: 15 (date, open, high, low, close, volume, sma_20, sma_50, 
           sma_200, ema_12, ema_26, rsi_14, atr_14, macd, macd_signal)
  Dtypes: date(datetime64), OHLCV(float), indicators(float), volume(int)
  Size: 145KB

[2025-01-15 14:32:48] FETCHING MSFT
  API call: TIME_SERIES_DAILY_ADJUSTED (from cache)
  ...

[2025-01-15 14:36:30] COLLECTION SUMMARY
  Total tickers processed: 5
  Successful: 5 (100%)
  Failed: 0
  Partial: 0
  
  Total rows created: 3765
  Total data size: 725KB
  Processing time: 3m 52s
  Average rate: 1.3 tickers/min
  
[2025-01-15 14:36:35] DATA ISSUES ENCOUNTERED
  [⚠] GOOGL: 5 missing dates (2024-11-28, 2024-12-06 weekends - expected)
  [⚠] AMZN: Volume outlier 325M (3σ above mean), verified as legitimate
  [⚠] META: SMA(200) has 200 NaN values at start (expected)
  
  All issues reviewed and acceptable.

[2025-01-15 14:36:37] SESSION COMPLETED SUCCESSFULLY
  Exit code: 0
  Files created: 5 CSV files in raw_data/
  Total API calls: 18 (12 cached, 6 fresh)
  API quota remaining: 4182 / 4500 (approx. 93%)
```

### 4.3 API Activity Log Example

```
[API] 2025-01-15 14:32:19 | GET | alpha_vantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL
      Status: 200 OK
      Response size: 247KB
      Cache: MISS (fresh fetch)
      Time: 1.34s

[API] 2025-01-15 14:32:35 | GET | alpha_vantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT
      Status: 200 OK
      Response size: 251KB
      Cache: HIT (expires in 23h 47m)
      Time: 0.03s

[API] 2025-01-15 14:33:02 | GET | alpha_vantage.co/query?function=INCOME_STATEMENT&symbol=AAPL
      Status: 429 Too Many Requests
      Rate limit exceeded. Waiting 45 seconds...
      Retry: 1/3
      Time: 45.12s

[API] 2025-01-15 14:33:47 | GET | alpha_vantage.co/query?function=INCOME_STATEMENT&symbol=AAPL [RETRY 1]
      Status: 200 OK
      Response size: 189KB
      Cache: MISS
      Time: 1.87s
```

### 4.4 Validation Log Example

```
[VALIDATION] AAPL_daily.csv
  ✓ File exists and readable (145KB)
  ✓ Row count: 753 (expected > 100)
  ✓ Column count: 15 (expected)
  ✓ Required columns present: date, open, high, low, close, volume
  ✓ Date format: datetime64[ns] (valid)
  ✓ Date order: ascending, no duplicates
  ✓ Date range: 2022-01-03 to 2025-01-15 (1000 trading days ✓)
  
  Column: open
    ✓ Type: float64
    ✓ Min: 104.23, Max: 199.62 (reasonable)
    ✓ No NaNs in 753 rows
    ✓ No negative values
    ✓ 0 outliers (3σ)
  
  Column: high
    ✓ Type: float64
    ✓ High >= Low on all 753 rows
    ✓ High >= Open, Close on all 753 rows
    ✓ No NaNs
  
  Column: close
    ✓ Type: float64
    ✓ Reasonable range (104.23 - 199.62)
    ✓ No NaNs
  
  Column: volume
    ✓ Type: int64
    ✓ Min: 12M, Max: 89M (avg: 50M)
    ✓ No negative values
    ✓ No zeros (unusual volume values: 0 instances)
    ⚠ Outlier: 325M on 2024-12-23 (3.2σ above mean)
      → Verified as legitimate (holiday trading)
  
  Column: sma_20
    ✓ Type: float64
    ✓ Expected NaNs at start: 20 (verified)
    ✓ Range: 104.80 - 199.12 (reasonable)
    ✓ 20 NaNs, 733 valid values
  
  [All 15 columns validated successfully]
  
  OVERALL: ✓ PASS (100% data quality)
```

### 4.5 Decisions Log Example

```
[2025-01-15 14:45:32] DECISION | User selected API configuration
  Setting: rate_limit_per_minute
  Value: 75
  Source: Detected from subscription plan (Premium basic)
  
[2025-01-15 14:46:15] DECISION | User selected missing data handling
  Tab: Daily Data
  Setting: handle_missing_data
  Value: forward_fill
  Applied to: 3 missing dates in GOOGL
  
[2025-01-15 14:47:22] DECISION | Data ambiguity resolved
  Question: For AAPL Fundamental Data, which EPS field?
  Options: [reportedEPS, estimatedEPS]
  Selected: reportedEPS
  Applied to: AAPL and subsequent symbols
  Remember: YES (checkbox checked)
  
[2025-01-15 14:48:01] DECISION | Outlier acceptance
  Alert: AMZN volume outlier detected (325M, 3σ above mean)
  Recommendation: Investigate / Accept as legitimate / Skip
  User selected: Accept as legitimate (verified as holiday trading)
  Action: Included in final dataset
```

### 4.6 Metrics Log Example

```
SESSION METRICS
===============

Timing:
  Total duration: 3m 52s
  Data collection: 3m 24s
  Validation: 18s
  File I/O: 10s
  
API Performance:
  Total calls: 18
  Cached responses: 12 (67%)
  Fresh fetches: 6 (33%)
  Failed calls (retried): 1
  Average response time: 0.87s
  Retry rate: 5.6%
  
Rate Limiting:
  Calls per minute: avg 5.1, peak 12
  Rate limit exhaustion: 0 (limit: 75)
  Quota remaining: 4182 / 4500
  
Data Processing:
  Total rows processed: 3765
  Total cells processed: 56,475
  Validation checks: 847
  Issues found: 3 (warnings)
  Issues resolved: 3
  
Efficiency:
  Rows processed/second: 16.2
  API calls/second: 0.08
  Data quality score: 99.7%
  Cache hit ratio: 66.7%
```

### 4.7 Errors Log Example

```
[ERROR] 2025-01-15 14:33:02 | API Rate Limit Exceeded
  Endpoint: INCOME_STATEMENT
  Symbol: AAPL
  Status: 429 Too Many Requests
  Message: API rate limit 75 calls/min exceeded
  Recovery: Automatic retry with exponential backoff
  Retry: 1/3 successful after 45.12s
  Impact: Delayed but completed successfully
  
[WARNING] 2025-01-15 14:35:18 | Missing Data
  Symbol: GOOGL
  Dates: 2024-11-28, 2024-11-29, 2024-12-06, 2024-12-07, 2024-12-24
  Reason: Weekend and holidays (expected)
  Resolution: Forward-fill applied (user preference)
  Impact: None (expected and handled)
  
[WARNING] 2025-01-15 14:35:42 | Data Outlier Detected
  Symbol: AMZN
  Field: volume
  Date: 2024-12-23
  Value: 325M (mean: 98M, std: 71M, z-score: 3.2)
  Recommendation: Investigate or skip
  User decision: Accept as legitimate (holiday trading volume spike)
  Impact: Included in dataset
  
[ERROR] 2025-01-15 14:38:21 | API Timeout
  Endpoint: TIME_SERIES_DAILY_ADJUSTED
  Symbol: MSFT
  Timeout: 30 seconds
  Recovery: Automatic retry from cache
  Retry: 1/3 successful (cache hit)
  Impact: Minimal (cache fallback used)
```

### 4.8 Data Issues Log Example

```
DATA QUALITY REPORT
===================

AAPL:
  ✓ 753 rows
  ✓ 0 missing values (OHLCV)
  ✓ 0 data inconsistencies
  ✓ 0 invalid dates
  ⚠ 20 expected NaNs in SMA(20) (start-of-series, normal)
  Status: CLEAN

MSFT:
  ✓ 753 rows
  ✓ 0 missing values (OHLCV)
  ✓ 0 data inconsistencies
  ⚠ 3 date gaps (weekends, expected)
  ⚠ 50 expected NaNs in SMA(50) (start-of-series, normal)
  Status: CLEAN

GOOGL:
  ⚠ 750 rows (3 missing)
  ⚠ 3 missing OHLCV records
  ⚠ Missing dates: 2024-11-28 (Thanksgiving), 2024-12-06 (Friday after Thanksgiving)
  ⚠ Resolution: Forward-fill applied (user selected)
  ✓ Post-resolution: No missing values
  Status: CLEAN (after handling)

AMZN:
  ✓ 753 rows
  ⚠ 1 volume outlier (325M on 2024-12-23, z-score: 3.2)
  ⚠ Investigation: Identified as legitimate holiday trading spike
  ✓ User confirmed: Keep in dataset
  Status: CLEAN

META:
  ✓ 753 rows
  ✓ All data valid
  Status: CLEAN

OVERALL:
  Files validated: 5 / 5
  Quality score: 99.7%
  Issues found: 5
  Issues resolved: 5
  Critical issues: 0
  Summary: All data ready for backtesting
```

---

## 5. DATA PROCESSING PIPELINE

### 5.1 Validation Engine

**Purpose**: Ensure data integrity before writing to files.

**Checks**:

1. **Type Validation**
   - Date/time columns are datetime64
   - OHLC/price columns are float
   - Volume is integer
   - Indicators are float

2. **Value Range Validation**
   - Price columns > 0
   - Volume >= 0
   - High >= Low
   - High >= Open and Close
   - Low <= Open and Close
   - No infinity or extreme values

3. **Temporal Validation**
   - Chronological order (ascending dates)
   - No duplicate dates
   - No gaps > threshold (e.g., 5 trading days)
   - Valid date ranges

4. **Data Completeness**
   - Required columns present
   - No unexpected null values (except expected NaNs at indicator start)
   - Sufficient data points (min 50 rows recommended)

5. **Consistency Checks**
   - Volume and price correlation (unusual patterns flagged)
   - Indicator calculations verified (e.g., SMA = sum(last N) / N)
   - Missing data patterns documented

### 5.2 File Preparation

**CSV Formatting**:
- **Encoding**: UTF-8
- **Line endings**: LF (Unix-style, cross-platform)
- **Delimiter**: Comma
- **Quoting**: Only when needed (minimal)
- **NaN representation**: Empty cells (not "NaN" strings)
- **Date format**: YYYY-MM-DD HH:MM:SS (ISO 8601)
- **Numeric precision**: 2 decimals for prices, 6 for ratios, 0 for volume/counts

**Column Naming**:
- All lowercase
- Underscores for spaces (e.g., `days_to_expiration`)
- Consistent suffixes:
  - `_pct` for percentages
  - `_ttm` for trailing-twelve-month
  - `_yoy` for year-over-year
  - `_ma` or `_sma` for moving averages
  - `_ema` for exponential moving averages

**Folder Structure**:
```
raw_data/
├── daily/
│   ├── AAPL_daily.csv
│   ├── MSFT_daily.csv
│   └── ...
├── weekly/
│   ├── AAPL_weekly.csv
│   ├── MSFT_weekly.csv
│   └── ...
├── fundamentals/
│   ├── AAPL_fundamental.csv
│   ├── MSFT_fundamental.csv
│   └── ...
├── insider_transactions/
│   ├── AAPL_insider.csv
│   ├── MSFT_insider.csv
│   └── ...
├── forex/
│   ├── GBPUSD_weekly.csv
│   └── GBPEUR_weekly.csv
├── options/
│   ├── AAPL_20250326_calls.csv
│   ├── AAPL_20250326_puts.csv
│   └── ...
└── metadata.json
```

### 5.3 Metadata Generation

Each session generates a `metadata.json` file documenting:

```json
{
  "session_id": "20250115_143210_a7f2",
  "timestamp": "2025-01-15T14:32:10Z",
  "data_types": ["daily_prices", "weekly_prices", "fundamental_data"],
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "files_created": {
    "daily": {
      "AAPL_daily.csv": {
        "rows": 753,
        "columns": 15,
        "date_range": ["2022-01-03", "2025-01-15"],
        "size_kb": 145,
        "data_quality": "100%",
        "indicators": ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "rsi_14", "atr_14", "macd"],
        "validation_status": "PASS"
      }
    }
  },
  "api_usage": {
    "total_calls": 18,
    "cached_calls": 12,
    "fresh_calls": 6,
    "quota_remaining": 4182,
    "quota_total": 4500
  },
  "processing_stats": {
    "total_duration_seconds": 232,
    "total_rows_processed": 3765,
    "issues_found": 3,
    "issues_resolved": 3,
    "data_quality_score": 99.7
  },
  "decisions_made": [
    {
      "decision": "missing_data_handling",
      "value": "forward_fill",
      "timestamp": "2025-01-15T14:46:15Z"
    },
    {
      "decision": "eps_field_selection",
      "symbol": "AAPL",
      "value": "reportedEPS",
      "remembered": true,
      "timestamp": "2025-01-15T14:47:22Z"
    }
  ]
}
```

---

## 6. ERROR HANDLING & RESILIENCE

### 6.1 Retry Strategy

- **API Timeouts**: Exponential backoff (1s, 2s, 4s, 8s, 16s) up to 3 retries
- **Rate Limits**: Wait until rate limit resets (or 60s, whichever is lower) then retry
- **Transient Errors (5xx)**: Exponential backoff, max 3 retries
- **Invalid Data**: Skip and log, continue with next ticker
- **Network Errors**: Automatic retry with backoff, fallback to cache

### 6.2 User Control

During collection, users can:
- **Pause**: Halt collection, save progress
- **Resume**: Continue from pause point
- **Skip**: Skip current ticker, move to next
- **Abort**: Cancel collection, save what's been processed
- **Adjust settings**: Real-time rate limit or timeout adjustment

### 6.3 Fallback Mechanisms

1. **API Unavailable**: Fall back to cached data (if within expiry)
2. **Missing Indicators**: Calculate locally if formula known
3. **Incomplete Data**: Use interpolation/forward-fill per user setting
4. **Rate Limit Exceeded**: Queue requests and retry in batch

---

## 7. INTEGRATION WITH EXISTING FRAMEWORK

### 7.1 Data Consumption Points

The collected data feeds into:

1. **Single-Security Backtests** (via User Guide):
   - Daily/weekly CSV files → DataLoader → SingleSecurityEngine

2. **Portfolio Backtests** (via Portfolio Mode Guide):
   - Multiple CSV files → PortfolioEngine → Multi-security analysis

3. **Optimization** (via Optimization Guide):
   - Same CSV files → OptimizationEngine → Parameter search

4. **Analysis** (via future enhancement):
   - Fundamental + insider data → Backtest Analyzer → Enhanced insights

### 7.2 File Naming Conventions

Align with framework expectations:
- Daily: `{TICKER}_daily.csv`
- Weekly: `{TICKER}_weekly.csv`
- Fundamentals: `{TICKER}_fundamental.csv`
- Insider: `{TICKER}_insider.csv`
- Forex: `{PAIR}_weekly.csv` (e.g., `GBPUSD_weekly.csv`)
- Options: `{TICKER}_{EXPIRY_YYYYMMDD}_{CALL|PUT}.csv`

---

## 9. IMPLEMENTATION CHECKLIST

### Core Components
- [ ] Alpha Vantage API client (rate limiting, caching, retries)
- [ ] GUI main window with tabbed interface
- [ ] 6 tab implementations (Daily, Weekly, Fundamental, Insider, Forex, Options)
- [ ] Validation engine with comprehensive checks
- [ ] File preparation and writing system
- [ ] Comprehensive logging system (7 log types)
- [ ] Progress tracking and reporting

### Quality Assurance
- [ ] Unit tests for each validator
- [ ] Integration tests for data pipeline
- [ ] API mocking for testing
- [ ] Sample data files for all 6 data types
- [ ] Edge case handling (gaps, outliers, missing data)

### Documentation
- [ ] CLI help / --help text
- [ ] In-app tooltips for all settings
- [ ] Error message clarity and actionability
- [ ] User guide for new users
- [ ] Example output files

---

## 10. KEY DESIGN PRINCIPLES (SUMMARY)

1. **User-Centric**: Clear feedback, guided decisions, no silent failures
2. **Data-Centric**: Validation before file writing, comprehensive metadata
3. **Transparent**: Exhaustive logging enables debugging and auditing
4. **Resilient**: Automatic retries, graceful degradation, fallback mechanisms
5. **Performant**: Smart caching, batch processing, rate limit awareness
6. **Professional**: Production-grade error handling, monitoring, reporting
7. **Integrated**: Seamless connection to backtesting framework
8. **Extensible**: Modular design allows future data sources and features

---

## CONCLUSION

This refined specification defines a comprehensive, professional-grade data collection system that replaces all manual data collection processes in your backtesting framework. The system combines a user-friendly GUI, robust API integration, extensive validation, and comprehensive logging to ensure data integrity, traceability, and seamless integration with your existing backtesting pipeline.

The emphasis on logging and user feedback ensures that all decisions are documented, all issues are visible, and the system is transparent in its operation—critical for a tool used in financial analysis and backtesting.
