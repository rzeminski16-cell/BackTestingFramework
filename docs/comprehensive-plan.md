# Strategy Performance Factor Analysis Module – Comprehensive Design Plan

> High-level, implementation-ready specification for extending an existing Python backtesting framework to analyze how **market, technical, fundamental, insider, and options factors** relate to **trade-level performance**, while strictly avoiding forward-looking bias and data leakage.

---

## 1. Objectives & Scope

### 1.1 Primary Goals

1. **Explain Trade Performance via Factors**  
   Quantify how different factors (technical, fundamental, insider activity, options data, and market regime) correlate with:
   - Trade outcome classes: **good / bad / indeterminate**.
   - Continuous trade metrics: **PL**, **PL%**, **duration_days**, **Calmar/Sharpe-like ratios per scenario**.

2. **Identify Best/Worst Scenarios**  
   Detect factor combinations and market regimes under which the strategy tends to:
   - Perform **consistently well** (best scenarios: Calmar > 0.5 and ≥ 1 trade/year).
   - Perform **consistently poorly** (worst scenarios: Calmar ≤ 0.5 or unstable performance).

3. **Support Multiple Symbols & Logs**  
   Allow analysis of **multiple trade logs at once**:
   - Same strategy, different symbols.
   - Option to **pool trades globally** or **stratify** by symbol, period, and other metadata.

4. **Deliver Stakeholder-Ready Outputs**  
   - **Interactive GUI(s)** (tkinter-based; exact implementation delegated to Claude) for exploration.
   - **Excel reports** with professional formatting, charts, tables, and explanatory text.

5. **Avoid Forward-Looking Bias & Data Leakage**  
   - Enforce **strict temporal alignment** using report_date, filing dates, and snapshot dates with configurable lags.
   - Make all as-of decisions explicit and logged.
   - Provide data quality scoring and audit trails for every decision.

---

## 2. High-Level Architecture

### 2.1 Module Overview

Introduce a self-contained **"Factor Analysis" subsystem** within the backtesting framework with the following logical layers:

1. **Input Layer**  
   - Trade Logs Loader.
   - Raw Market & Indicator Data Loader.
   - Fundamental Data Loader.
   - Insider Activity Loader.
   - Options Data Loader.

2. **Preprocessing & Enrichment Layer**  
   - Data validation & quality checks.
   - Temporal alignment (as-of logic, lags, and delays).
   - Factor engineering (technical, value, quality, growth, insider, options, regime).
   - Trade classification (good / bad / indeterminate).
   - Multi-log aggregation with metadata tagging.
   - Data quality scoring and completeness reporting.

3. **Analysis Layer**  
   - Descriptive statistics (Tier 1).
   - Hypothesis testing & regression analysis (Tier 2).
   - Feature importance & scenario discovery (Tier 3, ML-based).
   - Best/worst scenario evaluation using Calmar and related metrics.

4. **Output Layer**  
   - Data structures for GUI consumption (JSON payloads).
   - Excel report generator (with formatting, charts, embedded tables).
   - Logging, audit trails, and configuration snapshots.

5. **Configuration & Profiles Layer**  
   - YAML/JSON configuration support.
   - Strategy profiles (persisted, reusable configurations).
   - Profile versioning and change tracking.

### 2.2 Module Boundaries

The Factor Analysis module should:

- **Not modify** the core backtesting engine.  
- **Depend only on** outputs already produced by the backtester (trade logs and raw data).  
- Provide clean APIs:
  - `analyze_trades(config, trade_logs, data_sources) -> AnalysisResult`
  - `export_excel_report(analysis_result, output_path)`
  - `prepare_gui_payloads(analysis_result) -> dict`
  - `save_profile(profile, name) -> None`
  - `load_profile(name) -> Config`

---

## 3. Input Data and Assumptions

### 3.1 Trade Logs

**Format:** CSV, one file per backtest per security.

**Columns (required):**
- `trade_id` (unique identifier)
- `symbol` (ticker)  
- `entry_date` (YYYY-MM-DD format, day close)
- `entry_price`  
- `exit_date` (YYYY-MM-DD format, day close)
- `exit_price`  
- `quantity`  
- `side` (long / short)  
- `initial_stop_loss`  
- `final_stop_loss`  
- `take_profit`  
- `pl` (final profit/loss, aggregated across all partial exits)
- `pl_pct` (final return %, aggregated)
- `security_pl` (PL in security currency)
- `fx_pl` (forex impact, if applicable)
- `entry_fx_rate` (entry FX rate, if applicable)
- `exit_fx_rate` (exit FX rate, if applicable)
- `security_currency` (e.g., USD, GBP, EUR)
- `duration_days` (holding period)
- `entry_equity` (account equity at entry)
- `entry_capital_available` (available capital at entry)
- `entry_reason` (categorical, e.g., "signal_triggered", "breakout", etc.)
- `exit_reason` (categorical, e.g., "take_profit_hit", "stop_loss", "time_based", etc.)
- `commission_paid` (included in PL, can be ignored for analysis)
- `partial_exits` (boolean or count; PL already aggregated)

**Assumptions/Constraints:**
- All timestamps are in the **same time zone**, but may have different frequencies (daily, weekly).
- `pl` and `pl_pct` represent **final trade outcome**, aggregated across any partial exits.
- FX rates and currency are documented but analysis focuses on **security-level performance**.
- Trade IDs must be **unique within a log** but may overlap across logs (handle via metadata tagging).

### 3.2 Price & Indicator Data

**Daily & Weekly OHLCV + Technical Indicators:**

**Columns:**
- `date` (YYYY-MM-DD format)  
- `open`, `high`, `low`, `close`, `volume`  
- **Indicators:** AD, ADOSC, ADX, AROON, ATR, Bollinger Bands, BOP, CCI, CMO, DEMA, DX, EMA, HT_*, KAMA, MACD, MFI, Momentum, NATR, OBV, PPO, ROC, RSI, SAR, SMA, Stochastic, T3, TEMA, TRIX, ULTOSC, WILLR, WMA.

**Key Characteristics:**
- Indicators are sourced from a **reputable provider** and computed on **close prices**.
- Trades are entered at **day close**, using indicator values from that same close.
- Indicator values at `entry_date` are **known at trade time**; no access to future bars.

**Critical Rule:**
- For a trade on `entry_date`, use indicator values from the same `entry_date` (computed on that close).
- Do **not** look ahead to post-entry bars.

### 3.3 Fundamental Data

**Columns (representative, not exhaustive):**

*Valuation Metrics:*
- `pe_ratio`, `forward_pe`, `trailing_pe`, `peg_ratio`, `price_to_book`, `price_to_sales_ttm`, `dividend_yield`

*Profitability:*
- `profit_margin`, `operating_margin_ttm`, `gross_profit_ttm`, `ebit`, `ebitda`, `netincome`

*Growth:*
- `revenue_growth_yoy`, `earnings_growth_yoy`, `revenue_ttm`, `eps`, `estimated_eps`, `earnings_surprise`, `surprise_pct`

*Quality & Financial Health:*
- `return_on_equity_ttm`, `return_on_assets_ttm`, `book_value`, `total_assets`, `totalliabilities`, `totalshareholderequity`, `currentratio`, `debt_to_equity` (derived)

*Cash Flow:*
- `operatingcashflow`, `cashflowfromfinancing`, `cashflowfrominvestment`, `capitalexpenditures`

*Other:*
- `beta`, `dividend_per_share`, `dividendpayout`, `common_stock_shares_outstanding`

**Timing Columns:**
- `date` (record date)  
- `fiscaldateending` (fiscal period end)  
- `report_date` (when fundamentals became public knowledge)  
- `report_type` (quarterly, annual, etc.)  
- `reporttime` (time of day, if available)

**Critical Rule for Forward-Looking Bias:**
- Use **`report_date`** as the canonical "as-of" date.
- Strict rule: **Only fundamentals with `report_date` ≤ trade `entry_date`** are allowed for that trade.
- If `report_date` is missing, use `fiscaldateending` + a configurable standard reporting delay (e.g., +45 days for quarterly earnings).

### 3.4 Insider Data

**Columns:**
- `date` (transaction date, when filed)  
- `symbol` (ticker)  
- `insider_title` (role, e.g., CEO, Director, Officer)  
- `transaction_type` (buy / sell)  
- `shares` (quantity transacted)  
- `price` (transaction price)  
- `value` (shares × price)  
- `executive` (boolean, whether executive level or board member)  
- `security_type` (stock, option, etc.)

**Timing Rule:**
- Insider transactions are filed with a delay (typically 2–5 business days).
- Use **filing date + configurable reporting delay** (default: 3 days) to represent when the market could have known.
- Strict rule: insider data becomes usable only after `filing_date + reporting_delay ≤ trade entry_date`.

### 3.5 Options Data

**Columns:**
- `snapshot_date` (date of options snapshot)  
- `symbol` (ticker)  
- `option_type` (call / put)  
- `expiration_date` (option expiry)  
- `strike` (strike price)  
- `days_to_expiration`  
- `bid`, `ask`, `last_price`, `mark`, `volume`, `open_interest`  
- **Greeks:** `implied_volatility`, `delta`, `gamma`, `vega`, `theta`, `rho`  
- `bid_size`, `ask_size`, `contractid`, `date` (alternative date column)

**Data Characteristics:**
- Often **missing** for many symbols, time periods, or underlying prices.
- Provided as yearly CSVs per security.
- Data sparse and may have gaps.

**Usage Rule:**
- For each trade, search for options snapshots **within a 60-day window before and around the trade entry date**.
- Use **`snapshot_date ≤ entry_date`** to avoid look-ahead bias.
- If multiple snapshots exist for the same trade, use the **closest one before entry** (most recent snapshot).

---

## 4. Configuration System & Strategy Profiles

### 4.1 Configuration Sources

1. **YAML/JSON Configuration Files** (version-controlled, reusable, repeatable).  
   - Stored in a dedicated `configs/` directory.
   - Support for multiple profiles per strategy.

2. **GUI Configuration Dialogs** (user-friendly, ephemeral sessions).  
   - Allow users to set parameters interactively.
   - Provide option to save dialog selections as a new profile.

3. **Strategy Profiles** (persisted, named configurations).
   - Named profiles enable quick reuse (e.g., "momentum_value_v2").
   - Include version/timestamp tracking.

### 4.2 Configuration Schema

A configuration object should be serializable and contain:

#### 4.2.1 Strategy Profile

```
{
  "profile_name": "momentum_value_2025",
  "strategy_name": "momentum_value_mix",
  "created_date": "2025-01-01",
  "last_modified": "2025-01-15",
  "description": "Momentum with value filter; trades 1–3 per quarter."
}
```

#### 4.2.2 Trade Classification Settings

```
{
  "trade_classification": {
    "good_threshold_pct": 2.0,          # PL > 2% = good
    "bad_threshold_pct": -1.0,          # PL < -1% = bad
    "indeterminate_max_days": 15,       # indeterminate if PL in [-1%, +2%] AND duration <= 15 days
    "bad_min_days": 20,                 # trade is only "bad" if held >= 20 days (else indeterminate)
    "threshold_type": "absolute"        # absolute or "percentile"
  }
}
```

#### 4.2.3 Data Alignment & Bias Prevention

```
{
  "data_alignment": {
    "fundamentals_reporting_delay_days": 0,
    "insiders_reporting_delay_days": 3,
    "options_lookback_days": 60,
    "price_forward_fill_allowed": true,
    "flag_price_gaps": true
  }
}
```

#### 4.2.4 Factor Sets & Feature Engineering

```
{
  "factor_engineering": {
    "categories": {
      "technical": {
        "enabled": true,
        "include_all": true,            # or specify list of indicators
        "lookback_days": 0,
        "normalization": "zscore"       # or "percentile_rank" or "none"
      },
      "value": {
        "enabled": true,
        "factors": ["pe_ratio", "price_to_book", "price_to_sales_ttm", "peg_ratio", "dividend_yield"],
        "lookback_days": 0,
        "normalization": "zscore"
      },
      "quality": {
        "enabled": true,
        "factors": ["return_on_equity_ttm", "return_on_assets_ttm", "current_ratio", "debt_to_equity"],
        "lookback_days": 0,
        "normalization": "zscore"
      },
      "growth": {
        "enabled": true,
        "factors": ["revenue_growth_yoy", "earnings_growth_yoy", "earnings_surprise"],
        "lookback_days": 0,
        "normalization": "zscore"
      },
      "insider": {
        "enabled": true,
        "aggregation_window_days": 30,  # e.g., insider activity in last 30 days
        "metrics": ["buy_count", "sell_count", "net_shares", "insider_score", "buy_sell_ratio"]
      },
      "options": {
        "enabled": true,
        "metrics": ["implied_volatility", "put_call_ratio", "iv_percentile"],
        "lookback_days": 60
      },
      "regime": {
        "enabled": true,
        "metrics": ["volatility_regime", "trend_regime"]  # computed from price/indicator data
      }
    },
    "null_handling": {
      "fundamental_factors": "skip",       # skip trades if fundamental data unavailable
      "insider_factors": "zero",           # count as 0 if no insider activity
      "options_factors": "impute",         # impute (e.g., forward-fill or cross-sectional mean)
      "price_data": "exclude"              # exclude trades with price data gaps
    },
    "outlier_handling": {
      "enabled": true,
      "method": "flag_and_report",         # or "winsorize" at 95th percentile
      "threshold_zscore": 3.0              # flag values > 3 std devs from mean
    }
  }
}
```

#### 4.2.5 Multi-Log Aggregation

```
{
  "multi_log_aggregation": {
    "aggregation_mode": "stratified",      # or "pooled"
    "metadata_tags": ["strategy", "symbol", "period"],  # dimensions for stratification
    "confounder_controls": [
      "control_symbol_effects",            # include symbol as fixed effect
      "control_period_effects"             # include period/year as fixed effect
    ]
  }
}
```

#### 4.2.6 Statistical Methods (User-Selectable)

```
{
  "statistical_analysis": {
    "tier1_exploratory": {
      "enabled": true,
      "descriptive_stats": true,           # mean, median, IQR, std dev by trade type
      "correlations": true,                # Pearson, Spearman
      "distributions": true                # histograms, KDE plots
    },
    "tier2_hypothesis_tests": {
      "enabled": true,
      "logistic_regression": true,         # factor -> P(good trade)
      "anova": true,                       # factor regimes vs outcomes
      "kruskal_wallis": true,              # non-parametric alternative
      "chi_square": true,                  # categorical associations
      "wilcoxon_mannwhitney": true         # non-parametric pairwise
    },
    "tier3_ml_analysis": {
      "enabled": true,
      "random_forest_importance": true,
      "shap_analysis": true,               # SHAP values for interpretability
      "mutual_information": true,          # non-linear dependencies
      "bayesian_logistic_regression": false  # optional, computationally intensive
    }
  }
}
```

#### 4.2.7 Multiple Testing Control

```
{
  "multiple_testing": {
    "correction_method": "fdr",            # or "bonferroni"
    "alpha_threshold": 0.05,
    "apply_to_all_tiers": true
  }
}
```

#### 4.2.8 Scenario Analysis Settings

```
{
  "scenario_analysis": {
    "scenario_mode": "binary",             # or "automatic_clustering" or "user_guided"
    "min_trades_per_scenario": 20,         # minimum N for validity
    "metric": "calmar_ratio",              # or "sharpe", "win_rate", etc.
    "best_scenario_threshold": 0.5,        # Calmar > 0.5 = best
    "interaction_mode": "user_guided"      # or "exhaustive" for 2-way interactions
  }
}
```

#### 4.2.9 Output Settings

```
{
  "output": {
    "excel_report": {
      "enabled": true,
      "include_summary_sheet": true,
      "include_factor_sheets": true,
      "include_method_details": true,
      "include_trade_details": true,
      "include_charts": true,
      "include_scenario_analysis": true
    },
    "gui_payloads": {
      "enabled": true,
      "json_output_path": "./analysis_output/payloads.json"
    },
    "audit_log": {
      "enabled": true,
      "output_path": "./analysis_output/audit_log.txt",
      "verbosity": "high"                  # high, medium, low
    }
  }
}
```

---

## 5. Data Preprocessing & Enrichment

### 5.1 Input Validation Phase

For each input file (trade log, price data, fundamentals, insider, options):

1. **Schema Validation**  
   - Check that all required columns are present.
   - Validate data types (dates as YYYY-MM-DD, numerics as float/int).
   - Flag any missing columns and provide guidance.

2. **Row-Level Validation**  
   - Check for null/NaN values in critical columns.
   - Validate date ranges (no future dates, consistent progression).
   - Validate numeric ranges (e.g., PL should not be infinitely large).
   - Flag and log any anomalies; do not silently drop rows.

3. **Cross-File Consistency**  
   - Verify that symbols in trade logs match symbols in price/fundamental data.
   - Verify date ranges align (no trades on dates with no price data).
   - Report any misalignments and ask user for remediation.

### 5.2 Trade Classification

Classify each trade as **good / bad / indeterminate** based on configuration:

```
def classify_trade(row, config):
    pl_pct = row['pl_pct']
    duration_days = row['duration_days']
    good_thresh = config['trade_classification']['good_threshold_pct']
    bad_thresh = config['trade_classification']['bad_threshold_pct']
    indet_max_days = config['trade_classification']['indeterminate_max_days']
    bad_min_days = config['trade_classification']['bad_min_days']
    
    if pl_pct > good_thresh:
        return 'good'
    elif pl_pct < bad_thresh:
        if duration_days >= bad_min_days:
            return 'bad'
        else:
            return 'indeterminate'
    else:  # pl_pct between thresholds
        if duration_days <= indet_max_days:
            return 'indeterminate'
        else:
            return 'bad'
```

### 5.3 Temporal Alignment & As-Of Logic

#### 5.3.1 Fundamental Data Alignment

For each trade, find the **most recent fundamental record** with `report_date ≤ (entry_date - fundamentals_reporting_delay_days)`:

```
trade_entry_date = 2025-01-20
fundamentals_reporting_delay_days = 0

Find: max(report_date) where report_date <= 2025-01-20
  and symbol matches and all fundamental metrics are non-null
```

**Logging:**
- If fundamental data found: log `(trade_id, report_date, days_before_entry)`.
- If not found: log warning and mark trade as missing fundamental data.

#### 5.3.2 Insider Data Alignment

For each trade, aggregate insider activity in a **trailing window**:

```
trade_entry_date = 2025-01-20
aggregation_window_days = 30
insiders_reporting_delay_days = 3

Usable window: (entry_date - aggregation_window_days) to (entry_date - reporting_delay_days)
             = 2024-12-21 to 2025-01-17

Count insider buys, sells, etc. in this window.
```

**Logging:**
- Log `(trade_id, insider_activity_count, buys, sells, window_dates)`.
- If zero insider activity: count as 0 (per configuration).

#### 5.3.3 Options Data Alignment

For each trade, find the **closest options snapshot before entry**:

```
trade_entry_date = 2025-01-20
options_lookback_days = 60

Find: max(snapshot_date) where snapshot_date <= entry_date
  and symbol matches and lookback (entry_date - snapshot_date) <= 60 days
```

**Aggregation across strikes/expirations:**
- If multiple snapshots exist for the same trade (e.g., different strikes), **aggregate** by:
  - IV: use median/mean across strikes.
  - Put/Call Ratio: compute from volume-weighted counts.
  - IV Percentile: compute historical IV percentile.

**Logging:**
- Log `(trade_id, snapshot_date, days_before_entry, num_snapshots_aggregated)`.
- If no options data: log as missing and mark trade accordingly.

#### 5.3.4 Price Data Alignment

For each trade's `entry_date`, retrieve the **close price and indicator values** for that date:

```
trade_entry_date = 2025-01-20

Fetch: price_data where date == 2025-01-20
  Include: close, all indicators
```

**Price Gap Handling:**
- If `entry_date` has no price data (e.g., market holiday), use **last known close** (forward-fill) **but flag the trade**.
- Log: `(trade_id, date_with_gap, last_available_date, gap_days)`.

---

## 6. Factor Engineering

### 6.1 Technical Indicators

**Available indicators** from raw price data (all computed at `entry_date`):
- RSI, MACD, ATR, Bollinger Bands, Stochastic, CCI, ADX, EMA, SMA, DEMA, TEMA, etc.

**Feature Engineering:**
- Use indicator values **as-is** (point-in-time) at entry date.
- Optionally compute **change** (delta) over trailing periods (e.g., RSI change over last 5 days).
- Normalize: z-score or percentile rank based on configuration.

### 6.2 Fundamental Value Factors

**Computed from fundamental data at most-recent `report_date`:**

1. **Valuation Ratios**
   - `pe_ratio` (direct)
   - `price_to_book` = `price / book_value_per_share` (derived if needed)
   - `price_to_sales_ttm` = `market_cap / revenue_ttm` (derived)
   - `peg_ratio` (direct if available, else `pe_ratio / earnings_growth_yoy`)
   - `dividend_yield` (direct)

2. **Quality & Profitability**
   - `return_on_equity_ttm` (direct)
   - `return_on_assets_ttm` (direct)
   - `gross_margin` = `gross_profit / revenue` (derived)
   - `operating_margin` = `operating_income / revenue` (direct as `operating_margin_ttm`)
   - `profit_margin` (direct)
   - `debt_to_equity` = `total_liabilities / total_shareholder_equity` (derived)
   - `current_ratio` = `current_assets / current_liabilities` (derived)
   - `fcf_yield` = `operating_cashflow - capex / market_cap` (derived)

3. **Growth Factors**
   - `revenue_growth_yoy` (direct)
   - `earnings_growth_yoy` (direct)
   - `earnings_surprise` = `(reported_eps - estimated_eps) / estimated_eps` (derived or direct)
   - `earnings_surprise_pct` (direct if available)

**Null Handling (per configuration):**
- Default: **Skip** (exclude trade from analysis if any key value missing).
- Optionally: impute with sector median, historical mean, etc.

**Normalization:**
- Z-score (subtract mean, divide by std dev).
- Percentile rank (0–100% within cross-section of all trades).
- Raw (as-is).

### 6.3 Insider Activity Metrics

For a **fixed aggregation window** (e.g., 30 days before entry, accounting for reporting delay):

1. **Count-Based Metrics**
   - `insider_buy_count`: # of buy transactions.
   - `insider_sell_count`: # of sell transactions.
   - `insider_total_count`: total transactions.

2. **Share-Based Metrics**
   - `insider_net_shares`: (shares bought - shares sold).
   - `insider_total_shares`: total shares transacted.

3. **Value-Based Metrics**
   - `insider_buy_value`: $ value of buys.
   - `insider_sell_value`: $ value of sells.
   - `insider_net_value`: buy_value - sell_value.

4. **Sentiment Metrics**
   - `insider_buy_sell_ratio`: buy_count / sell_count (handle division by zero).
   - `insider_score`: weighted score (e.g., executives weighted more than board members).
   - `insider_sentiment`: bullish/neutral/bearish based on ratios.

**Null Handling (per configuration):**
- Default: **Count as 0** (no insider activity detected).
- Optionally: exclude trade from insider analysis.

### 6.4 Options Data Metrics

For options snapshots **within 60 days before entry**, compute:

1. **Volatility Metrics**
   - `implied_volatility`: median IV across strikes/expirations.
   - `iv_percentile`: IV rank (compare to historical IV distribution for the security).
   - `iv_high_low_spread`: difference between highest and lowest IV.

2. **Skew & Greeks**
   - `put_iv_minus_call_iv`: skew metric.
   - `put_call_ratio`: put volume / call volume.
   - `delta_weighted_position`: aggregate delta across open positions.

3. **Market Depth**
   - `bid_ask_spread_pct`: (ask - bid) / mid.
   - `options_open_interest`: total open interest.

**Null Handling (per configuration):**
- Default: **Impute** (forward-fill last available snapshot, or cross-sectional mean).
- Optionally: exclude trade from options analysis.

### 6.5 Market Regime Metrics

Computed from price and indicator data:

1. **Volatility Regime**
   - `atr_percentile`: ATR as percentile of trailing 252-day distribution.
   - `volatility_regime`: classify as low/medium/high (quantile-based).
   - `vix_proxy`: computed from price ATR and ranges.

2. **Trend Regime**
   - `sma_trend`: price relative to 50/200 SMA (above/below/neutral).
   - `momentum_regime`: RSI-based classification (overbought/neutral/oversold).

3. **Market Conditions**
   - `trading_volume_regime`: volume as percentile of trailing mean.

---

## 7. Data Quality Scoring

For each trade, compute a **quality score** based on data completeness and reliability:

```
quality_score = (
    fundamental_data_available * 0.25 +
    insider_data_available * 0.20 +
    options_data_available * 0.15 +
    price_data_available * 0.30 +
    (1 - outlier_flags_count * 0.05)
) * 100

# Result: 0–100%
```

**Per-Factor Flags:**
- `fundamental_missing`: binary.
- `insider_missing`: binary.
- `options_missing`: binary.
- `price_gap_flagged`: binary.
- `outlier_flagged_count`: count of factors flagged as outliers.

**Data Quality Report:**
- Per-factor completion rate (% of trades with data available).
- Per-trade quality score.
- Recommendation: "Exclude trades with quality_score < 60% from analysis" or similar.

---

## 8. Multi-Log Aggregation & Metadata

### 8.1 Metadata Tagging

When ingesting multiple trade logs, attach **metadata** to each trade:

```
trade_id (globally unique, prefixed by log_id)
symbol (ticker)
period (year, quarter, or date range of backtest)
strategy (name, e.g., "momentum_value_mix")
log_id (source log identifier)
```

### 8.2 Aggregation Modes

#### 8.2.1 Pooled Aggregation

- Combine all trades across symbols into a single dataset.
- Optionally apply **confounder controls** (symbol and period as fixed effects in regression).
- **Output:** Single correlation matrix, single set of statistics.
- **Risk:** Confounding by symbol or period. Mitigate by flagging warnings (e.g., "70% of good trades are from MSFT").

#### 8.2.2 Stratified Aggregation

- Analyze separately by symbol, period, or other metadata.
- **Output:** Per-symbol analyses, per-period analyses, plus optional pooled aggregate.
- **Advantage:** Identify symbol-specific or period-specific patterns.
- **Disadvantage:** Smaller sample sizes per stratum.

### 8.3 Confounder Control

If pooling across symbols/periods, optionally **include confounder variables** in regression models:

```
Logistic Regression Model:
  P(good_trade) ~ factor1 + factor2 + ... + symbol_fixed_effect + period_fixed_effect

Random Forest:
  Include symbol_encoded and period_encoded as features, then inspect feature importance
  to see if they dominate (indicating confounding).
```

---

## 9. Statistical Analysis Framework

### 9.1 Tier 1: Exploratory Descriptive Statistics

**Always compute if enabled:**

1. **Per-Outcome Summary**
   - For each trade class (good, bad, indeterminate):
     - Count, mean, median, std dev, min, max of each factor.
     - Interquartile range (IQR).

2. **Correlations**
   - **Pearson correlation** (linear): `trade_class (encoded) vs. all factors`.
   - **Spearman correlation** (rank-based, robust to outliers).
   - **Output:** Correlation matrix, heatmaps.

3. **Visualizations**
   - Histograms of factor distributions by trade class.
   - Kernel Density Estimation (KDE) plots.
   - Box plots by outcome class.

### 9.2 Tier 2: Hypothesis Testing & Regression

**Compute if enabled:**

#### 9.2.1 Logistic Regression

```
Model: P(trade_outcome = "good") = logit(β₀ + β₁*factor1 + β₂*factor2 + ...)

Output per factor:
  - Coefficient (β)
  - Odds Ratio: exp(β)  [e.g., 1.5 = 50% increase in odds per unit increase in factor]
  - Standard Error
  - p-value (raw and corrected)
  - 95% Confidence Interval
  - Significance flag (yes/no after multiple testing correction)
```

**Interpretation:**
- Positive β: factor increases odds of good trade.
- Negative β: factor decreases odds of good trade.
- p-value < α (after correction): statistically significant.

#### 9.2.2 ANOVA (Parametric Test)

```
Null Hypothesis: Mean factor value is equal across all trade outcomes (good, bad, indeterminate).
Alternative: At least one outcome group has a different mean factor value.

Output:
  - F-statistic, p-value
  - Per-group means and confidence intervals
  - Effect size (eta-squared or partial eta-squared)
```

#### 9.2.3 Kruskal-Wallis (Non-Parametric Alternative to ANOVA)

```
Same hypotheses as ANOVA, but uses ranks instead of raw values.
More robust to outliers and non-normal distributions.

Output:
  - H-statistic, p-value
  - Interpretation similar to ANOVA
```

#### 9.2.4 Chi-Square Test (Categorical Data)

```
For categorical factors (e.g., entry_reason, exit_reason):
  Null Hypothesis: Factor and trade outcome are independent.
  Alternative: Association exists.

Output:
  - Chi-square statistic, p-value
  - Contingency table
  - Cramér's V (effect size)
```

#### 9.2.5 Wilcoxon/Mann-Whitney U Test (Pairwise)

```
For comparing two groups (e.g., "good" vs "bad"):
  Output:
    - U-statistic, p-value
    - Effect size (rank-biserial correlation)
```

### 9.3 Tier 3: Machine Learning & Advanced Analysis

**Compute if enabled:**

#### 9.3.1 Random Forest Feature Importance

```
1. Train Random Forest classifier on factors -> trade_outcome.
2. Compute importance (mean decrease in impurity / permutation-based).
3. Rank factors by importance.
4. Output: Feature importances, top N factors.
5. Advantage: Captures non-linear relationships and interactions.
```

#### 9.3.2 SHAP (SHapley Additive exPlanations) Values

```
1. Compute SHAP values for each prediction.
2. For each factor:
   - Average |SHAP value| = feature contribution to predictions.
   - SHAP plot: show impact direction and magnitude per observation.
3. Output: SHAP summary plots, feature interaction plots.
4. Advantage: Model-agnostic, interpretable feature importance.
```

#### 9.3.3 Mutual Information (Non-Linear Dependencies)

```
For each factor:
  MI(factor, trade_outcome) = entropy(outcome) - entropy(outcome | factor)
  
  Measures non-linear association; range [0, ∞).
  MI = 0: independent.
  Higher MI: stronger association.

Output:
  - MI values, ranked by strength
  - Visualization of pairwise MIs
```

#### 9.3.4 Bayesian Logistic Regression (Optional, Computationally Intensive)

```
Model: P(good_trade) ~ logistic(α + Σ βᵢ * factorᵢ)
  where β are random variables with priors.

Output:
  - Posterior distributions of β (credibility intervals).
  - Posterior predictive distributions.
  - Advantage: Quantifies uncertainty, suitable for decision-making.
```

### 9.4 Multiple Testing Correction

**When testing many factors, apply correction:**

1. **FDR (False Discovery Rate) Control** (default)
   - Less conservative; controls expected proportion of false positives.
   - Better for exploratory analysis with many comparisons.
   - **Procedure:** Benjamini-Hochberg.

2. **Bonferroni Correction** (conservative)
   - Adjusts p-value threshold to α / num_tests.
   - More stringent; use when strict control needed.

**Output per method:**
- Adjusted p-values.
- Significance threshold (e.g., α_corrected = 0.0025 for 20 tests with Bonferroni).
- Report both raw and adjusted p-values.

---

## 10. Scenario Analysis & Best/Worst Scenario Detection

### 10.1 Scenario Definition

A **scenario** is a combination of:
- One or more factors in specific ranges/classes (e.g., "P/E < 15 AND IV > 20").
- Associated trade population (all trades matching this scenario).
- Performance metrics (win rate, avg PL%, Calmar, Sharpe, etc.).

### 10.2 Scenario Modes

#### 10.2.1 Binary Scenario Mode

1. For each factor, compute **median** (or user-specified threshold).
2. Create two groups: **above median** and **below median**.
3. Compute performance metrics for each group.
4. Identify which group (above/below) has better performance.

**Example:**
```
Factor: pe_ratio, median = 18.5
Scenario A: PE < 18.5  → 60 trades, 65% win rate, Calmar = 0.72
Scenario B: PE >= 18.5 → 55 trades, 52% win rate, Calmar = 0.38
Conclusion: Low P/E is better.
```

#### 10.2.2 Automatic Clustering Mode

1. Use **K-means** or **hierarchical clustering** on normalized factors.
2. Let K be user-defined (e.g., K=3 for low/medium/high regimes).
3. For each cluster:
   - Describe factor characteristics (e.g., "Cluster 1: high IV, low P/E, positive momentum").
   - Compute performance metrics.
   - Identify if cluster is "best", "worst", or "neutral".

**Example:**
```
Cluster 1 (32 trades): High IV, Low P/E, Positive Momentum
  → 75% win rate, Calmar = 1.1, avg duration = 45 days → BEST

Cluster 2 (25 trades): Low IV, Medium P/E, Neutral Momentum
  → 48% win rate, Calmar = 0.25, avg duration = 22 days → WORST

Cluster 3 (58 trades): Medium IV, High P/E, Negative Momentum
  → 55% win rate, Calmar = 0.52, avg duration = 38 days → NEUTRAL
```

### 10.3 Scenario Filtering & Validation

Apply the following filters to ensure scenario validity:

1. **Minimum Trade Count**
   - Only report scenarios with N_trades ≥ `min_trades_per_scenario` (config, e.g., 20).
   - Log scenarios with N < 20 as "insufficient sample size".

2. **Statistical Significance**
   - For binary scenarios: use Mann-Whitney U or chi-square to test if performance differs significantly.
   - For clusters: use ANOVA or Kruskal-Wallis to test if cluster means differ.
   - Report p-value; flag scenario if not significant.

3. **Confidence Intervals**
   - Compute 95% CI for Calmar, Sharpe, win rate using bootstrap.
   - Report CI width; wide CIs indicate uncertainty.

### 10.4 Best & Worst Scenario Definitions

**Best Scenario:**
- **Calmar Ratio > 0.5** (configured threshold).
- **N_trades ≥ 1 per year** (annualized).
- **Win rate > 50%** (recommended).
- **p-value < 0.05** (statistically significant vs. worst scenario).

**Worst Scenario:**
- **Calmar Ratio ≤ 0.5** (default threshold, user-configurable).
- Low win rate (< 50%).
- Statistically distinguishable from best scenario.

**Neutral Scenario:**
- Neither best nor worst (Calmar 0.3–0.5, moderate win rate).

### 10.5 Interaction Effects (User-Guided)

Users can specify **two-factor interactions**:

```
Example: P/E < 15 AND Insider Buy Count > 5
  → "Low P/E + Insider Buying"
  → Compute performance metrics.
  → Compare to single-factor scenarios.
```

**System role:**
- Enumerate user-specified interactions.
- Compute performance for each.
- Report interaction effects (does combination perform better/worse than sum of parts?).

**Not exhaustive search** (too many combinations), but **user-directed**.

---

## 11. Error Handling & Data Quality Feedback

### 11.1 Input Validation Errors

| Issue | Handling | User Feedback |
|-------|----------|---------------|
| Missing required column | Stop with error | "Column 'pe_ratio' not found in fundamentals. Required columns: [...]" |
| Invalid date format | Flag and skip row | "Row 1025: entry_date '2025-13-40' is invalid; skipped." |
| Negative quantity (long trade) | Flag as warning | "Trade ID 042: negative quantity (short?). Assuming short side." |
| PL > max_reasonable | Flag and log | "Trade ID 089: PL = $999,999,999 seems extreme; flagged as outlier." |
| Symbol mismatch | Stop or skip | "Symbol 'XXYZ' in trade log not found in price data. Check spelling." |

### 11.2 Data Alignment Warnings

| Issue | Handling | User Feedback |
|-------|----------|---------------|
| No fundamental data for trade | Use null_handling policy (skip/impute) | "Trade ID 015: Fundamentals missing. Excluded per configuration." |
| Insider data incomplete for window | Count as zero | "Trade ID 042: No insider activity found in [-30, -3] day window. Set to 0." |
| Options snapshot unavailable | Use imputation | "Trade ID 089: Options data missing. Imputed using cross-section mean." |
| Price gap (market holiday) | Forward-fill and flag | "Trade ID 050: Price data gap on 2025-12-25 (holiday). Forward-filled; trade flagged." |
| Report date in future | Skip | "Trade ID 072: Fundamental report_date (2025-02-15) > entry_date (2025-02-01). Excluded." |

### 11.3 Quality Score Interpretation

```
Quality Score Ranges & Recommended Actions:

> 80%: Excellent   → Include in all analysis
60–80%: Good       → Include but note caveats
40–60%: Fair       → Include with caution; consider stratified analysis
< 40%: Poor        → Exclude from primary analysis; report separately
```

### 11.4 Outlier Handling

**Detection:**
- Flag any factor value >3 standard deviations from mean.
- Count outliers per trade.

**Reporting:**
```
Trade ID 025:
  - pe_ratio = 250 (zscore = 4.2)  ← OUTLIER
  - earnings_surprise = 450%        ← OUTLIER
  Quality Impact: -0.10 (reduced from 0.85 to 0.75)
  Recommendation: Include with caution; or exclude from analysis.
```

**Handling Options (per configuration):**
- **Flag and report** (default): log but keep in analysis.
- **Winsorize**: cap at 95th percentile.
- **Exclude**: remove from analysis.

---

## 12. Audit Trail & Logging

### 12.1 Comprehensive Logging

Log **every decision** to a text file with **high verbosity**:

```
[2025-01-20 14:23:15] === FACTOR ANALYSIS SESSION START ===
[2025-01-20 14:23:15] Profile: momentum_value_2025
[2025-01-20 14:23:15] Input Files:
  - trade_log: /data/trades_2024.csv (1,250 rows)
  - fundamentals: /data/fundamentals.csv (18,456 rows)
  - insider: /data/insider.csv (5,203 rows)
  - options: /data/options_2024.csv (89,234 rows)
  - price_data: /data/price_indicators.csv (3,654 rows)

[2025-01-20 14:23:20] TRADE CLASSIFICATION:
  - good_threshold_pct: 2.0
  - bad_threshold_pct: -1.0
  - Classified: 745 good, 341 bad, 164 indeterminate

[2025-01-20 14:24:10] FUNDAMENTAL DATA ALIGNMENT:
  - Trades with fundamentals: 1,198 / 1,250 (95.8%)
  - Missing: 52 trades (logged separately)

[2025-01-20 14:25:30] OUTLIER DETECTION:
  - Total outliers flagged: 34
  - Trade 089 (pe_ratio = 250): zscore = 4.2
  - Trade 156 (earnings_surprise = 450%): zscore = 5.1

[2025-01-20 14:26:00] FACTOR ENGINEERING COMPLETE:
  - Technical factors: 47 indicators enabled
  - Fundamental factors: 25 ratios computed
  - Insider factors: 5 metrics computed
  - Options factors: 4 metrics computed (1,045 / 1,250 trades with data)

[2025-01-20 14:27:30] CORRELATION ANALYSIS:
  - Pearson correlation: pe_ratio vs. trade_outcome → r = -0.18, p < 0.001
  - Spearman correlation: insider_sentiment vs. trade_outcome → ρ = 0.22, p < 0.001

[2025-01-20 14:35:00] LOGISTIC REGRESSION RESULTS:
  - Model: P(good) ~ pe_ratio + insider_sentiment + momentum_regime + ... + symbol_effect
  - Iterations: 120 (converged)
  - Factor: pe_ratio, β = -0.0234, OR = 0.977, p = 0.004 (FDR-corrected) → SIGNIFICANT
  - ... [rest of coefficients]

[2025-01-20 15:15:30] SCENARIO ANALYSIS:
  - Mode: binary
  - Factor: pe_ratio, threshold: 18.5 (median)
  - Scenario A (PE < 18.5): 600 trades, 65.2% win rate, Calmar = 0.72 ← BEST
  - Scenario B (PE >= 18.5): 650 trades, 52.1% win rate, Calmar = 0.31 ← WORST
  - p-value (Mann-Whitney): 0.001 (significant)

[2025-01-20 15:45:00] === ANALYSIS COMPLETE ===
[2025-01-20 15:45:05] Output Files Generated:
  - Excel Report: /output/factor_analysis_2025-01-20.xlsx
  - Audit Log: /output/audit_log_2025-01-20.txt
```

### 12.2 Configuration Snapshot

Save the exact configuration used for reproducibility:

```
CONFIGURATION SNAPSHOT
Generated: 2025-01-20 14:23:15
Profile Name: momentum_value_2025
Profile Version: 2.1
MD5 Hash: a1b2c3d4e5f6...

[Full configuration YAML/JSON appended]
```

---

## 13. Output Formats & Report Generation

### 13.1 Excel Report Structure

**Workbook with multiple sheets:**

#### Sheet 1: Executive Summary
- Analysis metadata (date, profile, input files, record counts).
- Key findings (e.g., "P/E ratio is a significant predictor of trade success").
- Best and worst scenarios (1–2 tables).
- High-level statistics (% good trades, avg win rate, overall Sharpe).

#### Sheet 2: Data Quality Report
- Per-factor completion rates (% of trades with data).
- Per-trade quality scores (distribution histogram).
- Missing data summary table (by data source and factor).
- Recommendation: "Exclude trades with quality_score < 60%".

#### Sheet 3: Descriptive Statistics
- Summary table: mean, median, std dev, min, max for each factor, stratified by trade outcome (good / bad / indeterminate).
- Visualization: box plots or summary charts.

#### Sheet 4: Correlation Analysis
- Correlation matrix (Pearson and Spearman).
- Heatmap visualization.
- Interpretation: "Values close to 1 or -1 indicate strong association."

#### Sheet 5–N: Per-Factor Analysis Sheets (if Tier 2/3 enabled)
- One sheet per factor or statistical method.
- Logistic Regression Results: coefficients, p-values, odds ratios.
- ANOVA/Kruskal-Wallis: F-statistic, p-values, effect sizes.
- Random Forest Importances: ranked features with bar chart.
- SHAP summaries: feature impact visualization.

#### Sheet (N+1): Scenario Analysis
- Best scenarios: definition, N trades, win rate, Calmar, confidence intervals, p-values.
- Worst scenarios: same metrics.
- Interaction scenarios (if user-guided interactions enabled).

#### Sheet (N+2): Trade Details (Optional)
- Detailed trade-level data (trade_id, symbol, entry_date, exit_date, pl_pct, classification, factor values, data_quality_score).
- Sortable/filterable by outcome, factor, symbol, etc.

#### Sheet (N+3): Method Documentation
- Tier 1 explanation: "Descriptive statistics summarize factor distributions per trade outcome."
- Tier 2 explanation: "Logistic regression models P(good trade) as a function of factors; p-values indicate significance after multiple testing correction (FDR)."
- Tier 3 explanation: "Random Forest captures non-linear relationships. SHAP values explain per-prediction contributions."
- Assumptions: "Observations are independent; logistic regression assumes no multicollinearity."
- Limitations: "Limited sample size in Scenario B may inflate confidence; consider with caution."

### 13.2 Excel Formatting

- **Headers:** Bold, colored background (teal/green).
- **Significant results:** Green background (p < 0.05 post-correction).
- **Non-significant results:** Gray or default.
- **Tables:** Borders, alternating row colors.
- **Charts:** Embedded (bar, box, scatter, heatmap as appropriate).
- **Numbers:** 2–4 decimal places; percentages with % sign.

### 13.3 JSON Payloads for GUI

Structure JSON outputs for easy consumption by GUI:

```json
{
  "analysis_metadata": {
    "analysis_date": "2025-01-20",
    "profile_name": "momentum_value_2025",
    "input_files": ["trades_2024.csv", "fundamentals.csv", ...],
    "record_counts": {
      "total_trades": 1250,
      "good_trades": 745,
      "bad_trades": 341,
      "indeterminate_trades": 164
    }
  },
  "data_quality": {
    "completion_rates": {
      "fundamental": 0.958,
      "insider": 0.920,
      "options": 0.836,
      "price": 0.995
    },
    "quality_score_distribution": {
      "mean": 78.5,
      "median": 82.0,
      "std_dev": 15.3
    }
  },
  "correlation_analysis": {
    "pearson_correlations": [
      {"factor": "pe_ratio", "outcome": "good_trade", "r": -0.18, "p_value": 0.0008},
      {"factor": "insider_sentiment", "outcome": "good_trade", "r": 0.22, "p_value": 0.0002},
      ...
    ],
    "spearman_correlations": [...]
  },
  "logistic_regression_results": {
    "factors": [
      {
        "name": "pe_ratio",
        "coefficient": -0.0234,
        "odds_ratio": 0.977,
        "se": 0.0089,
        "p_value_raw": 0.0043,
        "p_value_fdr_corrected": 0.0084,
        "ci_lower": -0.0407,
        "ci_upper": -0.0061,
        "significant": true
      },
      ...
    ],
    "model_statistics": {
      "aic": 1456.23,
      "bic": 1589.45,
      "n_obs": 1250
    }
  },
  "scenario_analysis": {
    "scenarios": [
      {
        "id": "scenario_001",
        "name": "Low P/E",
        "definition": "pe_ratio < 18.5",
        "n_trades": 600,
        "win_rate": 0.652,
        "calmar_ratio": 0.72,
        "sharpe_ratio": 1.45,
        "calmar_ci_lower": 0.61,
        "calmar_ci_upper": 0.83,
        "classification": "best",
        "sample_trades": [
          {"trade_id": "T001", "entry_date": "2024-01-15", "pl_pct": 5.2, "factors": {...}},
          ...
        ]
      },
      ...
    ]
  },
  "feature_importance": {
    "method": "random_forest",
    "top_features": [
      {"name": "pe_ratio", "importance": 0.185},
      {"name": "insider_sentiment", "importance": 0.143},
      {"name": "momentum_regime", "importance": 0.128},
      ...
    ]
  }
}
```

---

## 14. Configuration File Examples

### 14.1 Example YAML Profile

```yaml
# momentum_value_2025.yml
profile_name: momentum_value_2025
strategy_name: momentum_value_mix
description: Momentum with value filter, trades 1–3 per quarter
created_date: 2025-01-01

trade_classification:
  good_threshold_pct: 2.0
  bad_threshold_pct: -1.0
  indeterminate_max_days: 15
  bad_min_days: 20
  threshold_type: absolute

data_alignment:
  fundamentals_reporting_delay_days: 0
  insiders_reporting_delay_days: 3
  options_lookback_days: 60
  price_forward_fill_allowed: true
  flag_price_gaps: true

factor_engineering:
  categories:
    technical:
      enabled: true
      include_all: true
      lookback_days: 0
      normalization: zscore
    value:
      enabled: true
      factors: ["pe_ratio", "price_to_book", "price_to_sales_ttm", "peg_ratio"]
      lookback_days: 0
      normalization: zscore
    growth:
      enabled: true
      factors: ["revenue_growth_yoy", "earnings_growth_yoy", "earnings_surprise"]
      lookback_days: 0
      normalization: percentile_rank
    insider:
      enabled: true
      aggregation_window_days: 30
      metrics: ["buy_count", "sell_count", "insider_score"]
    options:
      enabled: true
      metrics: ["implied_volatility", "put_call_ratio"]
      lookback_days: 60
  null_handling:
    fundamental_factors: skip
    insider_factors: zero
    options_factors: impute
    price_data: exclude
  outlier_handling:
    enabled: true
    method: flag_and_report
    threshold_zscore: 3.0

multi_log_aggregation:
  aggregation_mode: stratified
  metadata_tags: [strategy, symbol, period]
  confounder_controls: [control_symbol_effects]

statistical_analysis:
  tier1_exploratory:
    enabled: true
    descriptive_stats: true
    correlations: true
    distributions: true
  tier2_hypothesis_tests:
    enabled: true
    logistic_regression: true
    anova: true
    kruskal_wallis: true
    chi_square: true
  tier3_ml_analysis:
    enabled: true
    random_forest_importance: true
    shap_analysis: true
    mutual_information: true
    bayesian_logistic_regression: false

multiple_testing:
  correction_method: fdr
  alpha_threshold: 0.05
  apply_to_all_tiers: true

scenario_analysis:
  scenario_mode: binary
  min_trades_per_scenario: 20
  metric: calmar_ratio
  best_scenario_threshold: 0.5
  interaction_mode: user_guided

output:
  excel_report:
    enabled: true
    include_summary_sheet: true
    include_factor_sheets: true
    include_method_details: true
    include_trade_details: false
    include_charts: true
    include_scenario_analysis: true
  gui_payloads:
    enabled: true
    json_output_path: ./analysis_output/payloads.json
  audit_log:
    enabled: true
    output_path: ./analysis_output/audit_log.txt
    verbosity: high
```

---

## 15. Integration with Backtesting Framework

### 15.1 API Specification

```python
# Entry point for analysis
from factor_analysis import FactorAnalyzer, ProfileManager

# Load configuration
profile_manager = ProfileManager(config_dir="./configs")
config = profile_manager.load("momentum_value_2025")

# Initialize analyzer
analyzer = FactorAnalyzer(config)

# Load data
trade_logs = [
  "./data/trades_AAPL_2024.csv",
  "./data/trades_MSFT_2024.csv"
]
raw_data = {
  "price_indicators": "./data/price_indicators_daily.csv",
  "fundamentals": "./data/fundamentals.csv",
  "insider": "./data/insider.csv",
  "options": "./data/options_2024.csv"
}

# Run analysis
result = analyzer.analyze_trades(
  trade_logs=trade_logs,
  data_sources=raw_data,
  stratify_by=["symbol", "period"],
  output_dir="./output"
)

# Export outputs
analyzer.export_excel_report(result, "./output/factor_analysis.xlsx")
gui_payloads = analyzer.prepare_gui_payloads(result)

# Save for future reference
profile_manager.save(config, "momentum_value_2025_v2")
```

### 15.2 Module Structure

```
backtesting_framework/
├── core/
│   ├── backtester.py
│   └── ...
├── factor_analysis/
│   ├── __init__.py
│   ├── analyzer.py              # Main FactorAnalyzer class
│   ├── preprocessor.py          # Data loading, validation, enrichment
│   ├── factor_engineer.py       # Factor computation
│   ├── statistical_analysis.py  # Tier 1, 2, 3 analyses
│   ├── scenario_analysis.py     # Scenario detection, best/worst
│   ├── report_generator.py      # Excel & JSON output
│   ├── config.py                # Configuration & ProfileManager
│   ├── logger.py                # Audit trails and logging
│   └── utils.py                 # Helper functions
├── configs/
│   ├── momentum_value_2025.yml
│   ├── mean_reversion_2025.yml
│   └── ...
└── output/
    ├── factor_analysis_2025-01-20.xlsx
    ├── payloads_2025-01-20.json
    └── audit_log_2025-01-20.txt
```

---

## 16. Validation & Testing Strategy

### 16.1 Data Validation Tests

1. **Temporal Consistency**
   - Verify no trades on dates with missing price data.
   - Verify report_date ≤ trade entry_date for fundamentals.
   - Verify insider filing_date + delay ≤ trade entry_date.

2. **Value Validation**
   - Check PL values are reasonable (not infinities or extreme outliers).
   - Check factor values are within expected ranges.
   - Verify dates parse correctly.

3. **Cross-File Alignment**
   - Verify all symbols in trade logs exist in price data.
   - Verify all trade dates have corresponding price data (or are flagged).
   - Verify fundamental dates align with reporting conventions.

### 16.2 Statistical Tests

1. **Result Reproducibility**
   - Same input data + same config → identical output (deterministic random seeds).
   - Version control factor definitions; changes produce audit trail.

2. **Sanity Checks**
   - Correlation values in [-1, 1].
   - p-values in [0, 1].
   - Logistic regression probabilities in [0, 1].
   - Calmar/Sharpe ratios reasonable (e.g., < 5).

### 16.3 End-to-End Integration Tests

1. **Sample Backtests**
   - Create synthetic trades with known good/bad/indeterminate outcomes.
   - Create synthetic factors with known correlations.
   - Run analysis; verify detected correlations match synthetic setup.

2. **Real Data Tests**
   - Run analysis on historical backtest data.
   - Verify outputs match manual spot-checks.

---

## 17. Performance & Scalability

### 17.1 Expected Scale

- **Trade Logs:** 1–3 trades per quarter × 15 years = 45–180 trades per symbol.
- **Multiple Symbols:** 2–5 symbols per analysis session.
- **Total Trades:** 90–900 trades per analysis run.
- **Factors:** 50–100+ derived factors.

### 17.2 Computation Strategy

1. **Data Loading & Preprocessing:** O(n log m) where n = trades, m = data rows.
   - Efficient pandas merges and vectorized operations.

2. **Correlation & Descriptive Stats:** O(n × p) where p = # factors.
   - Vectorized with numpy/pandas; <1 second for typical scale.

3. **Logistic Regression:** O(n × p) iterations × 100–200 iterations.
   - Typically <10 seconds with scikit-learn.

4. **Random Forest:** O(n × p × trees × depth) ≈ O(n × p × 1000).
   - Typically 30–60 seconds with 100+ trees.

5. **SHAP Analysis:** O(n × p × 2^p) in worst case; approximations reduce to O(n × p × 100).
   - Can take 2–5 minutes for large factor sets; use background thread in GUI.

**Total Typical Runtime:** 5–30 minutes (acceptable per requirement: max 2 hours).

### 17.3 Memory Usage

- Trade log (900 rows × 25 columns) ≈ 1 MB.
- Raw data (10 years × 252 days × 100 columns) ≈ 100 MB.
- Enriched trades (900 × 100 factors) ≈ 1 MB.
- Total: ~100 MB in-memory; well within typical laptop memory.

### 17.4 Optimization Recommendations

1. Use **pandas for vectorized operations** (avoid Python loops).
2. Cache **loaded data** to avoid re-reading files.
3. Run **SHAP and advanced analyses in background threads** (don't block GUI).
4. Use **sparse matrices** if factor set becomes very large (100s of factors).

---

## 18. Robustness & Edge Case Handling

### 18.1 Common Edge Cases

| Scenario | Handling |
|----------|----------|
| Single symbol only | Bypass stratification; report across all trades. |
| No insider data for any trade | Disable insider analysis; note in report. |
| All trades good / no bad trades | Scenario analysis shows single scenario only; note class imbalance. |
| Highly correlated factors | Report multicollinearity warning; use regularization in regression. |
| Tiny scenario (N=5 trades) | Exclude from best/worst scenarios; report in footnote. |
| Extreme outlier factor | Winsorize or exclude; log decision in audit trail. |
| Missing entire data source (e.g., no options file) | Skip options enrichment gracefully; continue analysis. |

### 18.2 Handling Class Imbalance

If good/bad/indeterminate classes are severely imbalanced:

1. **Report imbalance ratio** in summary (e.g., "7:3:2 good:bad:indeterminate").
2. **Logistic regression:** use **class weights** to balance training.
3. **Random Forest:** use **balanced class weights** option.
4. **Interpretation:** Account for imbalance when evaluating statistical significance.

---

## 19. Documentation & Reporting Standards

### 19.1 Report Metadata (Required in Every Report)

```
Analysis Report Metadata
═════════════════════════════════════════
Generated: 2025-01-20 15:45:00 UTC
Analysis Tool Version: Factor Analysis v1.2.3
Profile Name: momentum_value_2025
Profile Version: 2.1
Profile MD5: a1b2c3d4e5f6...

Input Files
───────────
Trade Log Files (2):
  - /data/trades_AAPL_2024.csv (750 rows)
  - /data/trades_MSFT_2024.csv (500 rows)
  Total Trades: 1,250

Raw Data Files:
  - Price/Indicators: /data/price_indicators_daily.csv (3,654 rows, 105 columns)
  - Fundamentals: /data/fundamentals.csv (18,456 rows)
  - Insider: /data/insider.csv (5,203 rows)
  - Options: /data/options_2024.csv (89,234 rows)

Configuration Applied
─────────────────────
Trade Classification:
  good_threshold: 2.0%
  bad_threshold: -1.0%
  indeterminate_max_days: 15
  bad_min_days: 20

Data Alignment:
  fundamentals_reporting_delay: 0 days
  insiders_reporting_delay: 3 days
  options_lookback: 60 days

Enabled Analyses:
  ✓ Tier 1 (Exploratory)
  ✓ Tier 2 (Hypothesis Testing)
  ✓ Tier 3 (ML Analysis)

Multiple Testing Correction: FDR (alpha = 0.05)

Aggregation Mode: Stratified (by symbol)
Confounder Controls: symbol_effects

Trade Classifications
─────────────────────
Good Trades: 745 (59.6%)
Bad Trades: 341 (27.3%)
Indeterminate: 164 (13.1%)
Total: 1,250

Data Quality Summary
────────────────────
Fundamental Data: 95.8% coverage (1,198 / 1,250 trades)
Insider Data: 92.0% coverage (1,150 / 1,250 trades)
Options Data: 83.6% coverage (1,045 / 1,250 trades)
Price Data: 99.5% coverage (1,244 / 1,250 trades)
Average Quality Score: 78.5% ± 15.3%

Key Findings
────────────
1. P/E Ratio is a significant negative predictor of trade success.
   (OR = 0.977, p = 0.0084, 95% CI = [-0.0407, -0.0061])

2. Insider buying activity (30-day window) is a significant positive predictor.
   (OR = 1.34, p = 0.0032, 95% CI = [1.11, 1.62])

3. Best Scenario: Low P/E (< 18.5) + Insider Buying
   Calmar = 0.72, Win Rate = 65.2%, N = 125 trades

4. Worst Scenario: High P/E (>= 18.5) + No Insider Activity
   Calmar = 0.18, Win Rate = 42.1%, N = 210 trades

Researcher Notes
────────────────
[Any additional context or caveats]

═════════════════════════════════════════
```

### 19.2 Method Documentation (Per Sheet/Section)

For each statistical method employed, include:

1. **Plain-Language Explanation**
   - "Logistic Regression: Models the probability of a trade being 'good' as a function of factors."

2. **Mathematical Formula** (if appropriate)
   - P(Y=1) = logit(α + Σ βᵢ × Xᵢ)

3. **Key Assumptions**
   - Independence of observations.
   - No severe multicollinearity among factors.
   - Sufficient sample size per group.

4. **Limitations & Caveats**
   - "Results assume no unmeasured confounders (e.g., sector effects not explicitly controlled)."
   - "Small scenario sizes (N<30) should be interpreted with caution."

5. **Interpretation Guide**
   - "Odds Ratio > 1: factor increases odds of good trade."
   - "p-value < 0.05 (after FDR correction): statistically significant at 5% level."

---

## 20. Success Criteria & Deliverables

### 20.1 Functional Requirements

- ✅ Load and validate trade logs, price/indicator data, fundamentals, insider, options.
- ✅ Classify trades as good / bad / indeterminate per configuration.
- ✅ Enforce strict temporal alignment with configurable delays (no forward-looking bias).
- ✅ Engineer 50+ derived factors (technical, fundamental, insider, options, regime).
- ✅ Compute data quality scores and comprehensive missing-data handling.
- ✅ Support Tier 1, 2, 3 statistical analyses (user-selectable).
- ✅ Apply multiple testing correction (FDR / Bonferroni).
- ✅ Detect best/worst scenarios with statistical validation.
- ✅ Support multi-log aggregation with stratification and confounder controls.
- ✅ Export professional Excel reports with formatting, charts, explanatory text.
- ✅ Generate JSON payloads for GUI consumption.
- ✅ Provide comprehensive audit trails and logging.
- ✅ Support strategy profile persistence and versioning.
- ✅ Handle edge cases (missing data, class imbalance, outliers, etc.).
- ✅ Complete analysis in <2 hours (typical: 5–30 minutes).

### 20.2 Deliverables

1. **Factor Analysis Module** (Python)
   - Preprocessor, Factor Engineer, Statistical Analyzer, Scenario Detector, Report Generator, Configuration Manager.
   - Clean API for integration with backtesting framework.

2. **Configuration System**
   - YAML/JSON support, ProfileManager, example profiles.

3. **Documentation**
   - High-level design (this document).
   - Inline code documentation.
   - User guide (how to create/load profiles, interpret results).

4. **Examples & Tests**
   - Sample trade logs and raw data.
   - Unit tests for data validation, factor engineering, statistical methods.
   - Integration tests (end-to-end analysis runs).

---

## 21. Future Enhancements

(Out of scope for this plan, but noted for roadmap):

1. **Real-Time Analysis:** Stream trades and factors as they occur.
2. **Portfolio-Level Analysis:** Extend from per-trade to portfolio-level metrics (drawdown, portfolio Sharpe, etc.).
3. **Causal Inference:** Use causal graphs or Bayesian networks to infer causal relationships (not just correlation).
4. **Optimization:** Suggest factor thresholds that maximize Calmar/Sharpe.
5. **Deep Learning:** Neural networks or gradient boosting for non-linear factor relationships.
6. **A/B Testing:** Compare two strategy versions on the same data.
7. **Live Dashboard:** Real-time scenario monitoring with alerts.

---

## 22. Appendix: Data Format Examples

### 22.1 Trade Log Sample

```csv
trade_id,symbol,entry_date,entry_price,exit_date,exit_price,quantity,side,initial_stop_loss,final_stop_loss,take_profit,pl,pl_pct,security_pl,fx_pl,entry_fx_rate,exit_fx_rate,security_currency,duration_days,entry_equity,entry_capital_available,entry_reason,exit_reason,commission_paid,partial_exits
T001,AAPL,2024-01-15,150.23,2024-02-10,157.89,100,long,148.00,149.50,160.00,750.00,5.08,750.00,0.00,1.00,1.00,USD,26,50000,10000,signal_triggered,take_profit_hit,25.00,false
T002,MSFT,2024-01-18,310.45,2024-02-05,304.12,50,long,305.00,306.50,320.00,-315.00,-1.02,-315.00,0.00,1.00,1.00,USD,18,50000,10000,breakout,stop_loss_hit,15.00,false
T003,AAPL,2024-02-20,155.67,2024-03-15,163.45,100,long,153.00,154.20,165.00,778.00,5.01,778.00,0.00,1.00,1.00,USD,23,50000,10000,momentum_signal,time_based,25.00,true
```

### 22.2 Fundamental Data Sample (Condensed)

```csv
date,symbol,report_date,pe_ratio,price_to_book,dividend_yield,revenue_growth_yoy,earnings_growth_yoy,return_on_equity_ttm,return_on_assets_ttm,debt_to_equity
2024-01-15,AAPL,2024-01-10,28.5,45.2,0.52,8.3,12.1,105.8,20.5,1.23
2024-01-18,MSFT,2024-01-12,32.1,10.5,0.89,15.2,18.3,42.3,15.2,0.67
2024-02-20,AAPL,2024-02-14,29.1,46.8,0.51,9.1,11.8,106.2,20.8,1.25
```

### 22.3 Insider Data Sample

```csv
date,symbol,insider_title,transaction_type,shares,price,value,executive,security_type
2024-01-10,AAPL,CEO,buy,10000,145.50,1455000,true,stock
2024-01-12,AAPL,Director,sell,5000,148.23,741150,false,stock
2024-01-15,MSFT,CFO,buy,3000,308.50,925500,true,stock
```

### 22.4 Options Data Sample (Condensed)

```csv
snapshot_date,symbol,option_type,expiration_date,strike,implied_volatility,delta,bid,ask
2024-01-14,AAPL,call,2024-02-16,150,0.25,0.58,3.20,3.45
2024-01-14,AAPL,put,2024-02-16,150,0.24,-0.42,2.80,3.10
2024-01-17,MSFT,call,2024-02-16,310,0.22,0.61,4.10,4.35
```

---

**End of Design Plan Document**

---

## Summary

This **comprehensive design plan** specifies:

1. **Objectives:** Explain trade performance via factors; identify best/worst scenarios; support multi-log analysis; avoid forward-looking bias; deliver professional outputs.

2. **Architecture:** Modular subsystem with 5 layers (input, preprocessing, analysis, output, configuration).

3. **Data Handling:**
   - Strict temporal alignment with configurable reporting delays.
   - Multi-source enrichment (price, fundamentals, insider, options).
   - Comprehensive null/missing data handling per source.
   - Data quality scoring and audit trails.

4. **Feature Engineering:** 50+ derived factors across 6 categories (technical, value, quality, growth, insider, options, regime).

5. **Statistical Analysis:**
   - **Tier 1:** Exploratory (descriptive stats, correlations, distributions).
   - **Tier 2:** Hypothesis testing (logistic regression, ANOVA, chi-square).
   - **Tier 3:** ML-based (Random Forest, SHAP, Mutual Information, optional Bayesian).
   - Multiple testing correction (FDR/Bonferroni).

6. **Scenario Analysis:** Binary or automatic clustering, with statistical validation, best/worst definitions, and user-guided interactions.

7. **Multi-Log Support:** Metadata tagging, optional stratification, confounder control.

8. **Outputs:** Professional Excel reports with formatting/charts; JSON payloads for GUI; comprehensive audit trails.

9. **Robustness:** Error handling, data quality feedback, outlier detection, edge case handling, reproducibility via configuration snapshots.

10. **Performance:** Handles 90–900 trades with 50–100 factors in 5–30 minutes (< 2-hour requirement).

This plan is **ready for implementation** by Claude, with sufficient detail to avoid ambiguity while remaining high-level (no code).
