# Strategy Performance Factor Analysis Module – Design Plan

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
   - **Excel reports** with formatting, charts, tables, and explanatory text.

5. **Avoid Forward-Looking Bias & Data Leakage**  
   - Enforce **strict temporal alignment** using report_date, filing dates, and snapshot dates with configurable lags.
   - Make all as-of decisions explicit and logged.

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

3. **Analysis Layer**  
   - Descriptive statistics.
   - Hypothesis testing & regression analysis.
   - Feature importance & scenario discovery (ML-based where chosen).
   - Best/worst scenario evaluation using Calmar and related metrics.

4. **Output Layer**  
   - Data structures for GUI consumption.
   - Excel report generator.
   - Logging, audit trails, and configuration snapshots.

5. **Configuration & Profiles Layer**  
   - YAML/JSON configuration support.
   - Strategy profiles (persisted configurations for different strategies/use-cases).


### 2.2 Module Boundaries

The Factor Analysis module should:

- **Not modify** the core backtesting engine.  
- **Depend only on** outputs already produced by the backtester (trade logs and raw data).  
- Provide a clean API such as:

- `analyze_trades(config, trade_logs, data_sources) -> AnalysisResult`
- `export_excel_report(analysis_result, output_path)`
- `prepare_gui_payloads(analysis_result) -> dict`


---

## 3. Input Data and Assumptions

### 3.1 Trade Logs

**Format:** CSV, one file per backtest per security.

**Columns (given):**
- `trade_id`  
- `symbol` (ticker)  
- `entry_date`  
- `entry_price`  
- `exit_date`  
- `exit_price`  
- `quantity`  
- `side` (long / short)  
- `initial_stop_loss`  
- `final_stop_loss`  
- `take_profit`  
- `pl`  
- `pl_pct`  
- `security_pl`  
- `fx_pl`  
- `entry_fx_rate`  
- `exit_fx_rate`  
- `security_currency`  
- `duration_days`  
- `entry_equity`  
- `entry_capital_available`  
- `entry_reason` (categorical)  
- `exit_reason` (categorical)  
- `commission_paid` (already included in PL, can be ignored for PL-based analysis)  
- `partial_exits` (aggregated within `pl` and `pl_pct` already).

**Assumptions/Constraints:**
- All timestamps are in the **same time zone**, but may have different frequencies (daily, weekly, etc.).
- `pl` and `pl_pct` represent final trade outcome aggregated across partial exits.
- `fx_pl` and FX rates are in security’s native currency; analysis focuses on **security-level performance** unless explicitly extended.

### 3.2 Price & Indicator Data

**Daily & Weekly:**
- Columns: `date, open, high, low, close, volume,` plus a large set of **TA indicators** (RSI, MACD, ATR, etc.) computed from historical prices.
- Indicators are sourced from a **reputable provider** and are computed on **close**.
- Trades are entered at **day close**, using indicator values computed on that same close.

**Key requirement:**  
- Indicator values at `entry_date` must be treated as **known at the time of trade**. No access to future bars.

### 3.3 Fundamentals

**Columns (selected, not exhaustive):**
- `date` (fundamental record date), `symbol`, `eps`, `revenue_ttm`, `pe_ratio`, `beta`, `book_value`, `capitalexpenditures`, `cashflowfromfinancing`, `cashflowfrominvestment`, `changeincashandcashequivalents`, ...  
- `dividend_per_share`, `dividend_yield`, `dividendpayout`, ...  
- `earnings_growth_yoy`, `earnings_surprise`, `estimated_eps`, ...  
- `forward_pe`, `gross_profit_ttm`, `netincome`, `operating_margin_ttm`, `return_on_assets_ttm`, `return_on_equity_ttm`, `revenue_growth_yoy`, `price_to_book`, `price_to_sales_ttm`, `peg_ratio`, etc.  
- `fiscaldateending`, `report_date`, `report_type`, `reporttime`.

**Critical:**
- Use **`report_date`** as the date when the market could reasonably know the new fundamental info.
- Strict rule: **Only fundamentals with `report_date` ≤ trade `entry_date`** are allowed for that trade.

### 3.4 Insider Data

**Columns:**
- `date`, `symbol`, `insider_title`, `transaction_type` (buy/sell), `shares`, `price`, `value`, `executive` (flag), `security_type`.

**Timing Rule:**
- Use **as-of-known-by-market**: treat insider info as available only after **filing date + configurable delay**.
- This avoids using insider knowledge before the market could have seen it.

### 3.5 Options Data

**Columns:**
- `snapshot_date`, `symbol`, `option_type`, `expiration_date`, `strike`, `days_to_expiration`, `bid`, `ask`, `last_price`, `volume`, `open_interest`, `implied_volatility`, `delta`, `gamma`, `theta`, `vega`, `ask_size`, `bid_size`, `contractid`, `date`, `mark`, `rho`.

**Data Characteristics:**
- Often **missing** for many symbols/periods.
- Provided as yearly CSVs per security.

**Usage Rule:**
- For each trade, use options snapshots **within a 60-day window around the trade entry date**, applying strict **as-of** constraints (no future options info).

---

## 4. Configuration System & Strategy Profiles

### 4.1 Configuration Sources

1. **YAML/JSON configuration files** (version-controlled, reusable).  
2. **GUI configuration dialogs** (user-friendly, ephemeral).  
3. System should support **saving/loading strategy profiles**, which are essentially named configurations.


### 4.2 Core Config Elements

A configuration object/profile should contain:

1. **Strategy Profile** (per strategy):
   - `name`: Human-readable strategy name (e.g. `"momentum_value_mix"`).
   - `good_threshold_pct`: e.g. `2.0` (% PL above which trade is "good").
   - `bad_threshold_pct`: e.g. `-1.0` (% PL below which trade is "bad").
   - `indeterminate_max_days`: max duration for trade to be considered indeterminate if PL in between thresholds.
   - `bad_min_days`: min duration for trade to be flagged as bad when PL is unfavourable.
   - Whether thresholds are **absolute** or derived from **distribution percentiles**.

2. **Data Alignment & Bias Prevention Settings:**
   - `fundamentals_reporting_delay_days`: additional delay after `report_date` before fundamentals usable.
   - `insiders_reporting_delay_days`: delay after insider filing before usable.
   - `options_lookback_days`: e.g. `60` days around entry.
   - Policy for **price gaps** (already defined as forward-fill + flag).

3. **Factor Sets & Feature Engineering Options:**
   - Which factor **categories** to include:
     - `technical` (all or subset of indicators).
     - `value` (P/E, P/B, P/S, etc.).
     - `quality` (ROE, ROA, margins, etc.).
     - `growth` (revenue/EPS growth YoY, etc.).
     - `insider` (aggregated metrics, see factor section).
     - `options` (IV, skew, etc.).
   - Per-factor options:
     - `lookback_days` (0 for point-in-time at entry, or a trailing window).
     - `null_handling` (skip vs. other policies—see later).
     - `normalization` (z-score, percentile rank, or none).

4. **Multi-Log Aggregation Options:**
   - `aggregation_mode`: one of:
     - `"pooled"` (pool all trades across symbols).
     - `"stratified"` (analyze per symbol/period and also optionally aggregated).  
   - `confounder_controls`: options like:
     - `"control_symbol_effects"` (include symbol as fixed effect / dummy variable).
     - `"control_period_effects"` (e.g., year/quarter fixed effects).

5. **Statistical Methods (User-Selectable):**
   - `enable_tier1_exploratory`: bool.
   - `enable_tier2_hypothesis_tests`: bool.
   - `enable_tier3_ml_analysis`: bool.
   - Within Tiers:
     - Tier 1: descriptive stats, correlations, distributions.
     - Tier 2: logistic regression, ANOVA/Kruskal-Wallis, chi-square.
     - Tier 3: random forest feature importance, SHAP, mutual information, optionally Bayesian logistic.

6. **Multiple Testing Control:**
   - `multiple_testing_method`: `