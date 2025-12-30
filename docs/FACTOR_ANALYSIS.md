# Factor Analysis Module

Comprehensive documentation for the Strategy Performance Factor Analysis module.

---

## Overview

The Factor Analysis module analyzes how market, technical, fundamental, insider, and options factors relate to trade-level performance. It helps answer the critical question: **"What market conditions lead to good vs. bad trades?"**

### Key Capabilities

- **Trade Classification**: Categorize trades as good/bad/indeterminate based on P&L
- **Temporal Alignment**: Prevent forward-looking bias with as-of date logic
- **Factor Engineering**: Extract 50+ factors from multiple data sources
- **Statistical Analysis**: Three tiers from exploratory to machine learning
- **Scenario Detection**: Identify best/worst trading conditions
- **Report Generation**: Professional Excel and JSON outputs

---

## Quick Start

### Basic Usage

```python
from Classes.FactorAnalysis import FactorAnalyzer

# Initialize with default settings
analyzer = FactorAnalyzer(verbose=True)

# Analyze trades with price data
result = analyzer.analyze(
    trade_data=trades_df,          # DataFrame or CSV path
    price_data=prices_df,          # Optional: for technical factors
    fundamental_data=fund_df,      # Optional: for fundamental factors
    insider_data=insider_df,       # Optional: for insider factors
    options_data=options_df        # Optional: for options factors
)

# Check results
if result.success:
    print(f"Analyzed {result.data_summary['total_trades']} trades")
    print(f"Good: {result.data_summary['good_trades']}")
    print(f"Bad: {result.data_summary['bad_trades']}")

    # Generate Excel report
    analyzer.generate_excel_report(result, "analysis_report.xlsx")

    # Print summary
    analyzer.print_summary(result)
else:
    print(f"Analysis failed: {result.error}")
```

### With Custom Configuration

```python
from Classes.FactorAnalysis import (
    FactorAnalyzer,
    FactorAnalysisConfig,
    TradeClassificationConfig,
    DataAlignmentConfig
)

# Custom configuration
config = FactorAnalysisConfig(
    trade_classification=TradeClassificationConfig(
        good_threshold_pct=3.0,      # 3% = good trade
        bad_threshold_pct=-2.0,      # -2% = bad trade
        indeterminate_max_days=10    # Ignore short trades
    ),
    data_alignment=DataAlignmentConfig(
        fundamentals_reporting_delay_days=1,   # 1 day delay for fundamentals
        insiders_reporting_delay_days=2        # 2 day delay for insider filings
    )
)

analyzer = FactorAnalyzer(config=config)
result = analyzer.analyze(trades_df, price_data=prices_df)
```

---

## Input Data Requirements

### Trade Log Format

The trade log must contain these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `symbol` | string | Yes | Ticker symbol |
| `entry_date` | date | Yes | Trade entry date |
| `exit_date` | date | Yes | Trade exit date |
| `pl` | float | Yes | Profit/loss in dollars |
| `pl_pct` | float | Yes | Profit/loss percentage |
| `entry_price` | float | No | Entry price |
| `exit_price` | float | No | Exit price |

### Price Data Format

Daily OHLCV data with optional indicators:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `symbol` | string | Yes | Ticker symbol |
| `date` | date | Yes | Trading date |
| `open` | float | Yes | Open price |
| `high` | float | Yes | High price |
| `low` | float | Yes | Low price |
| `close` | float | Yes | Close price |
| `volume` | int | Yes | Trading volume |
| `rsi_14` | float | No | 14-day RSI |
| `macd` | float | No | MACD value |
| `atr_14` | float | No | 14-day ATR |
| ... | ... | No | Additional indicators |

### Fundamental Data Format

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Ticker symbol |
| `report_date` | date | When data was publicly available |
| `pe_ratio` | float | Price-to-earnings ratio |
| `price_to_book` | float | Price-to-book ratio |
| `return_on_equity_ttm` | float | ROE trailing 12 months |
| `revenue_growth_yoy` | float | Year-over-year revenue growth |
| ... | ... | Additional fundamental metrics |

### Insider Data Format

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Ticker symbol |
| `filing_date` | date | SEC filing date |
| `transaction_date` | date | Actual transaction date |
| `transaction_type` | string | "buy" or "sell" |
| `shares` | int | Number of shares |
| `value` | float | Transaction value |

### Options Data Format

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Ticker symbol |
| `date` | date | Quote date |
| `implied_volatility` | float | ATM implied volatility |
| `put_call_ratio` | float | Put/call volume ratio |
| `open_interest_calls` | int | Call open interest |
| `open_interest_puts` | int | Put open interest |

---

## Configuration Reference

### FactorAnalysisConfig

Main configuration container:

```python
@dataclass(frozen=True)
class FactorAnalysisConfig:
    trade_classification: TradeClassificationConfig
    data_alignment: DataAlignmentConfig
    factor_engineering: FactorEngineeringConfig
    statistical_analysis: StatisticalAnalysisConfig
    scenario_analysis: ScenarioAnalysisConfig
    output: OutputConfig
```

### TradeClassificationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `good_threshold_pct` | float | 2.0 | Min % return for "good" trade |
| `bad_threshold_pct` | float | -1.0 | Max % return for "bad" trade |
| `indeterminate_max_days` | int | 15 | Max days for indeterminate |
| `bad_min_days` | int | 20 | Min days before "bad" classification |
| `threshold_type` | enum | ABSOLUTE | ABSOLUTE or RELATIVE |

### DataAlignmentConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fundamentals_reporting_delay_days` | int | 0 | Delay after report_date |
| `insiders_reporting_delay_days` | int | 3 | Delay after filing_date |
| `options_lookback_days` | int | 60 | Days of options history |
| `price_forward_fill_allowed` | bool | True | Allow forward-fill gaps |

### FactorEngineeringConfig

Controls which factor categories are enabled:

```python
factor_engineering = FactorEngineeringConfig(
    technical=FactorCategoryConfig(enabled=True),
    value=FactorCategoryConfig(enabled=True),
    quality=FactorCategoryConfig(enabled=True),
    growth=FactorCategoryConfig(enabled=True),
    insider=FactorCategoryConfig(enabled=True),
    options=FactorCategoryConfig(enabled=True),
    outlier_handling=OutlierHandlingConfig(
        method="winsorize",
        threshold=3.0
    ),
    null_handling=NullHandlingConfig(
        strategy="drop",
        max_null_pct=0.5
    )
)
```

### StatisticalAnalysisConfig

```python
statistical_analysis = StatisticalAnalysisConfig(
    tier1_exploratory=Tier1Config(
        enabled=True,
        alpha=0.05,
        min_samples=30
    ),
    tier2_hypothesis_tests=Tier2Config(
        enabled=True,
        logistic_regression=True,
        anova=True,
        chi_square=True
    ),
    tier3_ml_analysis=Tier3Config(
        enabled=True,
        random_forest=True,
        shap=True,
        mutual_information=True,
        n_estimators=100,
        min_samples=50
    ),
    multiple_testing=MultipleTestingConfig(
        correction_method="fdr",  # or "bonferroni"
        alpha=0.05
    )
)
```

### ScenarioAnalysisConfig

```python
scenario_analysis = ScenarioAnalysisConfig(
    scenario_mode="binary",  # or "clustering" or "auto"
    min_trades_per_scenario=20,
    n_clusters=5,            # For clustering mode
    min_lift=1.2             # Minimum lift for scenario inclusion
)
```

---

## Factor Reference

### Technical Factors

Computed from price data at trade entry:

| Factor | Description | Range |
|--------|-------------|-------|
| `tech_rsi_14` | 14-day RSI | 0-100 |
| `tech_macd_signal_diff` | MACD - Signal line | ± |
| `tech_bb_pct` | Bollinger Band %B | 0-1+ |
| `tech_momentum_20` | 20-day price momentum | ± |
| `tech_volume_ratio` | Volume vs 20-day avg | 0+ |
| `tech_atr_pct` | ATR as % of price | 0+ |
| `tech_close_vs_sma_50` | Close/SMA50 ratio | 0+ |
| `tech_close_vs_sma_200` | Close/SMA200 ratio | 0+ |

### Fundamental Factors

#### Value Factors
| Factor | Description |
|--------|-------------|
| `fund_pe_ratio` | Price-to-earnings |
| `fund_price_to_book` | Price-to-book |
| `fund_price_to_sales` | Price-to-sales (TTM) |
| `fund_peg_ratio` | Price/earnings-to-growth |
| `fund_dividend_yield` | Dividend yield % |
| `fund_value_composite` | Composite value score |

#### Quality Factors
| Factor | Description |
|--------|-------------|
| `fund_roe` | Return on equity (TTM) |
| `fund_roa` | Return on assets (TTM) |
| `fund_current_ratio` | Current ratio |
| `fund_debt_to_equity` | Debt-to-equity |
| `fund_quality_composite` | Composite quality score |

#### Growth Factors
| Factor | Description |
|--------|-------------|
| `fund_revenue_growth_yoy` | Revenue growth YoY |
| `fund_earnings_growth_yoy` | Earnings growth YoY |
| `fund_eps_surprise` | Latest EPS surprise % |
| `fund_growth_composite` | Composite growth score |

### Insider Factors

Computed from insider transaction data:

| Factor | Description |
|--------|-------------|
| `insider_buy_count_30d` | Buy transactions (30 days) |
| `insider_sell_count_30d` | Sell transactions (30 days) |
| `insider_net_shares_30d` | Net shares bought/sold |
| `insider_buy_value_30d` | Buy dollar value |
| `insider_sell_value_30d` | Sell dollar value |
| `insider_sentiment_score` | Composite sentiment (-1 to +1) |

### Options Factors

Computed from options market data:

| Factor | Description |
|--------|-------------|
| `options_iv` | Implied volatility (ATM) |
| `options_iv_percentile` | IV percentile (1yr) |
| `options_put_call_ratio` | Put/call volume ratio |
| `options_oi_put_call_ratio` | Open interest P/C ratio |
| `options_iv_skew` | Put/call IV skew |
| `options_sentiment` | Composite sentiment |

### Regime Factors

Market regime classification:

| Factor | Values | Description |
|--------|--------|-------------|
| `regime_volatility` | low/medium/high | Volatility regime |
| `regime_trend` | bullish/neutral/bearish | Trend regime |
| `regime_momentum` | positive/neutral/negative | Momentum regime |

---

## Analysis Output

### AnalysisOutput Structure

```python
@dataclass
class AnalysisOutput:
    success: bool              # Did analysis complete?
    timestamp: str             # ISO timestamp
    config_used: Dict          # Configuration snapshot
    data_summary: Dict         # Trade counts, date range
    quality_score: Dict        # Data quality metrics
    enriched_trades: DataFrame # Trades with all factors
    tier1: Dict                # Exploratory results
    tier2: Dict                # Hypothesis test results
    tier3: Any                 # ML results
    multiple_testing: Any      # Correction results
    scenarios: Any             # Scenario detection results
    interactions: Any          # Factor interaction results
    key_findings: List[str]    # Auto-generated insights
    warnings: List[str]        # Issues encountered
    error: Optional[str]       # Error message if failed
```

### Tier 1 Results

```python
tier1 = result.tier1

# Point-biserial correlations
for corr in tier1['point_biserial']:
    print(f"{corr.factor}: r={corr.correlation:.3f}, p={corr.p_value:.4f}")

# Descriptive statistics
for factor, stats in tier1['descriptive_stats'].items():
    print(f"{factor}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

### Tier 2 Results

```python
tier2 = result.tier2

# Logistic regression
reg = tier2['logistic_regression']
print(f"Pseudo R²: {reg.pseudo_r2:.4f}")

for factor in reg.factor_results:
    if factor.p_value < 0.05:
        print(f"{factor.factor_name}: OR={factor.odds_ratio:.2f}, p={factor.p_value:.4f}")

# ANOVA results
for anova in tier2['anova']:
    print(f"{anova.factor}: F={anova.statistic:.2f}, p={anova.p_value:.4f}")
```

### Tier 3 Results

```python
tier3 = result.tier3

# Random Forest importance
print(f"CV Accuracy: {tier3.rf_cv_accuracy:.1%}")

for feat in tier3.rf_feature_importances[:10]:
    print(f"{feat.feature_name}: {feat.importance:.4f}")

# SHAP values
for shap in tier3.shap_results[:10]:
    print(f"{shap.feature_name}: {shap.mean_abs_shap:.4f} ({shap.direction})")

# Consensus features
consensus = tier3.get_consensus_top_features(n=5)
print(f"Top consensus features: {consensus}")
```

### Scenario Results

```python
scenarios = result.scenarios

# Best scenarios
for scenario in scenarios.best_scenarios[:5]:
    print(f"\n{scenario.name}")
    print(f"  Conditions: {scenario.get_condition_string()}")
    print(f"  N trades: {scenario.n_trades}")
    print(f"  Good rate: {scenario.good_trade_rate:.1%}")
    print(f"  Lift: {scenario.lift:.2f}x")

# Worst scenarios
for scenario in scenarios.worst_scenarios[:5]:
    print(f"\n{scenario.name}")
    print(f"  Lift: {scenario.lift:.2f}x (bad)")
```

---

## Multi-Log Analysis

Analyze multiple trade logs together with stratification:

```python
from Classes.FactorAnalysis import FactorAnalyzer

# Multiple trade logs
trade_logs = [
    "logs/backtests/strategy_A_trades.csv",
    "logs/backtests/strategy_B_trades.csv",
    "logs/backtests/strategy_C_trades.csv"
]

# Metadata for stratification
log_metadata = [
    {"name": "Strategy A", "strategy": "momentum"},
    {"name": "Strategy B", "strategy": "mean_reversion"},
    {"name": "Strategy C", "strategy": "momentum"}
]

analyzer = FactorAnalyzer()
result = analyzer.analyze(
    trade_data=trade_logs,
    log_metadata=log_metadata,
    price_data=prices_df
)

# Results include stratified analysis
# - Pooled analysis across all logs
# - Per-log breakdown
# - Strategy-level grouping
```

---

## Profile Management

Save and load analysis configurations:

```python
from Classes.FactorAnalysis import ProfileManager, FactorAnalysisConfig

# Create profile manager
pm = ProfileManager(config_dir="./configs/factor_analysis")

# Save current configuration
config = FactorAnalysisConfig(...)
pm.save_profile("momentum_analysis_v1", config)

# Load saved profile
loaded_config = pm.load_profile("momentum_analysis_v1")

# List available profiles
profiles = pm.list_profiles()
print(f"Available profiles: {profiles}")
```

---

## Audit Logging

Track all analysis decisions for reproducibility:

```python
from Classes.FactorAnalysis import FactorAnalyzer

analyzer = FactorAnalyzer(verbose=True)
result = analyzer.analyze(trades_df)

# Get audit log
audit_log = analyzer.get_audit_log()
for entry in audit_log[-10:]:  # Last 10 entries
    print(f"[{entry['timestamp']}] {entry['level']}: {entry['message']}")

# Save audit log to file
analyzer.save_audit_log("analysis_audit.log")
```

---

## Best Practices

### 1. Data Quality

- Ensure sufficient trade sample size (minimum 50-100 trades)
- Check for missing data in factor sources
- Validate date alignment between trade logs and factor data

### 2. Bias Prevention

- Always use temporal alignment (enabled by default)
- Set appropriate reporting delays for fundamentals/insiders
- Never use exit-date information for entry-date analysis

### 3. Statistical Rigor

- Apply multiple testing correction (FDR recommended)
- Check sample sizes per scenario (minimum 20 trades)
- Validate findings with out-of-sample data when possible

### 4. Interpretation

- Focus on factors with both statistical significance AND practical significance
- Consider the economic rationale for any discovered relationship
- Be skeptical of overly complex scenarios

### 5. Integration

- Use findings to add entry filters, not replace strategy logic
- Start with high-confidence, high-lift scenarios
- Monitor strategy performance after implementing changes

---

## Troubleshooting

### Common Issues

**"Insufficient trades for analysis"**
- Ensure trade log has at least 30 trades after classification
- Check that good/bad thresholds aren't too extreme

**"No factor data available"**
- Verify price/fundamental data covers trade date range
- Check symbol names match between trade log and factor data

**"All trades classified as indeterminate"**
- Adjust `good_threshold_pct` and `bad_threshold_pct`
- Check `pl_pct` column values are decimals (0.05 = 5%), not percentages

**"Forward-looking bias detected"**
- Ensure `report_date` (not fiscal period end) is used for fundamentals
- Verify `filing_date` (not transaction date) is used for insider data

### Performance Tips

- For large datasets, disable Tier 3 ML during initial exploration
- Use sampling for datasets > 10,000 trades
- Disable SHAP calculation if scikit-learn is slow

---

## API Reference

### Main Classes

| Class | Description |
|-------|-------------|
| `FactorAnalyzer` | Main orchestrator |
| `FactorAnalysisConfig` | Root configuration |
| `AnalysisOutput` | Result container |

### Data Classes

| Class | Description |
|-------|-------------|
| `TradeLogLoader` | Load trade CSVs |
| `PriceDataLoader` | Load price data |
| `FundamentalLoader` | Load fundamentals |
| `InsiderLoader` | Load insider data |
| `OptionsLoader` | Load options data |

### Preprocessing

| Class | Description |
|-------|-------------|
| `TradeClassifier` | Classify trades |
| `TemporalAligner` | Align data temporally |
| `DataEnricher` | Merge factor data |
| `QualityScorer` | Score data quality |
| `MultiLogAggregator` | Aggregate multiple logs |

### Factors

| Class | Description |
|-------|-------------|
| `TechnicalFactors` | Technical indicators |
| `FundamentalFactors` | Fundamental metrics |
| `InsiderFactors` | Insider activity |
| `OptionsFactors` | Options metrics |
| `RegimeFactors` | Market regimes |
| `FactorNormalizer` | Z-score/percentile |
| `OutlierHandler` | Outlier detection |

### Analysis

| Class | Description |
|-------|-------------|
| `Tier1Exploratory` | Descriptive stats |
| `Tier2Hypothesis` | Regression, ANOVA |
| `Tier3ML` | Random Forest, SHAP |
| `MultipleTestingCorrector` | FDR/Bonferroni |

### Scenarios

| Class | Description |
|-------|-------------|
| `ScenarioDetector` | Find best/worst |
| `ScenarioValidator` | Validate scenarios |
| `InteractionAnalyzer` | Factor interactions |

### Output

| Class | Description |
|-------|-------------|
| `ExcelReportGenerator` | Excel workbooks |
| `JsonPayloadGenerator` | JSON for GUI |
| `ResultFormatter` | Text formatting |

---

## Graphical User Interface

The Factor Analysis module includes a comprehensive GUI built with CustomTkinter.

### Launching the GUI

```bash
# Main Dashboard
python ctk_factor_analysis_gui.py

# Configuration Manager only
python ctk_factor_analysis_gui.py --config

# Data Upload interface only
python ctk_factor_analysis_gui.py --upload
```

### Dashboard Features

The main Factor Analysis Dashboard provides:

1. **Data Summary View** - Overview of loaded trades and data quality
2. **Tier 1 Analysis View** - Correlation analysis and factor screening
3. **Tier 2 Analysis View** - Hypothesis testing results
4. **Tier 3 Analysis View** - ML feature importance and SHAP values
5. **Scenario Analysis View** - Detected market scenarios
6. **Export & Reports View** - Export to Excel, JSON, CSV, HTML
7. **Audit Trail View** - Analysis history and logging

### Configuration Manager

The Configuration Manager allows you to:

- Create and manage configuration profiles
- Set trade classification thresholds
- Configure data alignment settings
- Enable/disable factor categories
- Adjust statistical analysis parameters
- Configure scenario detection

### Data Upload Interface

The Data Upload interface provides:

- File upload for trade logs and supplementary data
- Automatic column detection
- Column mapping interface
- Data quality validation
- Data preview functionality

### GUI Requirements

```bash
# Install CustomTkinter
pip install customtkinter

# Required for full functionality
pip install pandas numpy scipy scikit-learn
```

### Programmatic GUI Access

```python
# From Python
from Classes.GUI.factor_analysis import FactorAnalysisDashboard

app = FactorAnalysisDashboard()
app.run()

# Configuration Manager
from Classes.GUI.factor_analysis import FactorConfigManagerGUI

config_app = FactorConfigManagerGUI()
config_app.run()

# Data Upload
from Classes.GUI.factor_analysis import FactorDataUploadGUI

upload_app = FactorDataUploadGUI()
upload_app.run()
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2024-01 | Added GUI interfaces |
| 1.0.0 | 2024-01 | Initial release |

---

## Related Documentation

- [USER_GUIDE.md](USER_GUIDE.md) - Overall framework workflow
- [CONFIGURATION.md](CONFIGURATION.md) - General configuration reference
- [comprehensive-plan.md](comprehensive-plan.md) - Original specification
