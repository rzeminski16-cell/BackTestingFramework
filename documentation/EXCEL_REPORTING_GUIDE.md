# Excel Reporting Guide

## Overview

The BackTestingFramework now includes comprehensive Excel report generation capabilities. These reports provide institutional-grade analysis of your backtest results with professional formatting, detailed metrics, and interactive visualizations.

## Features

### 📊 5-Sheet Structure

Your Excel report contains five comprehensive sheets:

1. **Summary Dashboard** - At-a-glance performance overview
2. **Trade Log** - Detailed transaction history
3. **Performance Analysis** - Deep-dive metrics and breakdowns
4. **Visualizations** - Charts and graphs
5. **Market Conditions** - Regime-based analysis template

### 📈 Comprehensive Metrics

The report includes **50+ metrics** across multiple categories:

#### A. Overall Performance
- Total Return ($ and %)
- CAGR (Compound Annual Growth Rate)
- Annualized Return
- Initial & Final Capital
- Number of Trades
- Win Rate & Loss Rate

#### B. Risk Metrics
- Maximum Drawdown ($ and %)
- Average Drawdown
- Drawdown Duration
- Volatility (annualized)
- Downside Deviation
- Best/Worst Day

#### C. Risk-Adjusted Performance Ratios
- **Sharpe Ratio** (target: >1.0)
- **Sortino Ratio** (target: >2.0)
- **Calmar Ratio** (target: >2.0)
- Recovery Factor
- **Profit Factor** (target: >1.5)

#### D. Trade Quality Metrics
- Average Win/Loss ($)
- Risk/Reward Ratio (target: ≥2:1)
- Largest Win/Loss
- Win/Loss Streaks (max consecutive)
- Average & Median Trade Duration

#### E. Consistency Metrics
- Profitable Months/Weeks (%)
- Monthly Consistency (std dev)
- Consecutive Losing Months
- Strategy Exposure (% time in market)

### 🎨 Professional Formatting

- Color-coded cells (green for positive, red for negative)
- Conditional formatting for metrics vs targets
- Auto-sized columns for readability
- Frozen headers for easy scrolling
- Professional styling and borders

### 📊 Interactive Charts

- **Equity Curve** - Portfolio value over time
- **Drawdown Chart** - Visualize underwater periods
- **Return Distribution** - Histogram of trade returns
- All charts are interactive and can be customized in Excel

---

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` >= 1.3.0
- `numpy` >= 1.21.0
- `openpyxl` >= 3.0.9 (for Excel generation)

---

## Usage

### Option 1: Using the Standalone Script

The easiest way to generate Excel reports:

```bash
python generate_excel_report.py
```

This script will:
1. Run a sample backtest on AAPL
2. Generate a comprehensive Excel report
3. Optionally generate reports for multiple symbols

**Customize the script** by editing the configuration section:

```python
# In generate_excel_report.py

# Change the symbol
symbol = 'MSFT'  # or any symbol in your raw_data folder

# Adjust strategy parameters
strategy = AlphaTrendStrategy(
    volume_short_ma=4,
    volume_long_ma=30,
    stop_loss_percent=0.0,
    atr_stop_loss_multiple=2.5
)

# Configure report settings
initial_capital = 100000.0
risk_free_rate = 0.02  # 2% annual risk-free rate
benchmark_name = "S&P 500"
```

### Option 2: Using the GUI

1. Launch the GUI:
   ```bash
   python run_gui.py
   ```

2. Configure your backtest as usual

3. **Check the "Generate Excel Report" checkbox** (enabled by default)

4. Click "Run Backtest"

5. The Excel report will be saved automatically to:
   ```
   logs/{backtest_name}/reports/{backtest_name}_{symbol}_report.xlsx
   ```

### Option 3: Using the CLI Examples

1. Edit `backtest.py` and uncomment the Excel report example:

```python
# In main() function
example_excel_report_generation()  # Uncomment this line
```

2. Run the script:
   ```bash
   python backtest.py
   ```

### Option 4: Programmatic Usage

Integrate Excel reporting into your own scripts:

```python
from pathlib import Path
from Classes.Analysis.excel_report_generator import ExcelReportGenerator

# After running your backtest and obtaining a BacktestResult object:

# Create report generator
excel_generator = ExcelReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000.0,
    risk_free_rate=0.02,
    benchmark_name="S&P 500"
)

# Generate report
report_path = excel_generator.generate_report(
    result=backtest_result,
    filename="my_backtest_report.xlsx"
)

print(f"Report saved to: {report_path}")
```

---

## Report Structure Details

### Sheet 1: Summary Dashboard

**Top-level overview** organized into sections:

- **Overall Performance Metrics** - Returns, capital, trades, win rate
- **Risk Metrics** - Drawdown, volatility, best/worst days
- **Risk-Adjusted Ratios** - Sharpe, Sortino, Calmar with target comparison
- **Trade Quality** - Win/loss amounts, R:R ratio, streaks
- **Consistency Metrics** - Monthly/weekly performance, exposure

All metrics include:
- Clear labels
- Proper formatting (currency, percentages, decimals)
- Color coding for quick visual assessment

### Sheet 2: Trade Log

**Complete transaction history** with 20+ columns per trade:

| Column | Description |
|--------|-------------|
| Trade # | Sequential trade number |
| Entry Date | Date trade opened |
| Entry Price | Entry execution price |
| Entry Reason | Strategy signal that triggered entry |
| Exit Date | Date trade closed |
| Exit Price | Exit execution price |
| Exit Reason | Why trade was closed |
| Trade Type | LONG/SHORT |
| Position Size | Number of shares/contracts |
| Gross P/L | Profit before fees |
| Fees/Commission | Transaction costs |
| **Net P/L** | Final profit/loss |
| Return % | Percentage return |
| Trade Duration | Days held |
| **Cumulative P/L** | Running total |
| Stop Loss | Initial & final levels |
| Take Profit | Target price |
| Partial Exits | Number of scaling out events |

**Features:**
- Color-coded P/L (green/red)
- Frozen header row
- Auto-sized columns
- Sortable/filterable in Excel

### Sheet 3: Performance Analysis

**Deep-dive analysis** including:

#### A. Drawdown Analysis Table
- Max drawdown ($ and %)
- Average drawdown
- Drawdown duration
- Count of significant drawdowns (>5%)

#### B. Monthly/Quarterly Returns
- **Calendar heatmap** of monthly returns
  - Rows = Months (Jan-Dec)
  - Columns = Years
  - Color-coded cells (green=profit, red=loss)
- Quick visual identification of seasonal patterns
- Monthly and quarterly aggregations

#### C. Trade Return Distribution
- Histogram data showing trade count by return range:
  - < -20%
  - -20% to -10%
  - -10% to -5%
  - -5% to 0%
  - 0% to 5%
  - 5% to 10%
  - 10% to 20%
  - \> 20%
- Percentages of total trades per bin

### Sheet 4: Visualizations

**Professional charts** embedded in Excel:

1. **Equity Curve Chart**
   - Line chart showing portfolio value over time
   - X-axis: Date
   - Y-axis: Portfolio value ($)
   - Shows growth trajectory and volatility

2. **Drawdown Chart**
   - Area chart showing drawdown periods
   - Visually identifies underwater periods
   - Shaded areas show severity and duration

3. **Return Distribution Chart**
   - Bar chart of trade return histogram
   - Shows distribution shape (normal, skewed, bimodal)
   - Helps identify consistency

All charts are:
- Fully interactive in Excel
- Customizable (colors, titles, axes)
- High-resolution for presentations

### Sheet 5: Market Conditions

**Template for regime-based analysis:**

Placeholder structure for categorizing performance by market conditions:
- Bull Market
- Bear Market
- High Volatility
- Low Volatility
- Ranging Market

For each condition:
- Number of trades
- Win rate
- Average return
- Sharpe ratio
- Notes

*Note: This sheet is a template. Full implementation would require market regime classification logic.*

---

## Configuration Options

### ExcelReportGenerator Parameters

```python
ExcelReportGenerator(
    output_directory: Path,      # Where to save reports
    initial_capital: float,       # Starting capital for calculations
    risk_free_rate: float = 0.02, # Annual risk-free rate (default 2%)
    benchmark_name: str = "S&P 500" # Benchmark for comparison
)
```

### generate_report() Parameters

```python
excel_generator.generate_report(
    result: BacktestResult,       # Your backtest result
    filename: Optional[str] = None # Custom filename (auto-generated if None)
)
```

**Auto-generated filename format:**
```
{strategy_name}_{symbol}_{timestamp}.xlsx
```

---

## Interpreting the Metrics

### Understanding Risk-Adjusted Ratios

#### Sharpe Ratio
- **Formula:** (Return - Risk-Free Rate) / Volatility
- **Target:** > 1.0
- **Interpretation:**
  - < 1.0: Poor risk-adjusted returns
  - 1.0 - 2.0: Good
  - 2.0 - 3.0: Very good
  - \> 3.0: Excellent

#### Sortino Ratio
- **Formula:** (Return - Risk-Free Rate) / Downside Deviation
- **Target:** > 2.0
- **Why it matters:** Only penalizes downside volatility
- Better than Sharpe for strategies with asymmetric returns

#### Calmar Ratio
- **Formula:** CAGR / Max Drawdown %
- **Target:** > 2.0
- **Interpretation:** Shows return per unit of worst-case risk
- Higher = better recovery from drawdowns

#### Profit Factor
- **Formula:** Gross Profit / Gross Loss
- **Target:** > 1.5
- **Interpretation:**
  - < 1.0: Losing strategy
  - 1.0 - 1.5: Marginally profitable
  - 1.5 - 2.5: Good
  - \> 2.5: Excellent

### Understanding Drawdown Metrics

- **Max Drawdown:** Largest peak-to-trough decline
  - Key indicator of worst-case loss
  - Used for position sizing and risk management

- **Drawdown Duration:** Days to recover from max drawdown
  - Tests investor patience
  - Long durations = psychological challenge

- **Avg Drawdown:** Typical pullback size
  - More representative than max drawdown
  - Better for expected loss estimation

### Trade Quality Indicators

- **Risk/Reward Ratio:** Avg Win / Avg Loss
  - Target: ≥ 2:1
  - Higher ratio allows for lower win rate

- **Win Streak / Loss Streak:**
  - Indicates consistency
  - High loss streaks = difficult to trade psychologically

---

## Best Practices

### 1. Compare Multiple Backtests
- Generate reports for different parameter sets
- Compare side-by-side in Excel
- Look for robustness across parameter ranges

### 2. Focus on Risk-Adjusted Metrics
- Don't chase highest return
- Prioritize Sharpe, Sortino, Calmar ratios
- Consider maximum drawdown carefully

### 3. Analyze Consistency
- Check monthly/weekly profitability percentages
- Look for long losing streaks
- Evaluate equity curve smoothness

### 4. Review Individual Trades
- Sort Trade Log by P/L
- Identify outlier trades
- Check if exit reasons align with strategy

### 5. Check Market Condition Dependency
- Does strategy work in all regimes?
- Are profits concentrated in specific periods?
- Would it survive different market conditions?

### 6. Document Your Assumptions
- Add comments in Excel about:
  - Commission rates used
  - Slippage assumptions (if any)
  - Date range tested
  - Market context (bull/bear market)

---

## Troubleshooting

### Report Generation Fails

**Error:** `ModuleNotFoundError: No module named 'openpyxl'`

**Solution:**
```bash
pip install openpyxl>=3.0.9
```

---

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
- Close the Excel file if it's open
- Check file permissions
- Try a different output directory

---

**Error:** Report is blank or missing data

**Solution:**
- Ensure your backtest ran successfully
- Check that `result.trades` is not empty
- Verify `result.equity_curve` has data

---

### Charts Not Displaying

**Issue:** Charts show as blank or error

**Possible causes:**
- Insufficient data points (need at least 2 rows)
- Excel version incompatibility

**Solution:**
- Ensure backtest has multiple bars
- Open in Microsoft Excel (best compatibility)
- Try LibreOffice Calc or Google Sheets as alternatives

---

### Performance Issues

**Issue:** Report generation is slow for large backtests

**Solutions:**
- Large trade logs (1000+ trades) may take 10-30 seconds
- This is normal - be patient
- Consider sampling for very large datasets

---

## Advanced Usage

### Customizing the Report

The `ExcelReportGenerator` class can be extended or modified:

```python
# Inherit and customize
class CustomExcelReportGenerator(ExcelReportGenerator):
    def _create_custom_sheet(self, wb, result, metrics):
        # Add your own analysis
        ws = wb.create_sheet("Custom Analysis")
        # Your custom logic here
        pass
```

### Adding Custom Metrics

Edit `_calculate_all_metrics()` method to include additional calculations:

```python
# In excel_report_generator.py

def _calculate_all_metrics(self, result):
    metrics = {}
    # ... existing code ...

    # Add your custom metric
    metrics['my_custom_metric'] = self._calculate_my_metric(result)

    return metrics
```

### Batch Report Generation

Generate reports for all backtests:

```python
from pathlib import Path
import pandas as pd

# Assuming you have multiple backtest results
results_dict = {
    'AAPL': aapl_result,
    'MSFT': msft_result,
    'GOOGL': googl_result
}

excel_gen = ExcelReportGenerator(
    output_directory=Path('logs/batch_reports'),
    initial_capital=100000
)

for symbol, result in results_dict.items():
    report_path = excel_gen.generate_report(
        result=result,
        filename=f"batch_{symbol}_report.xlsx"
    )
    print(f"✓ Generated: {report_path}")
```

---

## Examples

### Example 1: Basic Report Generation

```python
from pathlib import Path
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Data.data_loader import DataLoader
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Analysis.excel_report_generator import ExcelReportGenerator
from strategies.alphatrend_strategy import AlphaTrendStrategy

# Configure
config = BacktestConfig(
    initial_capital=100000,
    commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.001)
)

# Load and run
data_loader = DataLoader(Path('raw_data'))
data = data_loader.load_csv('AAPL', required_columns=['date', 'open', 'high', 'low', 'close', 'volume', 'atr_14', 'ema_50'])
strategy = AlphaTrendStrategy()
engine = SingleSecurityEngine(config)
result = engine.run('AAPL', data, strategy)

# Generate report
excel_gen = ExcelReportGenerator(
    output_directory=Path('logs/reports'),
    initial_capital=100000
)
report_path = excel_gen.generate_report(result)
print(f"Report: {report_path}")
```

### Example 2: Custom Configuration

```python
# Advanced configuration
excel_gen = ExcelReportGenerator(
    output_directory=Path('logs/custom_reports'),
    initial_capital=250000,           # Higher capital
    risk_free_rate=0.03,               # 3% risk-free rate
    benchmark_name="NASDAQ-100"        # Different benchmark
)

# Custom filename
report_path = excel_gen.generate_report(
    result=result,
    filename=f"strategy_v2_{symbol}_backtest.xlsx"
)
```

### Example 3: Portfolio Reports

```python
# Generate reports for each security in a portfolio
portfolio_results = portfolio_engine.run(data_dict, strategy)

excel_gen = ExcelReportGenerator(
    output_directory=Path('logs/portfolio_reports'),
    initial_capital=100000
)

for symbol, result in portfolio_results.items():
    report_path = excel_gen.generate_report(
        result=result,
        filename=f"portfolio_{symbol}.xlsx"
    )
    print(f"{symbol}: {report_path}")
```

---

## FAQ

**Q: Can I generate reports without running a new backtest?**

A: Yes, if you have a saved `BacktestResult` object. You can pickle/unpickle it or regenerate from saved trade logs.

**Q: Can I modify the Excel file after generation?**

A: Yes! The report is a standard Excel file. You can add sheets, modify formulas, change colors, etc.

**Q: How do I share reports with non-technical stakeholders?**

A: The Excel format is perfect for this. Just send the .xlsx file. All metrics are clearly labeled and color-coded.

**Q: Can I automate report generation for daily backtests?**

A: Yes! Integrate the `ExcelReportGenerator` into your automated backtest pipeline. Reports will be saved with timestamps.

**Q: Does this work with portfolio backtests?**

A: Yes! Generate a separate report for each security in the portfolio. Each report shows that security's performance within the portfolio context.

**Q: Can I use this with my custom strategies?**

A: Yes! As long as your strategy inherits from `BaseStrategy` and produces valid `Trade` objects, the reporter will work.

**Q: Are the charts static or dynamic?**

A: Charts are embedded Excel charts. They're dynamic in Excel (you can customize them) but static in the saved file.

---

## Changelog

### Version 1.0 (Current)
- ✓ 5-sheet comprehensive report structure
- ✓ 50+ calculated metrics
- ✓ Professional formatting and styling
- ✓ Interactive charts (equity curve, drawdown, distribution)
- ✓ Color-coded cells and conditional formatting
- ✓ Monthly returns calendar heatmap
- ✓ Trade log with cumulative P/L
- ✓ Risk-adjusted performance ratios
- ✓ GUI integration
- ✓ CLI integration
- ✓ Standalone generator script

### Future Enhancements (Roadmap)
- [ ] MFE/MAE (Max Favorable/Adverse Excursion) tracking
- [ ] Benchmark comparison overlay on equity curve
- [ ] Market regime classification for Sheet 5
- [ ] Monte Carlo simulation results
- [ ] Rolling metrics over time
- [ ] Correlation analysis for portfolio reports
- [ ] HTML report generation
- [ ] PDF export option

---

## Support

For issues, questions, or feature requests related to Excel reporting:

1. Check this guide first
2. Review the `generate_excel_report.py` example
3. Examine the `ExcelReportGenerator` class in `Classes/Analysis/excel_report_generator.py`
4. Open an issue on GitHub with:
   - Error message (if any)
   - Sample code
   - Expected vs actual behavior

---

## License

This Excel reporting module is part of the BackTestingFramework project and follows the same license.

---

**Happy Backtesting! 📈**
