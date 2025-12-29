# Backtesting Framework Documentation

Welcome to the Backtesting Framework documentation. This documentation is organized into two sections to help you find what you need quickly.

## Documentation Structure

### User Documentation

For users who want to run backtests, collect data, and analyze results.

| Guide | Description |
|-------|-------------|
| [Getting Started](user/GETTING_STARTED.md) | Installation, first backtest, basic concepts |
| [Tools & Applications](user/TOOLS.md) | All GUI applications and command-line tools |
| [Data Collection](user/DATA_COLLECTION.md) | Collecting market data from Alpha Vantage |
| [Running Backtests](user/BACKTESTING.md) | Single security and portfolio backtesting |
| [Portfolio Mode](user/PORTFOLIO_MODE.md) | Multi-security backtesting with shared capital |
| [Optimization](user/OPTIMIZATION.md) | Finding optimal strategy parameters |
| [Securities Universe](user/SECURITIES.md) | Available securities for testing |

### Technical Documentation

For developers who want to create strategies, understand the architecture, or extend the framework.

| Guide | Description |
|-------|-------------|
| [Architecture Overview](technical/ARCHITECTURE.md) | System design, components, data flow |
| [Strategy Development](technical/STRATEGIES.md) | Creating custom trading strategies |
| [Configuration Reference](technical/CONFIGURATION.md) | All configuration options |
| [API Reference](technical/API_REFERENCE.md) | Core classes and methods |
| [Backtest Analysis](technical/BACKTEST_ANALYSIS.md) | Deep-dive trade analysis tools |
| [Vulnerability Scoring](technical/VULNERABILITY_SCORING.md) | Portfolio capital allocation system |

## Quick Links

- **Run your first backtest**: [Getting Started Guide](user/GETTING_STARTED.md)
- **Create a custom strategy**: [Strategy Development Guide](technical/STRATEGIES.md)
- **Collect market data**: [Data Collection Guide](user/DATA_COLLECTION.md)
- **Optimize parameters**: [Optimization Guide](user/OPTIMIZATION.md)

## Need Help?

1. Check the relevant documentation section above
2. Review the [Troubleshooting](#troubleshooting) section in each guide
3. Report issues at the project's issue tracker

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| GUI doesn't start | Install tkinter: `sudo apt-get install python3-tk` (Linux) |
| No trades generated | Check strategy conditions and data column availability |
| Missing columns | Ensure your CSV has required columns for your strategy |
| Import errors | Run `pip install -r requirements.txt` |
