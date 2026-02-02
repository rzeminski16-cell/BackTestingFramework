# BackTesting Framework Documentation

Welcome to the BackTesting Framework documentation. This framework enables systematic development, testing, and refinement of trading strategies using historical market data.

---

## Quick Start

| Goal | Where to Go |
|------|-------------|
| Run your first backtest | [Quick Start Guide](overview/QUICK_START.md) |
| Understand the framework | [Introduction](overview/INTRODUCTION.md) |
| Create a custom strategy | [Strategy Development Guide](strategy-development/STRATEGY_GUIDE.md) |
| Learn how components interact | [System Architecture](overview/ARCHITECTURE.md) |

---

## Documentation Structure

```
docs/
├── USER_GUIDE.md                  ◄── Start here for workflow overview
│
├── overview/                      Framework fundamentals
│   ├── INTRODUCTION.md            What is this framework?
│   ├── ARCHITECTURE.md            System design with diagrams
│   └── QUICK_START.md             First backtest in 10 minutes
│
├── concepts/                      Core concepts explained
│   ├── BACKTESTING.md             Execution model, trade lifecycle
│   ├── STRATEGIES.md              How strategies work
│   ├── PORTFOLIO_MODE.md          Multi-security testing
│   ├── OPTIMIZATION.md            Parameter optimization
│   ├── EDGE_ANALYSIS.md           E-ratio validation
│   ├── VULNERABILITY_SCORING.md   Position scoring
│   └── FACTOR_ANALYSIS.md         Trade outcome analysis
│
├── applications/                  GUI application guides
│   ├── OVERVIEW.md                All 8 applications
│   ├── BACKTEST_GUI.md            Running backtests
│   ├── OPTIMIZATION_GUI.md        Walk-forward & univariate
│   ├── ANALYSIS_TOOLS.md          Edge, rule tester, vulnerability
│   ├── FACTOR_ANALYSIS_GUI.md     Factor analysis interface
│   └── DATA_COLLECTION.md         Gathering market data
│
├── strategy-development/          Creating custom strategies
│   ├── STRATEGY_GUIDE.md          Comprehensive guide
│   ├── STRATEGY_STRUCTURE.md      Anatomy of a strategy
│   ├── SIGNALS_AND_EXITS.md       Signal types, exit rules
│   └── TESTING_YOUR_STRATEGY.md   Validation techniques
│
├── alternative-system/            Simplified backtesting module
│   ├── OVERVIEW.md                When to use each system
│   └── USAGE.md                   How to use backtesting/
│
├── reference/                     Reference materials
│   ├── CONFIGURATION.md           All configuration files
│   ├── METRICS_GLOSSARY.md        50+ metrics defined
│   └── SECURITIES.md              Available securities
│
└── research/                      Research notes (unchanged)
```

---

## By Topic

### Getting Started

| Document | Description |
|----------|-------------|
| [Introduction](overview/INTRODUCTION.md) | Framework overview, capabilities, who it's for |
| [Quick Start](overview/QUICK_START.md) | Installation to first backtest in 10 minutes |
| [User Guide](USER_GUIDE.md) | Complete workflow walkthrough |

### Understanding the System

| Document | Description |
|----------|-------------|
| [Architecture](overview/ARCHITECTURE.md) | High-level design with component diagrams |
| [Backtesting Concepts](concepts/BACKTESTING.md) | How backtesting works, trade lifecycle |
| [Strategies](concepts/STRATEGIES.md) | Strategy framework, signals, included strategies |

### Using the Applications

| Document | Description |
|----------|-------------|
| [Applications Overview](applications/OVERVIEW.md) | All 8 GUI applications at a glance |
| [Backtest GUI](applications/BACKTEST_GUI.md) | Running single and portfolio backtests |
| [Optimization Tools](applications/OPTIMIZATION_GUI.md) | Walk-forward and univariate optimization |
| [Analysis Tools](applications/ANALYSIS_TOOLS.md) | Edge analysis, rule tester, vulnerability |
| [Factor Analysis](applications/FACTOR_ANALYSIS_GUI.md) | Statistical trade analysis |
| [Data Collection](applications/DATA_COLLECTION.md) | Fetching market data |

### Creating Strategies

| Document | Description |
|----------|-------------|
| [Strategy Guide](strategy-development/STRATEGY_GUIDE.md) | Complete guide to creating strategies |
| [Strategy Structure](strategy-development/STRATEGY_STRUCTURE.md) | Class anatomy and lifecycle |
| [Signals and Exits](strategy-development/SIGNALS_AND_EXITS.md) | Signal types and exit patterns |
| [Testing Strategies](strategy-development/TESTING_YOUR_STRATEGY.md) | Validation and common pitfalls |

### Advanced Topics

| Document | Description |
|----------|-------------|
| [Portfolio Mode](concepts/PORTFOLIO_MODE.md) | Multi-security with capital management |
| [Optimization](concepts/OPTIMIZATION.md) | Walk-forward and Bayesian optimization |
| [Edge Analysis](concepts/EDGE_ANALYSIS.md) | E-ratio methodology |
| [Vulnerability Scoring](concepts/VULNERABILITY_SCORING.md) | Position scoring for contention |
| [Factor Analysis](concepts/FACTOR_ANALYSIS.md) | Statistical factor analysis |

### Reference

| Document | Description |
|----------|-------------|
| [Configuration](reference/CONFIGURATION.md) | All configuration files and options |
| [Metrics Glossary](reference/METRICS_GLOSSARY.md) | 50+ performance metrics defined |
| [Securities](reference/SECURITIES.md) | Available securities universe |

---

## Quick Reference: Launch Commands

| Application | Command |
|-------------|---------|
| Main Launcher | `python ctk_main_gui.py` |
| Backtest GUI | `python ctk_backtest_gui.py` |
| Walk-Forward Optimization | `python ctk_optimization_gui.py` |
| Univariate Optimization | `python ctk_univariate_optimization_gui.py` |
| Edge Analysis | `python ctk_edge_analysis_gui.py` |
| Vulnerability Modeler | `python ctk_vulnerability_gui.py` |
| Factor Analysis | `python ctk_factor_analysis_gui.py` |
| Data Collection | `python apps/data_collection_gui.py` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GUI doesn't start | Install tkinter: `sudo apt-get install python3-tk` |
| Import errors | Run `pip install -r requirements.txt` |
| No trades generated | Check strategy conditions and date range |
| Missing columns | Ensure CSV has columns required by strategy |
| API rate limits | Wait for reset or upgrade Alpha Vantage tier |

---

## Need Help?

1. Start with the [User Guide](USER_GUIDE.md) for a complete workflow
2. Check the relevant documentation section above
3. Review troubleshooting sections in each guide
4. Report issues at the project's issue tracker
