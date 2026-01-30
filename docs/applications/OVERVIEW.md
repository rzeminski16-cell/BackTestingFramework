# Applications Overview

The BackTesting Framework provides eight GUI applications, each designed for a specific workflow. All applications share a consistent interface built with CustomTkinter.

---

## Application Launcher

Start here to access all applications:

```bash
python ctk_main_gui.py
```

The launcher displays clickable cards for each application:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKTESTING FRAMEWORK                                │
│                           Main Launcher                                     │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │                 │  │                 │  │                 │           │
│   │   Backtesting   │  │   Walk-Forward  │  │   Univariate    │           │
│   │                 │  │   Optimization  │  │   Optimization  │           │
│   │                 │  │                 │  │                 │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │                 │  │                 │  │                 │           │
│   │  Edge Analysis  │  │  Vulnerability  │  │ Factor Analysis │           │
│   │                 │  │    Modeler      │  │                 │           │
│   │                 │  │                 │  │                 │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐                                 │
│   │                 │  │                 │                                 │
│   │ Data Collection │  │  Rule Tester    │                                 │
│   │                 │  │                 │                                 │
│   │                 │  │                 │                                 │
│   └─────────────────┘  └─────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Application Summary

| Application | Purpose | Launch Command |
|------------|---------|----------------|
| **Backtesting** | Run single or portfolio backtests | `python ctk_backtest_gui.py` |
| **Walk-Forward Optimization** | Multi-period parameter optimization | `python ctk_optimization_gui.py` |
| **Univariate Optimization** | Single-parameter sweeps | `python ctk_univariate_optimization_gui.py` |
| **Edge Analysis** | E-ratio and R-multiple analysis | `python ctk_edge_analysis_gui.py` |
| **Vulnerability Modeler** | Position scoring optimization | `python ctk_vulnerability_gui.py` |
| **Factor Analysis** | Trade outcome factor analysis | `python ctk_factor_analysis_gui.py` |
| **Data Collection** | Fetch market data | `python apps/data_collection_gui.py` |
| **Rule Tester** | Test exit rule variations | `python ctk_rule_tester_gui.py` |

---

## Workflow Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TYPICAL WORKFLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────┐
                        │   Data Collection  │
                        │                    │
                        │  Gather historical │
                        │  market data       │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │    Backtesting     │
                        │                    │
                        │  Test strategy on  │
                        │  historical data   │
                        └─────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │ Walk-Forward│ │ Univariate  │ │    Rule     │
          │Optimization │ │Optimization │ │   Tester    │
          │             │ │             │ │             │
          │Find robust  │ │Understand   │ │Test exit    │
          │parameters   │ │param impact │ │variations   │
          └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                        │    Re-Backtest     │
                        │                    │
                        │  Test with new     │
                        │  parameters        │
                        └─────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │    Edge     │ │   Factor    │ │Vulnerability│
          │  Analysis   │ │  Analysis   │ │  Modeler    │
          │             │ │             │ │             │
          │Validate     │ │Find winning │ │Optimize     │
          │entry edge   │ │conditions   │ │allocation   │
          └─────────────┘ └─────────────┘ └─────────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                        │  Strategy Refined  │
                        │                    │
                        │  Apply insights,   │
                        │  iterate           │
                        └────────────────────┘
```

---

## Common Interface Elements

All applications share these UI patterns:

### Navigation

- **Wizard steps**: Multi-step workflows with Back/Next buttons
- **Tab views**: Organized sections for different functions
- **Sidebar panels**: Configuration options alongside main content

### Data Selection

- **File browsers**: Select data files and directories
- **Dropdown menus**: Choose securities, strategies, baskets
- **Multi-select lists**: Pick multiple items (e.g., securities for portfolio)

### Results Display

- **Tables**: Sortable data grids for metrics and trades
- **Charts**: Embedded matplotlib/plotly visualizations
- **Progress bars**: Track long-running operations

### Export Options

- **Excel reports**: Comprehensive workbooks with charts
- **CSV files**: Raw data for external analysis
- **JSON**: Machine-readable results

---

## Quick Reference

### Data Collection → Backtesting

```
1. Data Collection GUI: Fetch AAPL, MSFT, GOOGL daily data
2. Backtest GUI: Load securities, run portfolio backtest
```

### Backtesting → Analysis

```
1. Backtest GUI: Generate trade logs
2. Edge Analysis: Load trade logs, validate entry edge
3. Factor Analysis: Analyze what conditions affect outcomes
```

### Optimization → Backtesting

```
1. Univariate Optimization: Find optimal parameter values
2. Walk-Forward Optimization: Validate robustness
3. Backtest GUI: Re-test with optimized parameters
```

---

## Detailed Guides

Each application has a dedicated guide:

- [Backtest GUI](BACKTEST_GUI.md) — Running backtests
- [Optimization Tools](OPTIMIZATION_GUI.md) — Walk-forward and univariate optimization
- [Analysis Tools](ANALYSIS_TOOLS.md) — Edge analysis, rule tester, vulnerability
- [Factor Analysis](FACTOR_ANALYSIS_GUI.md) — Factor analysis interface
- [Data Collection](DATA_COLLECTION.md) — Gathering market data
