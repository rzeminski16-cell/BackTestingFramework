---
tags:
  - implementation/component
  - gui
---

# GUI Layer

The desktop interface built with CustomTkinter.

---

## Applications

| File | Tool | Description |
|---|---|---|
| `ctk_main_gui.py` | Main Launcher | Hub for accessing all tools |
| `ctk_backtest_gui.py` | Backtesting | Single and portfolio backtests |
| `ctk_optimization_gui.py` | Walk-Forward Optimisation | Rolling window parameter search |
| `ctk_univariate_optimization_gui.py` | Univariate Optimisation | Single parameter sweeps |
| `ctk_edge_analysis_gui.py` | Edge Analysis | E-ratio and R-multiple analysis |
| `ctk_factor_analysis_gui.py` | Factor Analysis | Tier 1/2/3 trade driver analysis |
| `ctk_vulnerability_gui.py` | Vulnerability Scoring | Portfolio scoring simulation |

---

## Architecture

Each GUI application follows the same pattern:

1. **Configuration panel** — strategy selection, parameter inputs, config options
2. **Action button** — triggers the analysis
3. **Results display** — metrics, charts, and status messages
4. **File output** — reports saved to `logs/` or `reports/`

The GUIs are thin wrappers. They collect user input, construct config objects, and delegate all computation to the [[Backtesting Engine]], [[Optimisation Engine]], or analysis classes.

---

## Shared Utilities

`Classes/GUI/` contains shared GUI utilities:
- Parameter input widgets
- Strategy selector components
- Progress indicators
- Common layout helpers

---

## Technology

- **CustomTkinter** — modern-looking Tk-based GUI framework

---

## Related

- [[Installation]] — how to launch the GUI
- [[Architecture Overview]] — where the GUI layer fits in the stack
