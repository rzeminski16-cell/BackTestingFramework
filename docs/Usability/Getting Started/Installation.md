---
tags:
  - usability/getting-started
  - setup
---

# Installation

How to set up the BackTesting Framework on your machine.

---

## Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Git** (to clone the repository)

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd BackTestingFramework
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling and numerical computation |
| `openpyxl`, `xlsxwriter` | Excel report generation |
| `yfinance` | Historical data download |
| `ta` | Technical analysis indicator library |
| `matplotlib`, `seaborn`, `scipy` | Visualisation and statistics |

### GUI Dependencies

| Package | Purpose |
|---|---|
| `customtkinter` | Desktop GUI framework |
| `streamlit`, `plotly` | Interactive web dashboards |

### Optimisation Dependencies

| Package | Purpose |
|---|---|
| `scikit-optimize` | Bayesian optimisation |
| `pyyaml` | YAML configuration parsing |

### Optional (Recommended)

| Package | Purpose |
|---|---|
| `numba` | JIT compilation for performance-critical loops (significant speedup) |

> [!tip] Numba
> The framework works without Numba but runs noticeably faster with it installed. Performance-critical strategy calculations are JIT-compiled when Numba is available.

---

## 3. Verify Installation

```bash
python -c "from Classes.Engine.single_security_engine import SingleSecurityEngine; print('OK')"
```

If this prints `OK`, your environment is ready.

---

## 4. Launch the GUI

```bash
python ctk_main_gui.py
```

This opens the **main launcher**, giving you access to all analysis tools from a single window.

> [!info] Alternative Entry Points
> You can also launch individual tools directly:
> - `python ctk_backtest_gui.py` — Backtesting
> - `python ctk_optimization_gui.py` — Walk-forward optimisation
> - `python ctk_univariate_optimization_gui.py` — Univariate sweeps
> - `python ctk_edge_analysis_gui.py` — Edge analysis

---

## Directory Structure

After installation, key directories are:

| Directory | Contents |
|---|---|
| `Classes/` | Core framework source code |
| `strategies/` | Trading strategy implementations |
| `config/` | Configuration files, baskets, presets |
| `raw_data/` | Historical market data (CSV) |
| `logs/` | Backtest results and reports |
| `tests/` | Test suite |

---

## Next Steps

→ [[Quick Start]] to run your first backtest.
