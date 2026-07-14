# Backtesting Framework

[![CI](https://github.com/rzeminski16-cell/BackTestingFramework/actions/workflows/ci.yml/badge.svg)](https://github.com/rzeminski16-cell/BackTestingFramework/actions/workflows/ci.yml)

A Python research platform for developing, testing, optimising, and evaluating
trading strategies on historical data — from raw data collection through
backtesting and optimisation to ML-based strategy diagnostics and an
interactive results dashboard.

## What It Does

- **Backtest trading strategies** bar-by-bar with strict look-ahead protection
- **Single-security or portfolio backtests** with shared capital, capital
  contention, and multi-currency (FX-converted) accounting
- **Optimise parameters** with walk-forward analysis and univariate sweeps
- **Analyse edge and risk** — E-ratio/R-multiple edge analysis, Monte Carlo
  simulation, rule testing, pattern analysis, vulnerability scoring
- **Collect data** from Alpha Vantage (prices, fundamentals, macro, benchmarks)
- **Diagnose strategies with ML** — a leakage-aware modelling & evaluation
  stage (purged/embargoed walk-forward, calibration, robustness checks) plus a
  Streamlit dashboard for exploring results
- **Generate Excel reports** with 50+ performance metrics and charts

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd BackTestingFramework

pip install -r requirements.txt        # full environment (GUIs + dashboard)
# or, for exact reproducible versions:
# pip install -r requirements.lock

# On Linux, the GUIs additionally need tkinter:
# sudo apt-get install python3-tk
```

Python 3.10+ (CI runs the suite on 3.10 and 3.11). Package metadata,
optional extras (`gui`, `dashboard`, `shap`, `dev`), and tool configuration
live in [pyproject.toml](pyproject.toml).

### Launch the Main Menu

```bash
python ctk_main_gui.py
```

The launcher provides access to all modules:

| Module | What it does |
|---|---|
| Backtesting | Run single-security or portfolio backtests with configurable strategies |
| Optimization | Walk-forward parameter optimisation (Bayesian search + sensitivity analysis) |
| Univariate Optimization | Sweep one parameter at a time to understand sensitivity |
| Edge Analysis | E-ratio and R-multiple analysis from trade logs |
| Data Collection | Collect prices/fundamentals/macro data from Alpha Vantage |
| Data Preparation | Build a point-in-time research run package (trades + factors) |
| Rule Tester | Test how entry/exit rule variations affect performance |
| Vulnerability Visualizer | Explore the per-trade vulnerability score and tune parameters live |
| Pattern Analysis | Analyse MA-crossover signal density before trade entries |
| Monte Carlo | Bootstrap trade returns to simulate equity curves and drawdown risk |
| Modelling & Evaluation | Train/evaluate ML models on a run package to diagnose when the strategy works |
| Results Dashboard | Interactive browser dashboard (Streamlit) for modelling results |

Each module can also be launched directly, e.g. `python ctk_backtest_gui.py`,
or `python -m streamlit run apps/modelling_dashboard.py` for the dashboard.

### Command line (headless)

The `btf` CLI scripts the core flows without a GUI — see
[Command Line Interface](docs/Usability/Reference/Command%20Line%20Interface.md):

```bash
python -m btf list                                                   # what's available
python -m btf backtest --strategy AlphaTrendV1Strategy --symbols AAPL --report
python -m btf optimize --strategy AlphaTrendV1Strategy --symbol AAPL
python -m btf montecarlo --trade-log logs/<run>/trades.csv
python -m btf dashboard
```

## Data Layout

Data lives under `raw_data/`, organised by type:

```
raw_data/
├── daily/       # Daily OHLCV + indicator CSVs, one per symbol (AAPL.csv, ...)
├── weekly/      # Weekly bars
├── forex/       # FX series for multi-currency conversion (e.g. GBPUSD_weekly.csv)
├── benchmarks/  # Benchmark index series
├── fundamentals/, insider_transactions/, options/  # Alpha Vantage extras
```

Each price CSV needs at minimum:

| Column | Required | Description |
|--------|----------|-------------|
| `date` or `time` | Yes | Timestamp |
| `close` | Yes | Closing price |
| `open`, `high`, `low` | Recommended | OHLC data |
| `volume` | Recommended | Trading volume |
| `atr_14`, `ema_50`, etc. | Strategy-dependent | Pre-calculated indicators |

Strategy-required indicators must exist in the raw CSVs (periods are baked
into column names); the Data Collection module can generate them.

## Project Structure

```
BackTestingFramework/
├── Classes/                  # Core framework code
│   ├── Config/              #   Configuration & presets
│   ├── Core/                #   Centralized performance & stable metrics
│   ├── Data/                #   Loading, validation, FX, security registry
│   ├── DataCollection/      #   Alpha Vantage collectors
│   ├── DataPrep/            #   Run-package builder (point-in-time features)
│   ├── Engine/              #   Single-security & portfolio engines
│   ├── Indicators/          #   Indicator engine
│   ├── Modelling/           #   ML modelling & evaluation stage
│   ├── MonteCarloAnalysis/  #   Monte Carlo simulator
│   ├── Optimization/        #   Walk-forward & univariate optimizers
│   ├── RuleTester/          #   Exit-rule testing
│   ├── Analysis/            #   Metrics, reports, trade logging
│   └── GUI/                 #   Shared CustomTkinter components & theme
├── strategies/              # Trading strategies (AlphaTrend family, etc.)
├── apps/                    # Data-collection GUI & Streamlit dashboard
├── ctk_*.py                 # GUI entry points (ctk_main_gui.py = launcher)
├── config/                  # Baskets, presets, benchmarks, settings
├── raw_data/                # Market data (see Data Layout)
├── docs/                    # Documentation (see below)
├── tests/                   # Test suite (829 tests)
└── tools/, scripts/         # Command-line utilities
```

> **Note:** the older parallel implementation (`backtesting/` package,
> `run_gui.py`, `examples/`, tkinter GUIs in `apps/`) has been removed — the
> maintained system is `Classes/` + the `ctk_*` GUIs. If you ever need the
> old code, it is preserved at commit `c817d98f` (tag it locally with
> `git tag legacy-frameworks c817d98f`).

## Documentation

The `docs/` folder is an Obsidian vault (open it in Obsidian for wiki-links
and graph view) plus standalone guides that read fine anywhere:

| Guide | Description |
|-------|-------------|
| [docs/Home.md](docs/Home.md) | Vault entry point — navigation to all Usability & Implementation pages |
| [Installation](docs/Usability/Getting%20Started/Installation.md) | Setup and dependencies |
| [Quick Start](docs/Usability/Getting%20Started/Quick%20Start.md) | First backtest in 5 minutes (GUI and Python API) |
| [Single Security Backtest](docs/Usability/Running%20Analysis/Single%20Security%20Backtest.md) | Step-by-step backtesting guide |
| [Portfolio Backtest](docs/Usability/Running%20Analysis/Portfolio%20Backtest.md) | Multi-security backtesting |
| [Walk-Forward Optimisation](docs/Usability/Running%20Analysis/Walk-Forward%20Optimisation.md) | Robust parameter search |
| [Metrics Glossary](docs/Usability/Reference/Metrics%20Glossary.md) | All performance metrics defined |
| [Adding a New Strategy](docs/Usability/Extending%20the%20System/Adding%20a%20New%20Strategy.md) | Strategy development guide |
| [Architecture Overview](docs/Implementation/Architecture%20Overview.md) | How the system is built |
| [Data Preparation](docs/DATA_PREPARATION.md) | Build a point-in-time research run package |
| [Modelling & Evaluation](docs/MODELLING_EVALUATION.md) | ML diagnostics: design & methodology |
| [Modelling User Guide](docs/MODELLING_USER_GUIDE.md) | Step-by-step modelling walkthrough |
| [Dashboard Guide](docs/DASHBOARD_GUIDE.md) | Every page of the results dashboard explained |
| [Modelling Data Dictionary](docs/MODELLING_DATA_DICTIONARY.md) | All features, targets, and metrics |
| [System Evaluation](docs/SYSTEM_EVALUATION.md) | Full audit findings & improvement roadmap (2026-07) |

## Key Features

### Execution Model
- Default: trades execute at **closing prices** (TradingView-style same-bar-close
  convention). Opt-in realism: `execution_timing=NEXT_BAR_OPEN` (fills at the
  next bar's open) and `intrabar_stops` (stop/TP trigger on the bar's high/low
  with gap-aware fills)
- No look-ahead bias — strategies only see data up to the current bar
  (enforced by `HistoricalDataView`, not just convention)
- Configurable commission (percentage or fixed) and slippage (direction-aware)
- Multi-currency support with strict FX validation (fails fast if rates are missing)
- LONG and SHORT strategies, with direction-aware fills, stops, and cash
  accounting (shorts are carried as posted collateral ± unrealised P/L; the
  equity curve and trade log are covered by an invariant test in both
  directions)

### Strategy Features
- Stop loss, take profit, trailing stops with indicator-based adjustments
- Partial exits and one-shot pyramiding with break-even stop
- Custom position sizing and fundamental entry filters

### Optimisation & Robustness
- Walk-forward optimisation with Bayesian search and sensitivity analysis
- **Deflated Sharpe Ratio** per window — deflates the winning in-sample Sharpe
  by the number of configurations the search examined
- Univariate parameter sweeps with stability metrics
- Monte Carlo bootstrap (simple + block, per-trade or daily-return pools) with
  drawdown-responsive sizing and annualized per-path Sharpe/Calmar distributions

### Risk & Benchmark Analytics
- Per-trade MAE/MFE tracked off bar extremes; exposure/time-in-market,
  historical VaR/CVaR, rolling Sharpe/volatility helpers
- Benchmark comparison in every report: alpha, beta, information ratio,
  tracking error, up/down capture, benchmark-relative max drawdown

### Modelling & Evaluation (ML diagnostics)
- Purged & embargoed chronological cross-validation keyed on true label windows
- Interpretable-first model ladder (baseline → linear → tree → constrained boosting)
- Economics-first leaderboard (out-of-sample Adjusted RAR% with guardrails)
- Bootstrap CIs, permutation tests, White's Reality Check, attempt ledger

### Reporting
- Excel reports with 50+ metrics, equity curves, drawdown and regression charts
- Trade-by-trade logs with FX P/L decomposition
- Interactive Streamlit dashboard for modelling runs

## Requirements

See [requirements.txt](requirements.txt) for the full list. Core: pandas,
numpy, openpyxl/xlsxwriter, scikit-learn, scikit-optimize, customtkinter,
streamlit + plotly (dashboard), pyarrow (run packages). `shap` is optional.

## License

[Your License Here]
