---
tags:
  - implementation/flow
  - data
  - reporting
---

# Data to Report Pipeline

The full lifecycle from raw CSV data to a finished Excel report.

---

## Pipeline

```mermaid
flowchart TD
    subgraph Input
        CSV["raw_data/daily/AAPL.csv"]
        META["config/security_metadata.json"]
        PARAMS["config/strategy_parameters.json"]
        FX["raw_data/forex/"]
    end

    subgraph Loading
        DL["DataLoader.load_csv('AAPL')"]
        DV["DataValidator: check columns, sort, dedup"]
        SR["SecurityRegistry: lookup currency"]
        CC["CurrencyConverter: load FX rates"]
    end

    subgraph Preparation
        PD["strategy.prepare_data(data)\nAdd custom indicators"]
    end

    subgraph Execution
        ENG["SingleSecurityEngine.run()"]
        CTX["StrategyContext per bar"]
        SIG["strategy.generate_signal()"]
        TEX["TradeExecutor: open/close positions"]
        PM["PositionManager: track state"]
    end

    subgraph Results
        TRADES["List[Trade]"]
        EQUITY["Equity curve"]
    end

    subgraph Reporting
        PERF["PerformanceMetrics: 50+ metrics"]
        EXCEL["ExcelReportGenerator: .xlsx"]
        TLOG["TradeLogger: .csv"]
    end

    subgraph Output
        XLSX["logs/backtests/.../report.xlsx"]
        TCSV["logs/backtests/.../trades.csv"]
    end

    CSV --> DL --> DV --> PD
    META --> SR --> ENG
    FX --> CC --> ENG
    PARAMS --> ENG

    PD --> ENG
    ENG --> CTX --> SIG --> TEX --> PM
    PM --> TRADES
    PM --> EQUITY

    TRADES --> PERF
    EQUITY --> PERF
    PERF --> EXCEL --> XLSX
    TRADES --> TLOG --> TCSV
```

---

## Stage Breakdown

### 1. Loading
`DataLoader` reads the CSV into a pandas DataFrame. `DataValidator` ensures required columns exist, sorts by date, and removes duplicates. `SecurityRegistry` looks up the security's currency so the engine knows whether FX conversion is needed.

### 2. Preparation
`strategy.prepare_data()` runs once. It validates that all `required_columns()` exist in the data, then calls `_prepare_data_impl()` for any strategy-specific indicator calculations. The framework checks new columns for look-ahead bias.

### 3. Execution
The engine iterates bar-by-bar. On each bar:
- `StrategyContext` is built with current data, position state, and FX rate
- Stop loss / take profit are checked
- `strategy.generate_signal()` returns a signal
- `TradeExecutor` executes the signal
- `PositionManager` updates state

### 4. Results
After all bars are processed, remaining positions are closed. The engine produces a list of completed `Trade` objects and an equity curve.

### 5. Reporting
`PerformanceMetrics` calculates 50+ metrics from the trades and equity curve. `ExcelReportGenerator` creates a multi-sheet Excel workbook with summary, trade list, charts, and monthly returns. `TradeLogger` exports a flat CSV of all trades.

### 6. Output
Files are saved to `logs/backtests/single_security/` (or `portfolio/` for portfolio mode) with timestamps in the filename.

---

## Data Dependencies

| What | Where | Required By |
|---|---|---|
| Price data | `raw_data/daily/*.csv` | DataLoader |
| Security metadata | `config/security_metadata.json` | SecurityRegistry |
| FX rates | `raw_data/forex/*.csv` | CurrencyConverter |
| Strategy params | `config/strategy_parameters.json` | GUI, Optimiser |
| Presets | `config/strategy_presets/*.json` | GUI |

---

## Related

- [[Data Layer]] — loading and validation components
- [[Backtesting Engine]] — execution engine
- [[Reporting]] — metrics and report generation
- [[Reading Reports]] — how to interpret the output
