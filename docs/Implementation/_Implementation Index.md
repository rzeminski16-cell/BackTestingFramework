---
tags:
  - implementation
  - index
---

# Implementation

This section explains **how the system works** — its architecture, individual components, and end-to-end flows.

---

## Architecture

Start here for the big picture.

→ [[Architecture Overview]] — layered design, module map, design principles

---

## Components

Each subsystem explained in isolation.

| Component | What It Does |
|---|---|
| [[Data Layer]] | Loading CSVs, validating data, currency conversion |
| [[Strategy Framework]] | BaseStrategy contract, StrategyContext, signal generation |
| [[Backtesting Engine]] | SingleSecurity and Portfolio engines, bar-by-bar execution |
| [[Position Management]] | PositionManager, TradeExecutor, trade lifecycle |
| [[Optimisation Engine]] | Walk-forward, Bayesian search, univariate sweeps |
| [[Factor Analysis Component]] | Tier 1/2/3 analysis, scenario detection |
| [[Vulnerability Scorer]] | Position scoring, immunity, decay, feature weighting |
| [[Reporting]] | Excel reports, trade logs, equity curves, metrics calculation |
| [[GUI Layer]] | CustomTkinter architecture and application structure |

---

## System Flows

How components work **together** in end-to-end scenarios.

| Flow | What It Shows |
|---|---|
| [[Backtest Execution Flow]] | Bar-by-bar execution of a single-security backtest |
| [[Portfolio Execution Flow]] | Multi-security execution with capital contention |
| [[Optimisation Flow]] | Walk-forward optimisation pipeline |
| [[Data to Report Pipeline]] | Full lifecycle from raw CSV to final Excel report |
