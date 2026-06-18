---
tags:
  - usability
  - index
---

# Usability

This section covers everything an analyst needs to **use** the backtesting framework — from running your first backtest to building your own strategies.

---

## Getting Started

Start here if you're new to the system.

1. [[Installation]] — Dependencies, setup, and environment configuration
2. [[Quick Start]] — Run your first backtest in under 5 minutes

---

## Running Analysis

Step-by-step guides for each analysis tool in the system.

| Guide | Purpose |
|---|---|
| [[Single Security Backtest]] | Test a strategy on one security |
| [[Portfolio Backtest]] | Test a strategy across multiple securities with shared capital |
| [[Walk-Forward Optimisation]] | Find robust parameters using rolling train/test windows |
| [[Univariate Optimisation]] | Sweep a single parameter to understand sensitivity |
| [[Edge Analysis]] | Validate entry quality with E-ratio and R-multiples |
| [[Reading Reports]] | Interpret Excel reports, trade logs, and equity curves |

---

## Extending the System

Guides for analysts who need to expand the framework's capabilities.

| Guide | Purpose |
|---|---|
| [[Adding a New Strategy]] | Create a strategy from scratch using `BaseStrategy` |
| [[Adding a New Security]] | Prepare and add CSV data for a new security |
| [[Adding a New Indicator]] | Add a pre-calculated indicator to raw data files |
| [[Creating a Security Basket]] | Define a group of securities for portfolio testing |
| [[Creating Strategy Presets]] | Save and load parameter combinations |

---

## Reference

Quick-lookup tables and glossaries.

| Reference | Contents |
|---|---|
| [[Metrics Glossary]] | All 50+ performance metrics defined |
| [[Signal Types]] | BUY, SELL, PARTIAL_EXIT, ADJUST_STOP, PYRAMID, HOLD |
| [[Configuration Options]] | Every config field across BacktestConfig and PortfolioConfig |
| [[Available Securities]] | Securities included with the framework and their data coverage |
