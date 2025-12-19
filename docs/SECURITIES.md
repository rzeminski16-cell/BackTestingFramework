# Backtesting Universe – Security Selection

This document defines the universe of securities used for backtesting trading strategies.  
Securities are categorised by asset class, sector, style, and other relevant characteristics.

---

## 1. Asset Classes

---

## Stocks

**Categorisation logic:**  
By **Sector** and **Market Capitalisation** within each sector:
- Large Cap  
- Mid Cap  
- Small Cap  

> _Fill in tickers or identifiers under each subsection._

---

### Consumer Discretionary  
*(Consumer Services + Consumer Durables + Retail Trade)*

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Materials & Manufacturing  
*(Non-energy Minerals + Process Industries + Producer Manufacturing)*

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Healthcare  
*(Health Services + Health Technology)*

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Services  
*(Commercial Services + Industrial Services)*

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Electronic Technology

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Technology Services

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Communications

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Distribution Services

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Energy

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Finance

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

### Utilities

- **Large Cap:**  
  - 
- **Mid Cap:**  
  - 
- **Small Cap:**  
  - 

---

## ETFs

**Categorisation logic:**  
By **Underlying Asset Class** and **Strategy/Style**

---

### Equity ETFs

#### Vanilla
- 

#### Value
- 

#### Growth
- 

#### Momentum
- 

#### Dividends
- 

#### Low Volatility
- 

#### Multi-Factor
- 

---

### Fixed Income ETFs

#### Vanilla
- 

#### Duration Hedged
- 

#### Target Duration
- 

---

### Commodity ETFs

#### Vanilla
- 

#### Optimised Commodity
- 

---

### Asset Allocation ETFs

#### Fixed Allocation
- 

#### Target Date
- 

#### Target Risk / Outcome
- 

---

### Alternative ETFs

#### Long–Short
- 

#### Buy-Write
- 

#### Options Collar
- 

#### Managed Futures
- 

---

## Cryptocurrencies

> _Spot, perpetuals, or proxies used in backtesting._

- 
- 
- 

---

## Currencies

> _FX pairs or currency proxies._

- 
- 
- 

---

## Notes

- Selection criteria (liquidity, data availability, survivorship bias controls):
  1) Filter for specific categories world wide
  2) If more than 100 results, filter only for USA, UK and Europe
  3) If still above 50 results filter for average 30D value above 100k
  4) Randomly select up to 5 securities
- Data source(s):
  - Trading View
