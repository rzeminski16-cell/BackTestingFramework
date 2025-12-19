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
7th
- **Large Cap:**  
  - Amazon (AMZN)
  - Shopify (SHOP)
  - CVS Health (CVS)
  - PayPal (PYPL)
  - Block (XYZ)
  - Tractor Supply Comapny (TSCO)
  - Dollar Tree (DLTR)
  - Global Payments (GPN)
  - Medpace Holdings (MEDP)
  - Chewy (CHWY)
- **Mid Cap:**  
  - Kanzhun Ltd (BZ)
  - VinFast Auto (VFS)
  - Federal Signal (FSS)
  - Macys INC (M)
  - Thor Industries (THO)
- **Small Cap:**  
  - Guardian Pharmacy Services (GRDN)
  - Weis Markets (WMK)
  - Leggett & Platt (LEG)
  - Winnebago Industries (WGO)
  - LGI Homes (LGIH)
---

### Materials & Manufacturing  
*(Non-energy Minerals + Process Industries + Producer Manufacturing)*

- **Large Cap:**  
  - Caterpillar (CAT)
  - 3M Company (MMM)
  - Air Products & Chemicals (APD)
  - Symbotic (SYM)
  - Hubbell (HUBB)
  - Lennox International (LII)
  - Watsco (WSO)
  - Lincoln Electric Holdings (LECO)
  - CNH Industrial (CNH)
  - Equinox Gold (EQX)
- **Mid Cap:**  
  - Chart Industries (GTLS)
  - Flowserve Corporation (FLS)
  - Commercial Metals Company (CMC)
  - ESAB Corporation (ESAB)
  - Timken Company (TKR)
- **Small Cap:**  
  - Sylvamo Corporation (SLVM)
  - SolarEdge Technologies (SEDG)
  - Power Solutions International (PSIX)
  - Lithium Americas Corp (LAC)
  - Ultra Clean Holdings (UCTT)

---

### Healthcare  
*(Health Services + Health Technology)*

- **Large Cap:**  
  - Eli Lilly and Company (LLY)
  - Intuitive Surgical (ISRG)
  - Medtronic (MDT)
  - IDEXX Laboratories (IDXX)
  - IQVIA Holdings (IQV)
  - Biogen (BIIB)
  - Insulet (PODD)
  - Tenet Healthcare (THC)
  - Universal Health Services (UHS)
  - Guardant Health (GH)
- **Mid Cap:**  
  - Qiagen (QGEN)
  - Hims & Hers Health (HIMS)
  - Bruker Corporation (BRKR)
  - Cogent Biosciences (COGT)
  - CRISPR Therapeutics (CRSP)
- **Small Cap:**  
  - AtriCure (ATRC)
  - Stroke Therapeutics (STOK)
  - Pharvis (PHVS)
  - Axogen (AXGN)
  - Immatics (IMTX)

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
  4) If still above 50 results filter for only USA or UK
  5) Randomly select 5 - 10 securities if possible
- Data source(s):
  - Trading View
