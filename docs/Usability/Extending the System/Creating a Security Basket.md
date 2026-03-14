---
tags:
  - usability/extending
  - portfolio
---

# Creating a Security Basket

Define a group of securities for use in [[Portfolio Backtest|portfolio backtests]].

---

## What is a Basket?

A basket is a JSON file in `config/baskets/` that lists a set of ticker symbols to test together. When running a portfolio backtest, you select a basket instead of picking securities individually.

---

## Step 1: Create the JSON File

Create a new file in `config/baskets/`:

```
config/baskets/My_Tech_Basket.json
```

```json
{
  "name": "My Tech Basket",
  "description": "Large-cap US tech stocks",
  "symbols": [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META"
  ]
}
```

---

## Step 2: Verify Data Availability

Every symbol in the basket must have a corresponding CSV file in `raw_data/daily/`:

```
raw_data/daily/AAPL.csv
raw_data/daily/MSFT.csv
raw_data/daily/GOOGL.csv
...
```

Each file must contain the columns required by your strategy (see [[Adding a New Security]]).

---

## Step 3: Use in Portfolio Backtest

The basket will appear in the GUI's basket dropdown, or you can load it programmatically:

```python
import json
from pathlib import Path

basket_path = Path('config/baskets/My_Tech_Basket.json')
basket = json.loads(basket_path.read_text())
symbols = basket['symbols']
```

---

## Existing Baskets

Check `config/baskets/` for pre-built baskets like `Large_Cap_15.json` and `Balanced_Large_Cap.json`.

---

## Next Steps

- [[Portfolio Backtest]] — run a backtest with your basket
- [[Adding a New Security]] — add data for securities not yet in the system
