---
tags:
  - usability/extending
  - strategy
  - presets
---

# Creating Strategy Presets

Save and load named parameter combinations for quick reuse.

---

## What is a Preset?

A preset is a JSON file in `config/strategy_presets/` that stores a specific set of parameter values for a strategy. This lets you quickly switch between configurations without manually re-entering values.

---

## File Location

```
config/strategy_presets/{StrategyName}__{PresetName}.json
```

Example:
```
config/strategy_presets/YourStrategy__Conservative.json
```

---

## Format

```json
{
  "strategy": "YourStrategy",
  "preset_name": "Conservative",
  "description": "Wide stops, low risk per trade",
  "parameters": {
    "atr_multiplier": 3.0,
    "risk_percent": 1.0
  }
}
```

---

## Using Presets

### In the GUI
Select the preset from the dropdown in the strategy parameters panel. The values are loaded automatically.

### In Code
```python
import json
from pathlib import Path

preset = json.loads(Path('config/strategy_presets/YourStrategy__Conservative.json').read_text())
strategy = YourStrategy(**preset['parameters'])
```

---

## Tips

> [!tip] Naming Convention
> Use descriptive names: `Aggressive`, `Conservative`, `Optimised_AAPL_2024`, `WalkForward_Best`.

---

## Next Steps

- [[Adding a New Strategy]] — create the strategy that uses these presets
- [[Walk-Forward Optimisation]] — find optimal parameters to save as a preset
