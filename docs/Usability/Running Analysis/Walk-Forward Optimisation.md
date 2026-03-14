---
tags:
  - usability/running-analysis
  - optimisation
---

# Walk-Forward Optimisation

Find robust strategy parameters by training on past data and validating on unseen data, rolling the windows forward through time.

---

## When to Use

- Finding parameters that generalise rather than overfit
- Validating strategy robustness across market regimes
- Producing out-of-sample performance estimates

---

## How It Works

```
Data Timeline:
|--- Train 1 ---|--- Test 1 ---|
         |--- Train 2 ---|--- Test 2 ---|
                  |--- Train 3 ---|--- Test 3 ---|
```

1. Split data into rolling **train** and **test** windows
2. On each train window, search for the best parameters (using Bayesian optimisation)
3. Apply those parameters to the adjacent test window
4. Collect out-of-sample results from all test windows
5. Report aggregated performance and parameter stability

> [!warning] Overfitting Protection
> The strategy never trades on data it was trained on. This is the key advantage over a single-pass optimisation.

---

## GUI Workflow

1. Launch `python ctk_optimization_gui.py` (or via the main launcher)
2. Select strategy and security/basket
3. Configure windows:

| Field | Description | Default |
|---|---|---|
| Train Period | Length of training window (days) | 252 (1 year) |
| Test Period | Length of test window (days) | 63 (3 months) |
| Step Size | How far to advance each roll (days) | 63 |
| Window Mode | `rolling` (fixed width) or `anchored` (expanding) | rolling |

4. Select which parameters to optimise and their ranges
5. Choose the **target metric** to optimise (e.g. `sharpe_ratio`, `total_return`, `profit_factor`)
6. Click **Run Optimisation**

---

## Python API

The optimisation system uses `config/optimization_config.yaml` for defaults, but all values can be overridden programmatically via the `WalkForwardOptimizer` class in `Classes/Optimization/walk_forward_optimizer.py`.

---

## Output

- **Walk-forward report** — aggregated out-of-sample performance
- **Parameter history** — which parameters were selected in each window
- **Parameter stability analysis** — how much parameters vary across windows (stable = robust)
- **Per-window results** — individual train/test performance

Reports saved to `logs/optimization_reports/`.

> [!tip] Parameter Stability
> If the optimiser selects wildly different parameters in each window, the strategy may not be robust. Look for parameters that cluster around similar values.

---

## Bayesian Optimisation

The optimiser uses **Bayesian optimisation** (via `scikit-optimize`) rather than grid search. This is more efficient because:

- It builds a probabilistic model of the objective function
- It samples promising regions more densely
- It finds good parameters with far fewer evaluations than brute force

---

## Next Steps

- [[Univariate Optimisation]] — sweep a single parameter for sensitivity analysis
- [[Optimisation Engine]] — implementation details
- [[Optimisation Flow]] — end-to-end system flow
