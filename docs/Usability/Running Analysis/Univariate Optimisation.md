---
tags:
  - usability/running-analysis
  - optimisation
---

# Univariate Optimisation

Sweep a single parameter across its range while holding all others constant. This reveals how sensitive the strategy is to each parameter.

---

## When to Use

- Understanding which parameters matter most
- Checking that performance degrades gracefully (no cliff edges)
- Visualising the objective landscape for a single dimension

---

## GUI Workflow

1. Launch `python ctk_univariate_optimization_gui.py` (or via the main launcher)
2. Select strategy and security
3. Choose **one parameter** to sweep
4. Set the sweep range (min, max, step) — defaults come from `config/strategy_parameters.json`
5. Choose the target metric
6. Click **Run**

---

## Interpreting Results

The output is a chart showing the target metric (y-axis) against the parameter value (x-axis).

> [!tip] What to Look For
> - **Broad plateau** around the optimum → robust (small changes don't break it)
> - **Sharp spike** at one value → fragile (likely overfit to that exact value)
> - **Monotonic trend** → the parameter has a directional effect, consider its bounds

---

## Next Steps

- [[Walk-Forward Optimisation]] — multi-parameter optimisation with out-of-sample validation
- [[Metrics Glossary]] — available target metrics
