"""
Aggregate analysis of Monte Carlo simulation output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .simulator import SimulationResult


@dataclass
class SimulationMetrics:
    """High-level metrics summarising a Monte Carlo run."""
    num_simulations: int
    num_trades: int
    initial_capital: float

    # Final equity
    median_final_equity: float
    mean_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    min_final_equity: float
    max_final_equity: float

    # Loss probability
    probability_of_loss: float       # P(final < initial)
    probability_of_ruin: float       # P(min equity along path <= 0.5 * initial)

    # Drawdowns (positive fractions, e.g. 0.25 == -25%)
    median_max_drawdown: float
    mean_max_drawdown: float
    p95_max_drawdown: float
    worst_max_drawdown: float

    # Risk-adjusted
    median_cagr_equivalent: float    # geometric mean per-trade return, annualised assumption-free
    median_sharpe: float             # per-trade Sharpe (mean / std of per-trade equity returns)

    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimulationAnalyzer:
    """Compute aggregate metrics and percentile curves from a SimulationResult."""

    def __init__(self, result: SimulationResult):
        self.result = result
        self._max_drawdowns: Optional[np.ndarray] = None
        self._equity_returns: Optional[np.ndarray] = None

    # ---- core arrays ----------------------------------------------------

    @property
    def equity_curves(self) -> np.ndarray:
        return self.result.equity_curves

    @property
    def final_equity(self) -> np.ndarray:
        return self.result.final_equity

    def max_drawdowns(self) -> np.ndarray:
        """Per-simulation maximum drawdown (positive fractions)."""
        if self._max_drawdowns is None:
            ec = self.equity_curves
            running_peak = np.maximum.accumulate(ec, axis=1)
            # drawdown at each step (fraction below peak)
            dd = np.where(running_peak > 0, 1.0 - ec / running_peak, 0.0)
            self._max_drawdowns = dd.max(axis=1)
        return self._max_drawdowns

    def equity_returns(self) -> np.ndarray:
        """Per-trade fractional returns of the equity curve, shape (n_sim, n_trades)."""
        if self._equity_returns is None:
            ec = self.equity_curves
            prev = ec[:, :-1]
            curr = ec[:, 1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.where(prev > 0, curr / prev - 1.0, 0.0)
            self._equity_returns = ret
        return self._equity_returns

    # ---- aggregate views ------------------------------------------------

    def percentile_curves(
        self, percentiles: Tuple[float, ...] = (5.0, 50.0, 95.0)
    ) -> Dict[float, np.ndarray]:
        """Equity curves at the given percentiles, computed elementwise.

        Note: percentile curves are NOT realisable simulations - they are the
        per-step quantiles across simulations, useful for visualising the
        spread of outcomes.
        """
        ec = self.equity_curves
        return {p: np.percentile(ec, p, axis=0) for p in percentiles}

    def sample_curve_indices(self, k: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return up to ``k`` random simulation indices for plotting."""
        n_sim = self.equity_curves.shape[0]
        k = min(k, n_sim)
        if rng is None:
            rng = np.random.default_rng(0)
        return rng.choice(n_sim, size=k, replace=False)

    # ---- summary metrics ------------------------------------------------

    def metrics(self) -> SimulationMetrics:
        cfg = self.result.config
        ec = self.equity_curves
        final = self.final_equity
        dd = self.max_drawdowns()
        eq_ret = self.equity_returns()

        # Per-trade Sharpe (no risk-free rate; per-trade granularity).
        with np.errstate(invalid="ignore"):
            mean_per_sim = eq_ret.mean(axis=1)
            std_per_sim = eq_ret.std(axis=1, ddof=1) if eq_ret.shape[1] > 1 else np.zeros_like(mean_per_sim)
            sharpe = np.where(std_per_sim > 0, mean_per_sim / std_per_sim, 0.0)

        # Geometric mean per-trade return (CAGR-equivalent expressed per trade).
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(final > 0, final / cfg.initial_capital, np.nan)
            geom_per_trade = np.where(
                np.isfinite(growth) & (growth > 0),
                growth ** (1.0 / max(1, cfg.num_trades)) - 1.0,
                -1.0,
            )

        ruin_threshold = 0.5 * cfg.initial_capital
        min_along_path = ec.min(axis=1)

        return SimulationMetrics(
            num_simulations=self.result.num_simulations,
            num_trades=self.result.num_trades,
            initial_capital=cfg.initial_capital,
            median_final_equity=float(np.median(final)),
            mean_final_equity=float(np.mean(final)),
            p5_final_equity=float(np.percentile(final, 5)),
            p95_final_equity=float(np.percentile(final, 95)),
            min_final_equity=float(np.min(final)),
            max_final_equity=float(np.max(final)),
            probability_of_loss=float(np.mean(final < cfg.initial_capital)),
            probability_of_ruin=float(np.mean(min_along_path <= ruin_threshold)),
            median_max_drawdown=float(np.median(dd)),
            mean_max_drawdown=float(np.mean(dd)),
            p95_max_drawdown=float(np.percentile(dd, 95)),
            worst_max_drawdown=float(np.max(dd)),
            median_cagr_equivalent=float(np.median(geom_per_trade)),
            median_sharpe=float(np.median(sharpe)),
            elapsed_seconds=self.result.elapsed_seconds,
        )

    # ---- export ---------------------------------------------------------

    def to_csv(self, path) -> None:
        """Export the equity-curve matrix to CSV.

        One row per simulation, one column per trade index. The first column
        holds the initial capital (trade index 0).
        """
        import pandas as pd
        n_sim, n_cols = self.equity_curves.shape
        cols = [f"t{i}" for i in range(n_cols)]
        df = pd.DataFrame(self.equity_curves, columns=cols)
        df.insert(0, "simulation_id", np.arange(n_sim))
        df.to_csv(path, index=False)
