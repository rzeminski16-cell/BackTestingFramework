"""
Monte Carlo simulator.

The engine produces a matrix ``equity_curves`` of shape
``(num_simulations, num_trades + 1)`` where column 0 is initial_capital and
column ``i+1`` is equity after trade ``i`` of the simulation.

Performance notes:
- Sampling (simple + block bootstrap) is fully vectorized via numpy fancy
  indexing.
- The compounding/fixed equity update is a single numpy reduction when no
  drawdown-based risk reduction is required.
- When drawdown-based risk reduction is enabled the equity update becomes
  state-dependent and we step through trades with a tight numpy loop. Each
  step still operates on all simulations at once, so cost is O(num_trades).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import time

import numpy as np

from .config import SamplingMethod, SimulationConfig, SizingMethod


ProgressCallback = Callable[[float, str], None]


@dataclass
class SimulationResult:
    """Output of a Monte Carlo simulation run."""
    equity_curves: np.ndarray            # shape (num_simulations, num_trades + 1)
    sampled_returns: np.ndarray          # shape (num_simulations, num_trades)
    config: SimulationConfig
    pool_size: int                       # number of historical trades sampled from
    elapsed_seconds: float = 0.0

    @property
    def num_simulations(self) -> int:
        return self.equity_curves.shape[0]

    @property
    def num_trades(self) -> int:
        return self.equity_curves.shape[1] - 1

    @property
    def final_equity(self) -> np.ndarray:
        """Final equity per simulation, shape (num_simulations,)."""
        return self.equity_curves[:, -1]


# ============================================================================
# Sampling
# ============================================================================

def _simple_bootstrap(
    pool: np.ndarray, num_simulations: int, num_trades: int, rng: np.random.Generator
) -> np.ndarray:
    """Random sampling with replacement. Shape (num_simulations, num_trades)."""
    idx = rng.integers(0, pool.size, size=(num_simulations, num_trades))
    return pool[idx]


def _block_bootstrap(
    pool: np.ndarray,
    num_simulations: int,
    num_trades: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap (overlapping blocks, wrap-around).

    Selects random starting indices and extracts contiguous blocks of length
    ``block_size`` from the historical pool, concatenating them until each
    simulation has ``num_trades`` returns. Wraps around the pool to allow
    blocks longer than ``pool.size - start``.
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if pool.size == 0:
        raise ValueError("pool is empty")

    blocks_per_sim = (num_trades + block_size - 1) // block_size
    total_blocks = num_simulations * blocks_per_sim

    starts = rng.integers(0, pool.size, size=total_blocks)
    # offsets within block: shape (1, block_size) -> broadcast with starts
    offsets = np.arange(block_size)
    # shape (total_blocks, block_size) of indices into pool, mod pool size
    indices = (starts[:, None] + offsets[None, :]) % pool.size

    sampled = pool[indices]                                    # (total_blocks, block_size)
    sampled = sampled.reshape(num_simulations, blocks_per_sim * block_size)
    return sampled[:, :num_trades]


# ============================================================================
# Equity update
# ============================================================================

def _vectorized_compounding(
    sampled: np.ndarray, initial_capital: float, risk: float
) -> np.ndarray:
    """Compounding equity update with constant risk and no drawdown logic.

    equity_{t+1} = equity_t * (1 + risk * sampled_t)
    """
    growth = 1.0 + risk * sampled
    # cumprod along trade axis
    cum = np.cumprod(growth, axis=1)
    n_sim = sampled.shape[0]
    out = np.empty((n_sim, sampled.shape[1] + 1), dtype="float64")
    out[:, 0] = initial_capital
    out[:, 1:] = initial_capital * cum
    return out


def _vectorized_fixed(
    sampled: np.ndarray, initial_capital: float, risk: float
) -> np.ndarray:
    """Fixed-fractional sizing on initial capital (no compounding)."""
    pl_per_trade = initial_capital * risk * sampled
    cum = np.cumsum(pl_per_trade, axis=1)
    n_sim = sampled.shape[0]
    out = np.empty((n_sim, sampled.shape[1] + 1), dtype="float64")
    out[:, 0] = initial_capital
    out[:, 1:] = initial_capital + cum
    return out


def _stepwise_with_drawdown(
    sampled: np.ndarray,
    initial_capital: float,
    base_risk: float,
    reduced_risk: float,
    drawdown_threshold: float,
    sizing: SizingMethod,
    progress_cb: Optional[ProgressCallback],
) -> np.ndarray:
    """
    Step through trades while tracking running peak per simulation. Risk is
    swapped to ``reduced_risk`` whenever the simulation is below
    ``(1 - drawdown_threshold) * peak``.

    Vectorized across simulations (numpy ops per trade), so cost is O(T).
    """
    n_sim, n_trades = sampled.shape
    out = np.empty((n_sim, n_trades + 1), dtype="float64")
    out[:, 0] = initial_capital

    equity = np.full(n_sim, initial_capital, dtype="float64")
    peak = equity.copy()

    progress_every = max(1, n_trades // 20)

    for t in range(n_trades):
        in_dd = equity < peak * (1.0 - drawdown_threshold)
        risk_t = np.where(in_dd, reduced_risk, base_risk)

        if sizing == SizingMethod.COMPOUNDING:
            equity = equity * (1.0 + risk_t * sampled[:, t])
        else:
            equity = equity + initial_capital * risk_t * sampled[:, t]

        peak = np.maximum(peak, equity)
        out[:, t + 1] = equity

        if progress_cb is not None and (t + 1) % progress_every == 0:
            progress_cb((t + 1) / n_trades, f"Trade {t + 1} / {n_trades}")

    return out


# ============================================================================
# Public API
# ============================================================================

class MonteCarloSimulator:
    """Run a Monte Carlo simulation against a pool of historical trade returns."""

    def __init__(self, config: SimulationConfig):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid SimulationConfig: " + "; ".join(errors))
        self.config = config

    def run(
        self,
        return_pool: np.ndarray,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> SimulationResult:
        """Run the simulation and return the full equity-curve matrix.

        Args:
            return_pool: 1-D array of historical per-trade returns (fractional
                units; e.g. 0.015 for +1.5%, or +1.0 for a +1R trade when
                returns are R-multiples).
            progress_cb: Optional callback ``(progress, status)`` invoked
                periodically during the simulation. ``progress`` is in [0, 1].

        Returns:
            SimulationResult with the equity-curve matrix.
        """
        if return_pool is None or len(return_pool) == 0:
            raise ValueError("Empty return_pool")

        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        pool = np.asarray(return_pool, dtype="float64")

        t_start = time.perf_counter()
        if progress_cb is not None:
            progress_cb(0.0, "Sampling trades...")

        if cfg.sampling_method == SamplingMethod.SIMPLE_BOOTSTRAP:
            sampled = _simple_bootstrap(
                pool, cfg.num_simulations, cfg.num_trades, rng
            )
        elif cfg.sampling_method == SamplingMethod.BLOCK_BOOTSTRAP:
            sampled = _block_bootstrap(
                pool, cfg.num_simulations, cfg.num_trades, cfg.block_size, rng
            )
        else:
            raise ValueError(f"Unknown sampling method: {cfg.sampling_method}")

        # Clip extreme sampled returns. This is most useful when sampling
        # R-multiples from a pool that contains a few outliers from tight
        # stops; without clipping those values can dominate the simulation.
        if cfg.return_clip > 0:
            sampled = np.clip(sampled, -cfg.return_clip, cfg.return_clip)

        # Apply costs (commission + slippage) once, vectorized.
        if cfg.commission_pct or cfg.slippage_pct:
            sampled = sampled - cfg.commission_pct - cfg.slippage_pct

        if progress_cb is not None:
            progress_cb(0.05, "Computing equity curves...")

        if cfg.drawdown_risk_reduction:
            equity_curves = _stepwise_with_drawdown(
                sampled,
                cfg.initial_capital,
                cfg.risk_per_trade,
                cfg.reduced_risk,
                cfg.drawdown_threshold,
                cfg.sizing_method,
                progress_cb,
            )
        else:
            if cfg.sizing_method == SizingMethod.COMPOUNDING:
                equity_curves = _vectorized_compounding(
                    sampled, cfg.initial_capital, cfg.risk_per_trade
                )
            else:
                equity_curves = _vectorized_fixed(
                    sampled, cfg.initial_capital, cfg.risk_per_trade
                )

        # Equity may go negative under fixed sizing with extreme samples; clamp
        # at zero to model "wiped out" rather than allowing nonsensical values.
        np.maximum(equity_curves, 0.0, out=equity_curves)

        elapsed = time.perf_counter() - t_start
        if progress_cb is not None:
            progress_cb(1.0, f"Done in {elapsed:.2f}s")

        return SimulationResult(
            equity_curves=equity_curves,
            sampled_returns=sampled,
            config=cfg,
            pool_size=int(pool.size),
            elapsed_seconds=elapsed,
        )

    # ---- estimation -----------------------------------------------------

    @staticmethod
    def estimate_runtime(
        num_simulations: int,
        num_trades: int,
        with_drawdown_logic: bool,
    ) -> float:
        """Rough back-of-envelope runtime estimate in seconds.

        Calibration: vectorized fast-path runs roughly 1.5e8 trade-updates per
        second on a modern laptop CPU; the stepwise-with-drawdown path is ~6x
        slower because of the per-step boolean mask + maximum. We deliberately
        round generously upward to under-promise and over-deliver.
        """
        ops = num_simulations * num_trades
        if with_drawdown_logic:
            secs = ops / 2.5e7
        else:
            secs = ops / 1.5e8
        # Add a small overhead for sampling + plotting.
        return max(0.05, secs + 0.05 + num_simulations * 1e-6)
