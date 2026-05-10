"""
Configuration containers for Monte Carlo simulation.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any
import json


class SamplingMethod(str, Enum):
    """How synthetic trade sequences are drawn from the historical pool."""
    SIMPLE_BOOTSTRAP = "simple_bootstrap"
    BLOCK_BOOTSTRAP = "block_bootstrap"


class SizingMethod(str, Enum):
    """Position sizing scheme."""
    COMPOUNDING = "compounding"
    FIXED = "fixed"


@dataclass
class SimulationConfig:
    """
    All configuration needed to run a Monte Carlo simulation.

    Returns interpretation:
        Each sampled return is treated as the per-trade return on the *risked*
        capital. For a trade where the system risks `risk_per_trade` of equity,
        equity is updated as:
            equity_next = equity + (equity * risk_per_trade) * sampled_return
        For % returns the sampled value is `pl_pct / 100`. For R-multiples it
        is the R value directly (a +1R trade returns +risk_per_trade of equity).

    Costs:
        Commission and slippage are subtracted from the sampled return on every
        trade *before* the equity update. Both are in percent units (e.g. 0.1 =
        0.1%). Defaults are 0 because the canonical BackTestingFramework trade
        log already includes commission in its pl_pct.
    """
    # General settings
    num_simulations: int = 5000
    num_trades: int = 200
    initial_capital: float = 10_000.0

    # Sampling
    sampling_method: SamplingMethod = SamplingMethod.SIMPLE_BOOTSTRAP
    block_size: int = 10  # only used for BLOCK_BOOTSTRAP

    # Position sizing
    sizing_method: SizingMethod = SizingMethod.COMPOUNDING
    risk_per_trade: float = 0.01  # fraction of equity at risk per trade

    # Drawdown-based risk reduction
    drawdown_risk_reduction: bool = False
    drawdown_threshold: float = 0.10  # fraction (0.10 == -10%)
    reduced_risk: float = 0.005

    # Costs (applied per trade, in *fractional* units, e.g. 0.001 == 0.1%)
    commission_pct: float = 0.0
    slippage_pct: float = 0.0

    # Reproducibility
    random_seed: Optional[int] = None

    # ---- helpers --------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty when valid)."""
        errors: list[str] = []
        if self.num_simulations < 1:
            errors.append("num_simulations must be >= 1")
        if self.num_trades < 1:
            errors.append("num_trades must be >= 1")
        if self.initial_capital <= 0:
            errors.append("initial_capital must be > 0")
        if not (0.0 < self.risk_per_trade <= 1.0):
            errors.append("risk_per_trade must be in (0, 1]")
        if self.sampling_method == SamplingMethod.BLOCK_BOOTSTRAP:
            if self.block_size < 1:
                errors.append("block_size must be >= 1 for block bootstrap")
            if self.block_size > self.num_trades:
                errors.append("block_size must be <= num_trades")
        if self.drawdown_risk_reduction:
            if not (0.0 < self.drawdown_threshold < 1.0):
                errors.append("drawdown_threshold must be in (0, 1)")
            if not (0.0 <= self.reduced_risk <= self.risk_per_trade):
                errors.append("reduced_risk must be in [0, risk_per_trade]")
        if self.commission_pct < 0 or self.slippage_pct < 0:
            errors.append("commission_pct and slippage_pct must be >= 0")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sampling_method"] = self.sampling_method.value
        d["sizing_method"] = self.sizing_method.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        d = dict(d)
        if "sampling_method" in d:
            d["sampling_method"] = SamplingMethod(d["sampling_method"])
        if "sizing_method" in d:
            d["sizing_method"] = SizingMethod(d["sizing_method"])
        return cls(**d)

    def to_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "SimulationConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
