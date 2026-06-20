"""
Adjusted RAR% — the primary, house-configurable selection metric.

The framework already defines this metric in
``Classes/Core/stable_metrics.py`` as RAR% (annualised return from a log-equity
regression) multiplied by R² to penalise noisy equity curves. We reuse that exact
definition as the default and expose its knobs (bars/year, R² weighting, clipping)
through :class:`~Classes.Modelling.config.AdjustedRARConfig`, because the docs
require the formula to remain the user's choice rather than a hidden default.

The economic backtest builds a **calendar-daily, forward-filled equity curve**
from a set of (possibly filtered) trades and scores it here, so a candidate
overlay can be compared to the baseline strategy on a like-for-like basis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import AdjustedRARConfig


def build_daily_equity_curve(
    trades: List[Dict[str, Any]],
    initial_capital: float = 100_000.0,
    weight: str = "pl",
) -> pd.DataFrame:
    """Construct a calendar-daily equity curve from realised trades.

    Each trade's profit is booked on its ``exit_date`` and the curve is
    forward-filled to a daily calendar grid, which is the cadence the log-equity
    regression in ``stable_metrics`` assumes (``BARS_PER_YEAR = 365``).

    Args:
        trades: dicts with at least ``exit_date`` and ``pl`` (base-currency P/L).
        initial_capital: starting equity.
        weight: column holding the realised P/L to accumulate.

    Returns:
        DataFrame with ``date`` and ``equity`` columns (empty if no trades).
    """
    if not trades:
        return pd.DataFrame(columns=["date", "equity"])

    df = pd.DataFrame(trades)
    if "exit_date" not in df.columns or weight not in df.columns:
        return pd.DataFrame(columns=["date", "equity"])

    df = df.dropna(subset=["exit_date"]).copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "equity"])

    # Realised P/L per calendar day.
    daily_pl = df.groupby(df["exit_date"].dt.normalize())[weight].sum().sort_index()

    # Anchor the curve one day before the first exit so the first booking shows up.
    start = (daily_pl.index.min() - pd.Timedelta(days=1)).normalize()
    end = daily_pl.index.max().normalize()
    grid = pd.date_range(start=start, end=end, freq="D")

    equity = pd.Series(initial_capital, index=grid, dtype=float)
    cumulative = daily_pl.cumsum()
    cumulative = cumulative.reindex(grid).ffill().fillna(0.0)
    equity = initial_capital + cumulative

    return pd.DataFrame({"date": grid, "equity": equity.values})


def _rar_and_r_squared(equity: np.ndarray, bars_per_year: int) -> Dict[str, float]:
    """Log-equity OLS → (rar_pct, r_squared). Mirrors stable_metrics, but with a
    configurable ``bars_per_year`` so the house formula stays editable."""
    equity = np.asarray(equity, dtype=float)
    mask = ~np.isnan(equity) & (equity > 0)
    equity = equity[mask]
    n = len(equity)
    if n < 2:
        return {"rar_pct": 0.0, "r_squared": 0.0}

    x = np.arange(1, n + 1, dtype=float)
    y = np.log(equity)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-10:
        return {"rar_pct": 0.0, "r_squared": 0.0}

    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    y_pred = y_mean + slope * (x - x_mean)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    r_squared = float(max(0.0, min(1.0, r_squared)))

    rar_pct = float((np.exp(slope * bars_per_year) - 1.0) * 100.0)
    return {"rar_pct": rar_pct, "r_squared": r_squared}


def adjusted_rar_from_equity(
    equity_curve: pd.DataFrame,
    config: Optional[AdjustedRARConfig] = None,
    equity_column: str = "equity",
) -> float:
    """Compute Adjusted RAR% from an equity curve under ``config``."""
    config = config or AdjustedRARConfig()
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    if equity_column in equity_curve.columns:
        equity = equity_curve[equity_column].values
    else:
        equity = equity_curve.iloc[:, -1].values

    parts = _rar_and_r_squared(equity, config.bars_per_year)
    value = parts["rar_pct"]
    if config.weight_by_r_squared:
        value *= parts["r_squared"]
    if config.clip_min is not None:
        value = max(value, config.clip_min)
    if config.clip_max is not None:
        value = min(value, config.clip_max)
    return float(value)


def adjusted_rar_from_trades(
    trades: List[Dict[str, Any]],
    config: Optional[AdjustedRARConfig] = None,
    initial_capital: float = 100_000.0,
) -> float:
    """Convenience: build the daily curve from trades then score Adjusted RAR%."""
    curve = build_daily_equity_curve(trades, initial_capital=initial_capital)
    return adjusted_rar_from_equity(curve, config=config)
