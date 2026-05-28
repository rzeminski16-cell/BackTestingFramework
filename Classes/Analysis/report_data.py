"""
Pure data computations for report visualizations (no matplotlib, no openpyxl).

This module is the SINGLE SOURCE OF TRUTH for every chart's plotted series. Both
the matplotlib PNG path (``EnhancedVisualizations.create_*``) and the native
Excel chart path consume these functions, so the two render byte-for-byte
identical numbers and cannot drift.

Reliability rules:
- Reuse the framework's centralized metrics where applicable (e.g. R-multiples).
- These functions compute the *chart series* only; headline metrics (CAGR,
  Sharpe, drawdown shown elsewhere) come from PerformanceMetrics on the FULL
  series, never from any down-sampled chart array.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

RISK_FREE_ANNUAL = 0.035  # matches EnhancedVisualizations rolling-metric default


def _iso(dt) -> str:
    ts = pd.Timestamp(dt)
    return ts.strftime("%Y-%m-%d")


def compute_equity_drawdown(equity_curve: pd.DataFrame) -> Dict[str, List]:
    """Equity, high-water-mark and underwater drawdown% series for the chart."""
    dates = pd.to_datetime(equity_curve["date"])
    equity = np.asarray(equity_curve["equity"].values, dtype=float)
    if len(equity) == 0:
        return {"dates": [], "equity": [], "high_water_mark": [], "drawdown_pct": []}
    running_max = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown_pct = ((equity - running_max) / running_max) * 100
    drawdown_pct = np.nan_to_num(drawdown_pct, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "dates": [_iso(d) for d in dates],
        "equity": [float(x) for x in equity],
        "high_water_mark": [float(x) for x in running_max],
        "drawdown_pct": [float(x) for x in drawdown_pct],
    }


def compute_monthly_returns(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    """Year x month returns grid (%) with a Year Total column. Matches PNG path."""
    df = equity_curve.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    monthly = df["equity"].resample("ME").last()
    monthly_returns = (monthly.pct_change() * 100).dropna()
    if len(monthly_returns) < 2:
        return {"years": [], "months": MONTH_NAMES, "rows": [], "year_totals": []}

    mr = monthly_returns.reset_index()
    mr.columns = ["date", "return"]
    mr["year"] = mr["date"].dt.year
    mr["month"] = mr["date"].dt.month
    pivot = mr.pivot_table(values="return", index="year", columns="month", aggfunc="first")

    years = [int(y) for y in pivot.index]
    rows: List[List[Optional[float]]] = []
    year_totals: List[float] = []
    for y in pivot.index:
        row: List[Optional[float]] = []
        for m in range(1, 13):
            val = pivot.loc[y, m] if m in pivot.columns else np.nan
            row.append(None if pd.isna(val) else float(val))
        rows.append(row)
        year_totals.append(float(np.nansum([v for v in row if v is not None] or [0.0])))
    return {"years": years, "months": MONTH_NAMES, "rows": rows, "year_totals": year_totals}


def compute_trade_return_distribution(trades: List[Any]) -> Dict[str, Any]:
    """Histogram of trade return % (same binning as the matplotlib path)."""
    if not trades:
        return {"labels": [], "counts": [], "n": 0}
    returns = np.asarray([t.pl_pct for t in trades], dtype=float)
    n_bins = min(50, max(10, len(returns) // 5))
    counts, edges = np.histogram(returns, bins=n_bins)
    labels = [f"{(edges[i] + edges[i + 1]) / 2:.1f}" for i in range(len(counts))]
    return {
        "labels": labels,
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "n": int(len(returns)),
        "mean": float(np.mean(returns)),
        "median": float(np.median(returns)),
        "std": float(np.std(returns)),
        "win_rate": float(sum(1 for r in returns if r > 0) / len(returns) * 100),
    }


def compute_r_multiple_distribution(trades: List[Any]) -> Dict[str, Any]:
    """
    R-multiple distribution over integer (1R) buckets, plus expectancy stats.

    Bins all R-multiples on integer edges from floor(min) to ceil(max)+1; this
    yields per-bucket counts identical to the matplotlib path's split
    losing/winning histograms.
    """
    from Classes.Core.performance_metrics import CentralizedPerformanceMetrics

    r_multiples = CentralizedPerformanceMetrics.calculate_r_multiples(trades)
    if not r_multiples:
        return {"labels": [], "counts": [], "n": 0, "available": False}

    arr = np.asarray(r_multiples, dtype=float)
    winning = [r for r in r_multiples if r >= 0]
    losing = [r for r in r_multiples if r < 0]
    min_bin = int(np.floor(arr.min()))
    max_bin = int(np.ceil(arr.max())) + 1
    edges = np.arange(min_bin, max_bin + 1, 1)
    if len(edges) < 2:
        edges = np.arange(min_bin, min_bin + 2, 1)
    counts, edges = np.histogram(arr, bins=edges)
    labels = [f"[{int(edges[i])},{int(edges[i + 1])})" for i in range(len(counts))]

    total = len(r_multiples)
    avg_win_r = float(np.mean(winning)) if winning else 0.0
    avg_loss_r = float(np.mean(losing)) if losing else 0.0
    return {
        "labels": labels,
        "counts": [int(c) for c in counts],
        "n": total,
        "available": True,
        "avg_r": float(np.mean(arr)),
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "win_rate": float(len(winning) / total * 100),
        "r_expectancy": float((len(winning) / total * avg_win_r) + (len(losing) / total * avg_loss_r)),
        "n_win": len(winning),
        "n_loss": len(losing),
    }


def compute_rolling_metrics(
    equity_curve: pd.DataFrame,
    window: int = 90,
    filter_anomalies: bool = True,
    anomaly_absolute_threshold: float = 10.0,
    anomaly_zscore_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Rolling Sharpe/Sortino/volatility series with the same anomaly filtering as
    the matplotlib chart. Anomalous Sharpe/Sortino points become None (chart gap).
    """
    df = equity_curve.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["returns"] = df["equity"].pct_change()

    anomalies: List[Dict[str, Any]] = []
    if len(df) < window + 10:
        return {"dates": [], "sharpe": [], "sortino": [], "volatility": [],
                "anomalies": anomalies, "available": False}

    risk_free_daily = (1 + RISK_FREE_ANNUAL) ** (1 / 252) - 1

    rolling_mean = df["returns"].rolling(window).mean()
    rolling_std = df["returns"].rolling(window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rolling_sharpe"] = ((rolling_mean - risk_free_daily) / rolling_std) * np.sqrt(252)
    df["rolling_sharpe"] = df["rolling_sharpe"].replace([np.inf, -np.inf], 0.0)

    returns = df["returns"]
    sortino_vals: List[float] = []
    for i in range(len(returns)):
        if i < window:
            sortino_vals.append(np.nan)
        else:
            window_returns = returns.iloc[i - window:i]
            excess = window_returns - risk_free_daily
            downside = window_returns[window_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = (excess.mean() / downside.std()) * np.sqrt(252)
                if np.isinf(sortino):
                    sortino = 0.0
            else:
                sortino = 0.0 if excess.mean() <= 0 else 99.99
            sortino_vals.append(sortino)
    df["rolling_sortino"] = sortino_vals
    df["rolling_volatility"] = rolling_std * np.sqrt(252) * 100

    if filter_anomalies:
        for metric, label in (("rolling_sharpe", "Sharpe Ratio"), ("rolling_sortino", "Sortino Ratio")):
            series = df[metric].dropna()
            if len(series) < 10:
                continue
            mean = series.mean()
            std = series.std()
            for date, value in series.items():
                is_anomaly = False
                reasons = []
                if abs(value) > anomaly_absolute_threshold:
                    is_anomaly = True
                    reasons.append(f"|value| > {anomaly_absolute_threshold}")
                if std > 0 and abs(value - mean) / std > anomaly_zscore_threshold:
                    is_anomaly = True
                    reasons.append(f"z-score {abs(value - mean) / std:.1f} > {anomaly_zscore_threshold}")
                if is_anomaly:
                    anomalies.append({"date": _iso(date), "metric": label,
                                      "value": float(value), "reason": " AND ".join(reasons)})
                    df.loc[date, metric] = np.nan

    def _col(name):
        return [None if pd.isna(v) else float(v) for v in df[name]]

    return {
        "dates": [_iso(d) for d in df.index],
        "sharpe": _col("rolling_sharpe"),
        "sortino": _col("rolling_sortino"),
        "volatility": _col("rolling_volatility"),
        "anomalies": anomalies,
        "available": True,
    }


def compute_win_rate_over_time(trades: List[Any], window: int = 20) -> Dict[str, Any]:
    """Rolling win-rate (%) and profit factor (capped at 10) over trades."""
    if len(trades) < window + 5:
        return {"dates": [], "win_rates": [], "profit_factors": [], "available": False}
    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    dates, win_rates, profit_factors = [], [], []
    for i in range(window, len(sorted_trades) + 1):
        window_trades = sorted_trades[i - window:i]
        dates.append(_iso(window_trades[-1].exit_date))
        wins = sum(1 for t in window_trades if t.pl > 0)
        win_rates.append(wins / window * 100)
        gross_profit = sum(t.pl for t in window_trades if t.pl > 0)
        gross_loss = abs(sum(t.pl for t in window_trades if t.pl < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 10
        profit_factors.append(min(pf, 10))
    return {"dates": dates, "win_rates": win_rates, "profit_factors": profit_factors, "available": True}


def compute_mae_mfe(trades: List[Any]) -> Dict[str, Any]:
    """Stop-loss-distance %, trade return %, and duration per trade (for scatter)."""
    if not trades:
        return {"mae": [], "returns": [], "durations": [], "available": False}
    returns, mae_estimates, durations = [], [], []
    for t in trades:
        returns.append(float(t.pl_pct))
        durations.append(float(t.duration_days))
        if t.initial_stop_loss and t.entry_price > 0:
            mae_estimates.append(abs((t.initial_stop_loss - t.entry_price) / t.entry_price * 100))
        else:
            mae_estimates.append(abs(min(0, t.pl_pct)))
    return {"mae": mae_estimates, "returns": returns, "durations": durations, "available": True}


def compute_contribution(symbol_pnl: Dict[str, float]) -> Dict[str, Any]:
    """P/L by security (sorted by absolute contribution) plus contribution share %."""
    if not symbol_pnl:
        return {"symbols": [], "pnls": [], "total": 0.0, "shares": [], "available": False}
    symbols = sorted(symbol_pnl.keys(), key=lambda s: abs(symbol_pnl[s]), reverse=True)
    pnls = [float(symbol_pnl[s]) for s in symbols]
    total = float(sum(pnls))
    abs_total = sum(abs(p) for p in pnls) or 1.0
    shares = [abs(p) / abs_total * 100 for p in pnls]
    return {"symbols": symbols, "pnls": pnls, "total": total, "shares": shares, "available": True}


def compute_capital_utilization(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    """Cash vs invested capital over time (stacks to total equity)."""
    df = equity_curve.copy()
    if "capital" not in df.columns or "position_value" not in df.columns:
        return {"available": False}
    df["date"] = pd.to_datetime(df["date"])
    cash = np.asarray(df["capital"].values, dtype=float)
    equity = np.asarray(df["equity"].values, dtype=float)
    invested = equity - cash
    return {
        "dates": [_iso(d) for d in df["date"]],
        "cash": [float(c) for c in cash],
        "invested": [float(v) for v in invested],
        "available": True,
    }


def compute_streaks(trades: List[Any]) -> Dict[str, Any]:
    """Win/loss streak sequence, length distribution, and summary stats."""
    if not trades:
        return {"available": False}
    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    streaks = []  # (length, type, pnl)
    cur_len, cur_type, cur_pnl = 0, None, 0.0
    for trade in sorted_trades:
        is_win = trade.pl > 0
        if cur_type is None:
            cur_type, cur_len, cur_pnl = is_win, 1, trade.pl
        elif is_win == cur_type:
            cur_len += 1
            cur_pnl += trade.pl
        else:
            streaks.append((cur_len, "Win" if cur_type else "Loss", cur_pnl))
            cur_type, cur_len, cur_pnl = is_win, 1, trade.pl
    streaks.append((cur_len, "Win" if cur_type else "Loss", cur_pnl))

    win_streaks = [s[0] for s in streaks if s[1] == "Win"]
    loss_streaks = [s[0] for s in streaks if s[1] == "Loss"]
    max_streak = max(max(win_streaks, default=1), max(loss_streaks, default=1))
    lengths = list(range(1, max_streak + 1))
    win_counts = [win_streaks.count(x) for x in lengths]
    loss_counts = [loss_streaks.count(x) for x in lengths]

    return {
        "available": True,
        "sequence_lengths": [s[0] if s[1] == "Win" else -s[0] for s in streaks],
        "sequence_pnls": [float(s[2]) for s in streaks],
        "lengths": lengths,
        "win_counts": win_counts,
        "loss_counts": loss_counts,
        "stats": {
            "num_win_streaks": len(win_streaks),
            "max_win_streak": max(win_streaks) if win_streaks else 0,
            "avg_win_streak": float(np.mean(win_streaks)) if win_streaks else 0.0,
            "num_loss_streaks": len(loss_streaks),
            "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_loss_streak": float(np.mean(loss_streaks)) if loss_streaks else 0.0,
            "total_streaks": len(streaks),
        },
    }
