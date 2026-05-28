"""
Benchmark comparison for generated reports.

Loads a stored benchmark/index price series (collected via INDEX_DATA into
``raw_data/benchmarks/``) and compares it against a strategy equity curve,
producing the metrics reports need: benchmark equity overlay, total return and
CAGR vs benchmark, excess return, alpha, beta, correlation, tracking error,
information ratio, and up/down capture.

This is the single entry point used by every report generator so the
comparison is computed consistently everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..Core.performance_metrics import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from ..DataCollection.benchmark_collector import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_REGISTRY_PATH,
    load_benchmark_registry,
    resolve_benchmark,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkComparison:
    """Result of comparing a strategy equity curve against a benchmark."""

    is_valid: bool
    benchmark_name: str = ""
    benchmark_symbol: str = ""
    reason: str = ""

    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    strategy_total_return_pct: float = 0.0
    benchmark_total_return_pct: float = 0.0
    excess_return_pct: float = 0.0

    strategy_cagr: float = 0.0
    benchmark_cagr: float = 0.0

    alpha: float = 0.0            # annualized, percent
    beta: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0   # annualized, percent
    information_ratio: float = 0.0
    up_capture: float = 0.0       # percent
    down_capture: float = 0.0     # percent

    # date / equity DataFrame for an overlay chart (benchmark rebased to the
    # strategy's starting equity).
    benchmark_equity: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "benchmark_name": self.benchmark_name,
            "benchmark_symbol": self.benchmark_symbol,
            "strategy_total_return_pct": self.strategy_total_return_pct,
            "benchmark_total_return_pct": self.benchmark_total_return_pct,
            "excess_return_pct": self.excess_return_pct,
            "strategy_cagr": self.strategy_cagr,
            "benchmark_cagr": self.benchmark_cagr,
            "alpha": self.alpha,
            "beta": self.beta,
            "correlation": self.correlation,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "up_capture": self.up_capture,
            "down_capture": self.down_capture,
        }

    def summary_rows(self) -> List[Tuple[str, Any]]:
        """(label, formatted value) rows for rendering into a report table."""
        if not self.is_valid:
            return [("Benchmark comparison", f"Unavailable: {self.reason}")]
        return [
            ("Benchmark", f"{self.benchmark_name} ({self.benchmark_symbol})"),
            ("Comparison Window",
             f"{self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}"),
            ("Strategy Total Return", f"{self.strategy_total_return_pct:.2f}%"),
            ("Benchmark Total Return", f"{self.benchmark_total_return_pct:.2f}%"),
            ("Excess Return", f"{self.excess_return_pct:.2f}%"),
            ("Strategy CAGR", f"{self.strategy_cagr:.2f}%"),
            ("Benchmark CAGR", f"{self.benchmark_cagr:.2f}%"),
            ("Alpha (annualized)", f"{self.alpha:.2f}%"),
            ("Beta", f"{self.beta:.2f}"),
            ("Correlation", f"{self.correlation:.2f}"),
            ("Tracking Error (annualized)", f"{self.tracking_error:.2f}%"),
            ("Information Ratio", f"{self.information_ratio:.2f}"),
            ("Up Capture", f"{self.up_capture:.1f}%"),
            ("Down Capture", f"{self.down_capture:.1f}%"),
        ]


def _invalid(reason: str, name: str = "", symbol: str = "") -> BenchmarkComparison:
    return BenchmarkComparison(is_valid=False, benchmark_name=name, benchmark_symbol=symbol, reason=reason)


def _cagr(start_value: float, end_value: float, years: float) -> float:
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return 0.0
    return (pow(end_value / start_value, 1.0 / years) - 1.0) * 100.0


def compute_benchmark_comparison(
    equity_curve: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    benchmark_name: str = "",
    benchmark_symbol: str = "",
    risk_free_rate: Optional[float] = None,
) -> BenchmarkComparison:
    """
    Compare a strategy equity curve against a benchmark price series.

    Args:
        equity_curve: DataFrame with 'date' and 'equity' columns.
        benchmark_prices: DataFrame with 'date' and 'close' columns.
        benchmark_name / benchmark_symbol: labels for the report.
        risk_free_rate: annual risk-free rate (defaults to framework default).

    Returns:
        BenchmarkComparison (is_valid=False with a reason if it cannot be
        computed, e.g. missing columns or no overlapping dates).
    """
    if risk_free_rate is None:
        risk_free_rate = DEFAULT_RISK_FREE_RATE

    if equity_curve is None or equity_curve.empty or "equity" not in equity_curve.columns or "date" not in equity_curve.columns:
        return _invalid("strategy equity curve missing date/equity", benchmark_name, benchmark_symbol)
    if benchmark_prices is None or benchmark_prices.empty or "close" not in benchmark_prices.columns or "date" not in benchmark_prices.columns:
        return _invalid("benchmark price series unavailable", benchmark_name, benchmark_symbol)

    ec = equity_curve[["date", "equity"]].copy()
    ec["date"] = pd.to_datetime(ec["date"], errors="coerce")
    ec = ec.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)

    bp = benchmark_prices[["date", "close"]].copy()
    bp["date"] = pd.to_datetime(bp["date"], errors="coerce")
    bp["close"] = pd.to_numeric(bp["close"], errors="coerce")
    bp = bp.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)

    if len(ec) < 2 or len(bp) < 2:
        return _invalid("not enough data points", benchmark_name, benchmark_symbol)

    # Align the most recent benchmark close to each strategy date (handles a
    # benchmark sampled less frequently than the daily equity curve).
    merged = pd.merge_asof(ec, bp, on="date", direction="backward")
    merged = merged.dropna(subset=["close"]).reset_index(drop=True)
    if len(merged) < 2:
        return _invalid("no overlapping dates between strategy and benchmark", benchmark_name, benchmark_symbol)

    start_equity = float(merged["equity"].iloc[0])
    start_close = float(merged["close"].iloc[0])
    if start_equity <= 0 or start_close <= 0:
        return _invalid("non-positive starting values", benchmark_name, benchmark_symbol)

    start_date = merged["date"].iloc[0]
    end_date = merged["date"].iloc[-1]
    years = (end_date - start_date).days / 365.25

    # Benchmark rebased to the strategy's starting equity (for an overlay).
    benchmark_equity = pd.DataFrame({
        "date": merged["date"],
        "equity": start_equity * (merged["close"] / start_close),
    })

    strat_total = (float(merged["equity"].iloc[-1]) / start_equity - 1.0) * 100.0
    bench_total = (float(merged["close"].iloc[-1]) / start_close - 1.0) * 100.0

    r_s = merged["equity"].pct_change()
    r_b = merged["close"].pct_change()
    rets = pd.DataFrame({"s": r_s, "b": r_b}).replace([np.inf, -np.inf], np.nan).dropna()

    beta = correlation = alpha = tracking_error = information_ratio = 0.0
    up_capture = down_capture = 0.0

    if len(rets) >= 2:
        var_b = float(rets["b"].var())
        if var_b > 0:
            beta = float(rets["s"].cov(rets["b"]) / var_b)
        if rets["s"].std() > 0 and rets["b"].std() > 0:
            correlation = float(rets["s"].corr(rets["b"]))

        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        alpha_daily = (rets["s"] - daily_rf).mean() - beta * (rets["b"] - daily_rf).mean()
        alpha = (pow(1 + alpha_daily, TRADING_DAYS_PER_YEAR) - 1) * 100.0

        active = rets["s"] - rets["b"]
        active_std = float(active.std())
        if active_std > 0:
            tracking_error = active_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
            information_ratio = float(active.mean() / active_std * np.sqrt(TRADING_DAYS_PER_YEAR))

        up = rets[rets["b"] > 0]
        down = rets[rets["b"] < 0]
        if len(up) > 0:
            bench_up = (1 + up["b"]).prod() - 1
            strat_up = (1 + up["s"]).prod() - 1
            if bench_up != 0:
                up_capture = float(strat_up / bench_up * 100.0)
        if len(down) > 0:
            bench_dn = (1 + down["b"]).prod() - 1
            strat_dn = (1 + down["s"]).prod() - 1
            if bench_dn != 0:
                down_capture = float(strat_dn / bench_dn * 100.0)

    return BenchmarkComparison(
        is_valid=True,
        benchmark_name=benchmark_name,
        benchmark_symbol=benchmark_symbol,
        start_date=start_date,
        end_date=end_date,
        strategy_total_return_pct=strat_total,
        benchmark_total_return_pct=bench_total,
        excess_return_pct=strat_total - bench_total,
        strategy_cagr=_cagr(start_equity, float(merged["equity"].iloc[-1]), years),
        benchmark_cagr=_cagr(start_close, float(merged["close"].iloc[-1]), years),
        alpha=alpha,
        beta=beta,
        correlation=correlation,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        up_capture=up_capture,
        down_capture=down_capture,
        benchmark_equity=benchmark_equity,
    )


def write_comparison_sheet(
    ws,
    comparison: Optional[BenchmarkComparison],
    *,
    title: str = "BENCHMARK COMPARISON",
    collect_hint: str = "python scripts/collect_benchmarks.py",
) -> None:
    """
    Render a benchmark comparison onto an openpyxl worksheet.

    Shared by the portfolio report generators so the benchmark section looks
    the same everywhere. Renders an "unavailable" note (never raises) when the
    comparison could not be computed.
    """
    from openpyxl.styles import Font

    ws.column_dimensions['A'].width = 34
    ws.column_dimensions['B'].width = 26

    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=14)
    row = 3

    if comparison is None or not comparison.is_valid:
        reason = comparison.reason if comparison is not None else "benchmark module unavailable"
        ws.cell(row=row, column=1, value="Benchmark comparison unavailable").font = Font(bold=True)
        ws.cell(row=row + 1, column=1, value=f"Reason: {reason}")
        ws.cell(row=row + 2, column=1, value=f"Collect index data with: {collect_hint}")
        return

    for label, value in comparison.summary_rows():
        ws.cell(row=row, column=1, value=label).font = Font(bold=label == "Benchmark")
        ws.cell(row=row, column=2, value=value)
        row += 1


class BenchmarkLoader:
    """Resolves and loads stored benchmark series, and runs comparisons."""

    def __init__(
        self,
        benchmarks_dir: Optional[Path] = None,
        registry_path: Optional[Path] = None,
    ):
        self.benchmarks_dir = Path(benchmarks_dir) if benchmarks_dir else DEFAULT_BENCHMARK_DIR
        self.registry = load_benchmark_registry(registry_path or DEFAULT_REGISTRY_PATH)

    def default_name(self) -> Optional[str]:
        return self.registry.get("default")

    def _file_candidates(self, symbol: str, interval: str) -> List[Path]:
        return [
            self.benchmarks_dir / f"{symbol}_{interval}.csv",
            self.benchmarks_dir / f"{symbol}.csv",
            self.benchmarks_dir / f"{symbol}_daily.csv",
        ]

    def load_series(self, name_or_symbol: str) -> pd.DataFrame:
        """Load a benchmark's [date, close] series. Empty if unavailable."""
        resolved = resolve_benchmark(name_or_symbol, self.registry)
        if resolved:
            _, entry = resolved
            symbol = entry.get("symbol", name_or_symbol)
            interval = entry.get("interval", "daily")
        else:
            symbol, interval = str(name_or_symbol), "daily"

        for path in self._file_candidates(symbol, interval):
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    df.columns = [str(c).lower().strip() for c in df.columns]
                    if "date" in df.columns and "close" in df.columns:
                        return df[["date", "close"]]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed reading benchmark %s: %s", path, exc)
        return pd.DataFrame()

    def resolved_label(self, name_or_symbol: str) -> Tuple[str, str]:
        """Return (friendly_name, symbol) for display."""
        resolved = resolve_benchmark(name_or_symbol, self.registry)
        if resolved:
            name, entry = resolved
            return name, entry.get("symbol", "")
        return str(name_or_symbol), str(name_or_symbol)

    def compare(
        self,
        equity_curve: pd.DataFrame,
        benchmark_name: Optional[str] = None,
        risk_free_rate: Optional[float] = None,
    ) -> BenchmarkComparison:
        """Load the configured benchmark and compare it to an equity curve."""
        name = benchmark_name or self.default_name()
        if not name:
            return _invalid("no benchmark configured")

        friendly, symbol = self.resolved_label(name)
        series = self.load_series(name)
        if series.empty:
            return _invalid(
                f"no stored data for '{friendly}' (collect it first)",
                friendly, symbol,
            )
        return compute_benchmark_comparison(
            equity_curve, series,
            benchmark_name=friendly, benchmark_symbol=symbol,
            risk_free_rate=risk_free_rate,
        )
