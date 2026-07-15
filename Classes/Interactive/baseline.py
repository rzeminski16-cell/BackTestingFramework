"""
AUTO (rules-only) baseline twin for interactive runs.

After an interactive run completes, the identical configuration is
re-run with ``decision_session=None`` — a fresh engine and a fresh
strategy instance, the same config and data — so the discretionary
layer's impact can be measured against what the rules alone would have
done. In portfolio mode the baseline uses the configured capital
contention mode (interactive mode superseded it); that difference is
part of what is being measured.

Baseline artifacts are written under <run folder>/baseline/ with their
own small manifest linking back to the interactive run.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from ..Config.config import BacktestConfig, PortfolioConfig
from ..Engine.portfolio_engine import PortfolioEngine, PortfolioBacktestResult
from ..Engine.single_security_engine import SingleSecurityEngine
from .models import BacktestRunManifest


def run_auto_baseline(engine_type: str,
                      config,
                      data_by_symbol: Dict[str, pd.DataFrame],
                      strategy_factory: Callable[[], Any],
                      currency_converter=None,
                      security_registry=None,
                      progress_callback=None):
    """
    Run the rules-only twin of an interactive run.

    Args:
        engine_type: "single" | "portfolio"
        config: The identical BacktestConfig / PortfolioConfig used by
            the interactive run.
        data_by_symbol: Symbol -> raw DataFrame (pre prepare_data; the
            engine prepares it exactly as the interactive run did).
        strategy_factory: Zero-arg callable returning a FRESH strategy
            instance (strategies hold per-run state).
        currency_converter / security_registry / progress_callback:
            passed through to the engine.

    Returns:
        BacktestResult (single) or PortfolioBacktestResult (portfolio).
    """
    strategy = strategy_factory()
    if engine_type == "single":
        symbol = next(iter(data_by_symbol))
        engine = SingleSecurityEngine(config, currency_converter,
                                      security_registry)
        return engine.run(symbol, data_by_symbol[symbol].copy(), strategy,
                          progress_callback=progress_callback)

    engine = PortfolioEngine(config, currency_converter, security_registry)
    data_copy = {sym: df.copy() for sym, df in data_by_symbol.items()}
    return engine.run(data_copy, strategy,
                      progress_callback=progress_callback)


def extract_trades_and_equity(result):
    """
    Normalize a BacktestResult / PortfolioBacktestResult into
    (trades list sorted by entry date, equity_curve DataFrame).
    """
    if isinstance(result, PortfolioBacktestResult) or hasattr(
            result, 'portfolio_equity_curve'):
        trades = [t for r in result.symbol_results.values() for t in r.trades]
        trades.sort(key=lambda t: (str(t.entry_date), t.symbol))
        return trades, result.portfolio_equity_curve
    return list(result.trades), result.equity_curve


def write_baseline_outputs(result, run_dir: Path,
                           interactive_run_id: str) -> Path:
    """
    Persist the baseline's trades, equity curve, and manifest under
    <run_dir>/baseline/. Returns the baseline directory.
    """
    baseline_dir = Path(run_dir) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    trades, equity_curve = extract_trades_and_equity(result)
    if trades:
        pd.DataFrame([t.to_dict() for t in trades]).to_csv(
            baseline_dir / "baseline_trades.csv", index=False)
    if equity_curve is not None and len(equity_curve) > 0:
        equity_curve.to_csv(baseline_dir / "baseline_equity.csv", index=False)

    manifest = {
        'mode': "auto_baseline",
        'interactive_run_id': interactive_run_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'final_equity': float(result.final_equity),
        'total_return': float(result.total_return),
        'total_return_pct': float(result.total_return_pct),
        'num_trades': len(trades),
    }
    with open(baseline_dir / "baseline_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    return baseline_dir
