"""
Implementations of the ``btf`` CLI commands.

Each command is a plain function taking the parsed argparse namespace and
returning a process exit code. Heavy imports happen inside the functions so
``btf --help`` stays fast and commands only pay for what they use.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "raw_data" / "daily"
DEFAULT_FOREX_DIR = REPO_ROOT / "raw_data" / "forex"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas/datetime values into JSON-serialisable ones."""
    import numpy as np
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _parse_params(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Parse repeated ``--param name=value`` into a typed dict."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects name=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        for caster in (int, float):
            try:
                out[key] = caster(raw)
                break
            except ValueError:
                continue
        else:
            if raw.lower() in ("true", "false"):
                out[key] = raw.lower() == "true"
            else:
                out[key] = raw
    return out


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"Dates must be YYYY-MM-DD, got {value!r}")


def _print_metrics(metrics: Dict[str, Any], keys: List[str]) -> None:
    width = max(len(k) for k in keys) + 2
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key:<{width}} {value:,.4f}")
        else:
            print(f"  {key:<{width}} {value}")


def _build_fx(base_currency: str):
    """Wire the FX converter + security registry the way the GUIs do.

    Returns (converter, registry) or (None, None) when FX data is absent —
    the engines then run in single-currency mode.
    """
    try:
        from Classes.Data.currency_converter import CurrencyConverter
        from Classes.Data.security_registry import SecurityRegistry
        if not DEFAULT_FOREX_DIR.exists():
            return None, None
        converter = CurrencyConverter(base_currency=base_currency)
        converter.load_rates_directory(DEFAULT_FOREX_DIR)
        registry = SecurityRegistry()
        return converter, registry
    except Exception as exc:
        logger.warning("FX wiring unavailable (%s); running single-currency.", exc)
        return None, None


# --------------------------------------------------------------------------- #
# btf list
# --------------------------------------------------------------------------- #

def cmd_list(args) -> int:
    what = args.what

    if what in ("strategies", "all"):
        from strategies.registry import STRATEGY_REGISTRY
        print("Strategies:")
        for name in sorted(STRATEGY_REGISTRY):
            print(f"  {name}")

    if what in ("securities", "all"):
        from Classes.Data.data_loader import DataLoader
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            symbols = DataLoader(data_dir).get_available_symbols()
            print(f"Securities ({len(symbols)} in {data_dir}):")
            for sym in symbols:
                print(f"  {sym}")
        else:
            print(f"Securities: data directory not found: {data_dir}")

    if what in ("baskets", "all"):
        from Classes.Config.basket import BasketManager
        names = BasketManager().list_baskets()
        print(f"Baskets ({len(names)}):")
        for name in names:
            print(f"  {name}")

    if what in ("benchmarks", "all"):
        try:
            from Classes.DataCollection.benchmark_collector import (
                DEFAULT_REGISTRY_PATH, load_benchmark_registry)
            registry = load_benchmark_registry(DEFAULT_REGISTRY_PATH)
            entries = registry.get("benchmarks", {})
            default = registry.get("default")
            print(f"Benchmarks ({len(entries)}):")
            for name, entry in entries.items():
                marker = " (default)" if name == default else ""
                print(f"  {name} -> {entry.get('symbol', '?')}{marker}")
        except Exception as exc:
            print(f"Benchmarks: unavailable ({exc})")

    return 0


# --------------------------------------------------------------------------- #
# btf backtest
# --------------------------------------------------------------------------- #

def cmd_backtest(args) -> int:
    import pandas as pd  # noqa: F401  (ensures a clear error before engine imports)
    from Classes.Config.config import (
        BacktestConfig, CommissionConfig, CommissionMode, ExecutionTiming,
        PortfolioConfig,
    )
    from Classes.Data.data_loader import DataLoader
    from strategies.registry import get_strategy_class

    try:
        strategy_class = get_strategy_class(args.strategy)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    params = _parse_params(args.param)
    try:
        strategy = strategy_class(**params)
    except Exception as exc:
        print(f"error: could not construct {args.strategy} with {params}: {exc}",
              file=sys.stderr)
        return 2

    # Resolve the symbol set.
    symbols: List[str] = []
    if args.basket:
        from Classes.Config.basket import BasketManager
        basket = BasketManager().load_basket(args.basket)
        if basket is None:
            print(f"error: basket not found: {args.basket}", file=sys.stderr)
            return 2
        symbols = list(basket.symbols)
    elif args.symbols:
        symbols = list(args.symbols)
    if not symbols:
        print("error: provide --symbols (one or more) or --basket", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 2
    loader = DataLoader(data_dir)

    commission = CommissionConfig(
        mode=CommissionMode.FIXED if args.fixed_commission else CommissionMode.PERCENTAGE,
        value=args.commission,
    )
    timing = (ExecutionTiming.NEXT_BAR_OPEN if args.next_bar_open
              else ExecutionTiming.SAME_BAR_CLOSE)
    converter, registry = (None, None) if args.no_fx else _build_fx(args.base_currency)

    portfolio_mode = len(symbols) > 1
    if portfolio_mode:
        result, trades, equity_curve = _run_portfolio_backtest(
            args, symbols, strategy, loader, commission, timing,
            converter, registry, PortfolioConfig)
    else:
        result, trades, equity_curve = _run_single_backtest(
            args, symbols[0], strategy, loader, commission, timing,
            converter, registry, BacktestConfig)
    if result is None:
        return 1

    # Metrics summary.
    from Classes.Core.performance_metrics import CentralizedPerformanceMetrics
    metrics = CentralizedPerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_curve, trades=trades,
        initial_capital=args.capital,
    )
    metrics["final_equity"] = result.final_equity
    metrics["total_return_dollars"] = result.total_return
    metrics["total_return_pct"] = result.total_return_pct

    print(f"\n{args.strategy} on {', '.join(symbols)}"
          f"  [{timing.value}{', intrabar stops' if args.intrabar_stops else ''}]")
    print("-" * 60)
    _print_metrics(metrics, [
        "final_equity", "total_return_pct", "annual_return", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "max_drawdown_pct", "exposure_pct",
        "var_95", "cvar_95", "total_trades", "win_rate", "profit_factor",
        "avg_mfe_pct", "avg_mae_pct",
    ])

    # Outputs.
    if args.trades_csv:
        out = Path(args.trades_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.to_dict() for t in trades]).to_csv(out, index=False)
        print(f"\nTrade log: {out}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(metrics), indent=2))
        print(f"Metrics JSON: {out}")

    if args.report:
        _write_backtest_report(args, result, symbols, portfolio_mode)

    return 0


def _run_single_backtest(args, symbol, strategy, loader, commission, timing,
                         converter, registry, BacktestConfig):
    from Classes.Engine.single_security_engine import SingleSecurityEngine
    config = BacktestConfig(
        initial_capital=args.capital,
        commission=commission,
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        slippage_percent=args.slippage,
        base_currency=args.base_currency,
        execution_timing=timing,
        intrabar_stops=args.intrabar_stops,
    )
    try:
        data = loader.load_csv(symbol, required_columns=strategy.required_columns())
        engine = SingleSecurityEngine(config, currency_converter=converter,
                                      security_registry=registry)
        result = engine.run(symbol, data, strategy)
    except Exception as exc:
        print(f"error: backtest failed: {exc}", file=sys.stderr)
        return None, None, None
    return result, result.trades, result.equity_curve


def _run_portfolio_backtest(args, symbols, strategy, loader, commission, timing,
                            converter, registry, PortfolioConfig):
    from Classes.Engine.portfolio_engine import PortfolioEngine
    config = PortfolioConfig(
        initial_capital=args.capital,
        commission=commission,
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        slippage_percent=args.slippage,
        base_currency=args.base_currency,
        basket_name=args.basket,
        execution_timing=timing,
        intrabar_stops=args.intrabar_stops,
    )
    try:
        data_dict = {}
        for sym in symbols:
            data_dict[sym] = loader.load_csv(
                sym, required_columns=strategy.required_columns())
        engine = PortfolioEngine(config, currency_converter=converter,
                                 security_registry=registry)
        result = engine.run(data_dict, strategy)
    except Exception as exc:
        print(f"error: portfolio backtest failed: {exc}", file=sys.stderr)
        return None, None, None
    trades = [t for r in result.symbol_results.values() for t in r.trades]
    return result, trades, result.portfolio_equity_curve


def _write_backtest_report(args, result, symbols, portfolio_mode) -> None:
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if portfolio_mode:
            from Classes.Analysis.portfolio_report_generator import (
                PortfolioReportGenerator)
            generator = PortfolioReportGenerator(out_dir, use_enhanced=True)
            path = generator.generate_portfolio_report(result)
        else:
            from Classes.Analysis.excel_report_generator import ExcelReportGenerator
            generator = ExcelReportGenerator(output_directory=out_dir,
                                             initial_capital=args.capital)
            path = generator.generate_report(
                result=result,
                filename=f"cli_{symbols[0]}_{stamp}.xlsx")
        print(f"Excel report: {path}")
    except Exception as exc:
        print(f"warning: Excel report generation failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# btf optimize
# --------------------------------------------------------------------------- #

def cmd_optimize(args) -> int:
    from Classes.Data.data_loader import DataLoader
    from Classes.Optimization.walk_forward_optimizer import (
        WalkForwardMode, WalkForwardOptimizer)
    from strategies.registry import get_strategy_class

    try:
        strategy_class = get_strategy_class(args.strategy)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 2

    try:
        data = DataLoader(data_dir).load_csv(args.symbol)
    except Exception as exc:
        print(f"error: could not load {args.symbol}: {exc}", file=sys.stderr)
        return 2

    mode = WalkForwardMode(args.mode) if args.mode else None
    selected = {name: True for name in args.optimize_params} if args.optimize_params else None

    optimizer = WalkForwardOptimizer(config_path=args.config)

    def progress(stage, current, total):
        print(f"\r{stage} [{current}/{total}]", end="", flush=True)

    print(f"Walk-forward optimization: {args.strategy} on {args.symbol}")
    results = optimizer.optimize(
        strategy_class=strategy_class,
        symbol=args.symbol,
        data=data,
        selected_params=selected,
        progress_callback=progress,
        walk_forward_mode=mode,
    )
    print()

    print(f"\nWindows: {results.total_windows} "
          f"(passed constraints: {results.windows_passed_constraints})")
    print(f"Avg OOS Sortino: {results.avg_out_sample_sortino:.3f}   "
          f"Avg OOS Sharpe: {results.avg_out_sample_sharpe:.3f}   "
          f"Avg Sharpe degradation: {results.avg_sharpe_degradation_pct:.1f}%")
    print("Most common parameters:")
    for key, value in results.most_common_params.items():
        print(f"  {key} = {value}")
    print("\nPer-window (IS DSR = deflated in-sample Sharpe; >0.95 survives "
          "its own search):")
    for w in results.windows:
        print(f"  W{w.window_id + 1}: OOS Sharpe {w.out_sample_sharpe:6.3f}  "
              f"OOS trades {w.out_sample_num_trades:3d}  "
              f"IS DSR {w.in_sample_dsr:.3f} ({w.n_trials} trials)")

    if args.report:
        try:
            from Classes.Optimization.optimization_report_generator import (
                OptimizationReportGenerator)
            out_dir = Path(args.report_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            generator = OptimizationReportGenerator(out_dir)
            path = generator.generate_report(results)
            print(f"\nReport: {path}")
        except Exception as exc:
            print(f"warning: report generation failed: {exc}", file=sys.stderr)

    return 0


# --------------------------------------------------------------------------- #
# btf montecarlo
# --------------------------------------------------------------------------- #

def cmd_montecarlo(args) -> int:
    import numpy as np
    from Classes.MonteCarloAnalysis.analyzer import SimulationAnalyzer
    from Classes.MonteCarloAnalysis.config import (
        SamplingMethod, SimulationConfig, SizingMethod)
    from Classes.MonteCarloAnalysis.simulator import MonteCarloSimulator
    from Classes.MonteCarloAnalysis.trade_log_loader import (
        TradeLogReturnSource, load_daily_returns, load_trade_logs)

    periods_per_year = args.periods_per_year

    if args.daily_curve:
        pool, warnings = load_daily_returns(args.daily_curve)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if pool.size == 0:
            print("error: no usable daily returns loaded", file=sys.stderr)
            return 2
        risk = 1.0  # each step is a full day of portfolio exposure
        if periods_per_year is None:
            periods_per_year = 252.0
        print(f"Pool: {pool.size} daily returns from {args.daily_curve}")
    elif args.trade_log:
        loaded = load_trade_logs(args.trade_log)
        for warning in loaded.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        source = (TradeLogReturnSource.R_MULTIPLE if args.source == "r"
                  else TradeLogReturnSource.PCT_RETURN)
        pool = loaded.returns_for(source)
        if pool.size == 0:
            print("error: trade log produced an empty return pool", file=sys.stderr)
            return 2
        risk = args.risk
        print(f"Pool: {pool.size} {args.source} returns from {len(loaded.source_files)} file(s)")
    else:
        print("error: provide --trade-log or --daily-curve", file=sys.stderr)
        return 2

    config = SimulationConfig(
        num_simulations=args.simulations,
        num_trades=args.steps,
        initial_capital=args.capital,
        sampling_method=(SamplingMethod.BLOCK_BOOTSTRAP if args.block_size
                         else SamplingMethod.SIMPLE_BOOTSTRAP),
        block_size=args.block_size or 10,
        sizing_method=SizingMethod.COMPOUNDING,
        risk_per_trade=risk,
        random_seed=args.seed,
        periods_per_year=periods_per_year,
    )
    result = MonteCarloSimulator(config).run(np.asarray(pool, dtype="float64"))
    metrics = SimulationAnalyzer(result).metrics()

    d = metrics.to_dict()
    print(f"\nMonte Carlo: {args.simulations} paths x {args.steps} steps")
    print("-" * 60)
    _print_metrics(d, [
        "median_final_equity", "p5_final_equity", "p95_final_equity",
        "probability_of_loss", "probability_of_ruin",
        "median_max_drawdown", "p95_max_drawdown",
        "median_annualized_sharpe", "p5_annualized_sharpe",
        "median_annualized_cagr", "median_calmar", "p5_calmar",
    ])

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(d), indent=2))
        print(f"\nMetrics JSON: {out}")
    return 0


# --------------------------------------------------------------------------- #
# btf dashboard
# --------------------------------------------------------------------------- #

def cmd_dashboard(args) -> int:
    app = REPO_ROOT / "apps" / "modelling_dashboard.py"
    if not app.exists():
        print(f"error: dashboard app not found: {app}", file=sys.stderr)
        return 2
    env = os.environ.copy()
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.is_dir():
            print(f"error: run directory not found: {run_dir}", file=sys.stderr)
            return 2
        env["MODEL_RUN_DIR"] = str(run_dir)
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app)], env=env)
