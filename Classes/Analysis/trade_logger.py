"""
Trade logging to CSV files with organized folder structure.

Folder structure:
logs/
├── backtests/
│   ├── single_security/
│   │   └── {backtest_name}/
│   │       ├── {strategy}_{symbol}_trades.csv
│   │       ├── {strategy}_{symbol}_parameters.json
│   │       └── reports/
│   │           └── {timestamp}.xlsx
│   └── portfolio/
│       └── {backtest_name}/
│           ├── trades/
│           │   └── {symbol}_trades.csv (per security)
│           ├── portfolio_trades.csv (consolidated)
│           ├── signal_rejections.csv
│           ├── vulnerability_log.csv (if using vulnerability score)
│           ├── config.json
│           └── reports/
│               └── portfolio_report_{timestamp}.xlsx
├── optimizations/
│   ├── single_security/
│   │   └── {optimization_name}/
│   │       └── {strategy}_{symbol}/
│   └── portfolio/
│       └── {optimization_name}/
│           └── {basket_name}/
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from ..Models.trade import Trade


class LoggingPath:
    """Manages logging paths for the backtesting framework."""

    BASE_DIR = Path("logs")

    @classmethod
    def get_single_security_backtest_dir(cls, backtest_name: str) -> Path:
        """Get directory for single security backtest."""
        path = cls.BASE_DIR / "backtests" / "single_security" / backtest_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_portfolio_backtest_dir(cls, backtest_name: str) -> Path:
        """Get directory for portfolio backtest."""
        path = cls.BASE_DIR / "backtests" / "portfolio" / backtest_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_single_security_optimization_dir(cls, optimization_name: str,
                                              strategy_name: str, symbol: str) -> Path:
        """Get directory for single security optimization."""
        path = cls.BASE_DIR / "optimizations" / "single_security" / optimization_name / f"{strategy_name}_{symbol}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_portfolio_optimization_dir(cls, optimization_name: str, basket_name: str) -> Path:
        """Get directory for portfolio optimization."""
        path = cls.BASE_DIR / "optimizations" / "portfolio" / optimization_name / basket_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def ensure_reports_dir(cls, base_path: Path) -> Path:
        """Ensure reports subdirectory exists."""
        reports_path = base_path / "reports"
        reports_path.mkdir(parents=True, exist_ok=True)
        return reports_path

    @classmethod
    def ensure_trades_dir(cls, base_path: Path) -> Path:
        """Ensure trades subdirectory exists (for portfolio)."""
        trades_path = base_path / "trades"
        trades_path.mkdir(parents=True, exist_ok=True)
        return trades_path


class TradeLogger:
    """
    Logs trades to CSV files for detailed analysis.

    Supports both single-security and portfolio backtesting with
    organized folder structure.
    """

    def __init__(self, output_directory: Path):
        """
        Initialize trade logger.

        Args:
            output_directory: Directory to save trade logs
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def log_trades(self, symbol: str, strategy_name: str, trades: List[Trade],
                   strategy_params: Optional[Dict[str, Any]] = None) -> Path:
        """
        Log trades to CSV file.

        Args:
            symbol: Security symbol
            strategy_name: Strategy name
            trades: List of trades
            strategy_params: Optional dictionary of strategy parameters

        Returns:
            Path to created CSV file
        """
        if not trades:
            print(f"No trades to log for {symbol}")
            return None

        # Convert trades to list of dicts
        trade_dicts = [trade.to_dict() for trade in trades]

        # Create DataFrame
        df = pd.DataFrame(trade_dicts)

        # Create filename
        filename = f"{strategy_name}_{symbol}_trades.csv"
        filepath = self.output_directory / filename

        # Save to CSV
        df.to_csv(filepath, index=False)

        print(f"Logged {len(trades)} trades to {filepath}")

        # Save strategy parameters if provided
        if strategy_params:
            self.log_strategy_params(symbol, strategy_name, strategy_params)

        return filepath

    def log_strategy_params(self, symbol: str, strategy_name: str,
                           strategy_params: Dict[str, Any]) -> Path:
        """
        Log strategy parameters to JSON file.

        Args:
            symbol: Security symbol
            strategy_name: Strategy name
            strategy_params: Dictionary of strategy parameters

        Returns:
            Path to created JSON file
        """
        # Create filename
        filename = f"{strategy_name}_{symbol}_parameters.json"
        filepath = self.output_directory / filename

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(strategy_params, f, indent=2)

        print(f"Logged strategy parameters to {filepath}")
        return filepath

    def log_multiple(self, trades_dict: dict, strategy_name: str) -> dict:
        """
        Log trades for multiple securities.

        Args:
            trades_dict: Dictionary mapping symbol to trade list
            strategy_name: Strategy name

        Returns:
            Dictionary mapping symbol to filepath
        """
        filepaths = {}

        for symbol, trades in trades_dict.items():
            filepath = self.log_trades(symbol, strategy_name, trades)
            if filepath:
                filepaths[symbol] = filepath

        return filepaths


class PortfolioTradeLogger:
    """
    Specialized logger for portfolio backtesting with enhanced tracking.

    Logs:
    - Individual trades per symbol
    - Consolidated portfolio trades
    - Signal rejections
    - Vulnerability score history (if enabled)
    - Configuration used
    """

    def __init__(self, backtest_name: str, basket_name: Optional[str] = None):
        """
        Initialize portfolio trade logger.

        Args:
            backtest_name: Name of the backtest run
            basket_name: Optional basket name (for reference)
        """
        self.backtest_name = backtest_name
        self.basket_name = basket_name
        self.base_dir = LoggingPath.get_portfolio_backtest_dir(backtest_name)
        self.trades_dir = LoggingPath.ensure_trades_dir(self.base_dir)
        self.reports_dir = LoggingPath.ensure_reports_dir(self.base_dir)

    def log_portfolio_result(self, result, strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Log complete portfolio backtest result.

        Args:
            result: PortfolioBacktestResult object
            strategy_params: Optional strategy parameters

        Returns:
            Dict of logged file paths
        """
        logged_files = {}

        # Log trades per symbol
        for symbol, symbol_result in result.symbol_results.items():
            if symbol_result.trades:
                filepath = self._log_symbol_trades(symbol, symbol_result.trades)
                logged_files[f"trades_{symbol}"] = filepath

        # Log consolidated portfolio trades
        all_trades = []
        for symbol_result in result.symbol_results.values():
            all_trades.extend(symbol_result.trades)
        all_trades.sort(key=lambda t: t.entry_date)

        if all_trades:
            filepath = self._log_consolidated_trades(all_trades)
            logged_files["portfolio_trades"] = filepath

        # Log signal rejections
        if result.signal_rejections:
            filepath = self._log_signal_rejections(result.signal_rejections)
            logged_files["signal_rejections"] = filepath

        # Log vulnerability swaps and history
        if result.vulnerability_swaps:
            filepath = self._log_vulnerability_swaps(result.vulnerability_swaps)
            logged_files["vulnerability_swaps"] = filepath

        if result.vulnerability_history:
            filepath = self._log_vulnerability_history(result.vulnerability_history)
            logged_files["vulnerability_history"] = filepath

        # Log configuration
        filepath = self._log_config(result.config, strategy_params)
        logged_files["config"] = filepath

        print(f"Portfolio backtest logged to {self.base_dir}")
        return logged_files

    def _log_symbol_trades(self, symbol: str, trades: List[Trade]) -> Path:
        """Log trades for a single symbol."""
        trade_dicts = [trade.to_dict() for trade in trades]
        df = pd.DataFrame(trade_dicts)
        filepath = self.trades_dir / f"{symbol}_trades.csv"
        df.to_csv(filepath, index=False)
        return filepath

    def _log_consolidated_trades(self, trades: List[Trade]) -> Path:
        """Log all portfolio trades in one file."""
        trade_dicts = [trade.to_dict() for trade in trades]
        df = pd.DataFrame(trade_dicts)
        filepath = self.base_dir / "portfolio_trades.csv"
        df.to_csv(filepath, index=False)
        return filepath

    def _log_signal_rejections(self, rejections: List) -> Path:
        """Log signal rejections."""
        rejection_dicts = []
        for r in rejections:
            rejection_dict = {
                'date': r.date,
                'symbol': r.symbol,
                'signal_type': r.signal_type,
                'reason': r.reason,
                'available_capital': r.available_capital,
                'required_capital': r.required_capital
            }
            rejection_dicts.append(rejection_dict)

        df = pd.DataFrame(rejection_dicts)
        filepath = self.base_dir / "signal_rejections.csv"
        df.to_csv(filepath, index=False)
        print(f"Logged {len(rejections)} signal rejections")
        return filepath

    def _log_vulnerability_swaps(self, swaps: List) -> Path:
        """Log vulnerability score swaps."""
        swap_dicts = []
        for s in swaps:
            swap_dict = {
                'date': s.date,
                'closed_symbol': s.closed_symbol,
                'closed_score': s.closed_score,
                'new_symbol': s.new_symbol
            }
            swap_dicts.append(swap_dict)

        df = pd.DataFrame(swap_dicts)
        filepath = self.base_dir / "vulnerability_swaps.csv"
        df.to_csv(filepath, index=False)
        print(f"Logged {len(swaps)} vulnerability swaps")
        return filepath

    def _log_vulnerability_history(self, history: List[Dict]) -> Path:
        """Log daily vulnerability scores for all positions."""
        rows = []
        for day_idx, day_scores in enumerate(history):
            for symbol, result in day_scores.items():
                rows.append({
                    'day_index': day_idx,
                    'symbol': result.symbol,
                    'score': result.score,
                    'days_open': result.days_open,
                    'current_pl_percent': result.current_pl_percent,
                    'is_immune': result.is_immune,
                    'is_vulnerable': result.is_vulnerable,
                    'decay_type': result.decay_type
                })

        if rows:
            df = pd.DataFrame(rows)
            filepath = self.base_dir / "vulnerability_history.csv"
            df.to_csv(filepath, index=False)
            return filepath
        return None

    def _log_config(self, config, strategy_params: Optional[Dict[str, Any]] = None) -> Path:
        """Log configuration used for the backtest."""
        config_dict = {
            'backtest_name': self.backtest_name,
            'basket_name': self.basket_name,
            'portfolio_config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
            'strategy_params': strategy_params or {},
            'timestamp': datetime.now().isoformat()
        }

        filepath = self.base_dir / "config.json"
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        return filepath

    def get_reports_dir(self) -> Path:
        """Get the reports directory for this backtest."""
        return self.reports_dir


class OptimizationLogger:
    """Logger for optimization results."""

    def __init__(self, optimization_name: str, is_portfolio: bool = False,
                 basket_name: Optional[str] = None):
        """
        Initialize optimization logger.

        Args:
            optimization_name: Name of the optimization run
            is_portfolio: Whether this is portfolio optimization
            basket_name: Basket name (required if is_portfolio is True)
        """
        self.optimization_name = optimization_name
        self.is_portfolio = is_portfolio

        if is_portfolio:
            if not basket_name:
                raise ValueError("basket_name required for portfolio optimization")
            self.base_dir = LoggingPath.get_portfolio_optimization_dir(
                optimization_name, basket_name
            )
        else:
            # For single security, base_dir set per symbol
            self.base_dir = None

    def get_symbol_dir(self, strategy_name: str, symbol: str) -> Path:
        """Get directory for single-security optimization."""
        return LoggingPath.get_single_security_optimization_dir(
            self.optimization_name, strategy_name, symbol
        )

    def log_optimization_result(self, result, strategy_name: str,
                                symbol: Optional[str] = None) -> Path:
        """
        Log optimization result.

        Args:
            result: Optimization result object
            strategy_name: Strategy name
            symbol: Symbol (for single-security optimization)

        Returns:
            Path to logged directory
        """
        if self.is_portfolio:
            output_dir = self.base_dir
        else:
            if not symbol:
                raise ValueError("symbol required for single-security optimization")
            output_dir = self.get_symbol_dir(strategy_name, symbol)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Log summary
        summary = {
            'optimization_name': self.optimization_name,
            'strategy_name': strategy_name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }

        # Add result-specific data if available
        if hasattr(result, 'most_common_params'):
            summary['recommended_params'] = result.most_common_params
        if hasattr(result, 'avg_out_sample_sortino'):
            summary['avg_oos_sortino'] = result.avg_out_sample_sortino
        if hasattr(result, 'windows_passed_constraints'):
            summary['windows_passed'] = result.windows_passed_constraints
            summary['total_windows'] = result.total_windows

        filepath = output_dir / "optimization_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Optimization logged to {output_dir}")
        return output_dir
