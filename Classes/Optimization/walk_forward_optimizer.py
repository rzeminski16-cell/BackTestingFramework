"""
Walk-Forward Optimization Engine with Bayesian Optimization

This module implements walk-forward optimization to prevent overfitting:
1. Divide data into rolling training/testing windows
2. Optimize parameters on in-sample (training) data using Bayesian optimization
3. Test optimized parameters on out-of-sample (testing) data
4. Roll forward and repeat

This approach ensures parameters are validated on truly unseen data.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import yaml
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Config.config import BacktestConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Models.trade import Trade
from Classes.Strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Constraints that must be satisfied during optimization."""
    min_profit_factor: float = 1.5
    max_drawdown_percent: float = 25.0
    min_trades_per_year: int = 1


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Best parameters found in training
    best_params: Dict[str, Any]

    # In-sample (training) metrics
    in_sample_sortino: float
    in_sample_sharpe: float
    in_sample_profit_factor: float
    in_sample_max_drawdown_pct: float
    in_sample_num_trades: int
    in_sample_total_return_pct: float
    in_sample_calmar: float

    # Out-of-sample (testing) metrics
    out_sample_sortino: float
    out_sample_sharpe: float
    out_sample_profit_factor: float
    out_sample_max_drawdown_pct: float
    out_sample_num_trades: int
    out_sample_total_return_pct: float
    out_sample_calmar: float

    # Overfitting indicators
    sortino_degradation_pct: float
    sharpe_degradation_pct: float

    # Full backtest results
    in_sample_trades: List[Trade] = field(default_factory=list)
    out_sample_trades: List[Trade] = field(default_factory=list)


@dataclass
class WalkForwardResults:
    """Complete results from walk-forward optimization."""
    strategy_name: str
    symbol: str
    windows: List[WindowResult]

    # Aggregate statistics
    avg_in_sample_sortino: float
    avg_out_sample_sortino: float
    avg_sortino_degradation_pct: float

    avg_in_sample_sharpe: float
    avg_out_sample_sharpe: float
    avg_sharpe_degradation_pct: float

    # Parameter stability analysis
    parameter_ranges: Dict[str, Tuple[Any, Any]]  # min, max across all windows
    parameter_std: Dict[str, float]  # standard deviation of parameter values

    # Most frequent optimal parameters
    most_common_params: Dict[str, Any]

    # Success metrics
    windows_passed_constraints: int
    total_windows: int


class WalkForwardOptimizer:
    """
    Walk-forward optimizer with Bayesian optimization.

    Prevents overfitting by:
    1. Optimizing on in-sample data only
    2. Testing on truly out-of-sample data
    3. Rolling forward through time
    4. Analyzing parameter stability across windows
    """

    def __init__(self, config_path: str = "config/optimization_config.yaml"):
        """Initialize the walk-forward optimizer."""
        self.config = self._load_config(config_path)
        self.backtest_config = BacktestConfig()
        self.engine = SingleSecurityEngine(config=self.backtest_config)

        # For cancellation support
        self.should_cancel = False

        logger.info("Walk-forward optimizer initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load optimization configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded optimization config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def cancel(self):
        """Cancel the optimization process."""
        self.should_cancel = True
        logger.info("Optimization cancellation requested")

    def _split_into_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into rolling training/testing windows.

        Returns:
            List of (train_data, test_data) tuples
        """
        wf_config = self.config['walk_forward']
        train_days = wf_config['training_period_days']
        test_days = wf_config['testing_period_days']
        step_min = wf_config['step_size_min_days']
        step_max = wf_config['step_size_max_days']

        windows = []
        data = data.sort_values('date').reset_index(drop=True)

        # Start from the beginning
        current_start_idx = 0

        while True:
            if self.should_cancel:
                break

            # Define training window
            train_start_date = data.iloc[current_start_idx]['date']
            train_end_date = train_start_date + timedelta(days=train_days)

            # Define testing window
            test_start_date = train_end_date
            test_end_date = test_start_date + timedelta(days=test_days)

            # Extract training data
            train_mask = (data['date'] >= train_start_date) & (data['date'] < train_end_date)
            train_data = data[train_mask].copy()

            # Extract testing data
            test_mask = (data['date'] >= test_start_date) & (data['date'] < test_end_date)
            test_data = data[test_mask].copy()

            # Check if we have enough data
            if len(train_data) < 100 or len(test_data) < 20:
                break

            windows.append((train_data, test_data))

            # Roll forward by random step size
            step_days = random.randint(step_min, step_max)
            step_indices = len(data[(data['date'] >= train_start_date) &
                                    (data['date'] < train_start_date + timedelta(days=step_days))])

            current_start_idx += max(1, step_indices)

            # Check if we've reached the end
            if current_start_idx >= len(data) - (train_days + test_days) / 365 * 252:
                break

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def _get_parameter_space(self, strategy_class: Type[BaseStrategy],
                             selected_params: Optional[Dict[str, bool]] = None) -> Tuple[List, List[str], Dict[str, Any]]:
        """
        Get parameter space for Bayesian optimization.

        Args:
            strategy_class: Strategy class
            selected_params: Dict of {param_name: bool} indicating which to optimize

        Returns:
            (space, param_names, fixed_params) where:
            - space is a list of skopt dimensions for parameters to optimize
            - param_names are names of parameters to optimize
            - fixed_params are default values for parameters not being optimized
        """
        strategy_name = strategy_class.__name__
        param_config = self.config['strategy_parameters'].get(strategy_name, {})

        if not param_config:
            raise ValueError(f"No parameter configuration found for {strategy_name}")

        # Get default values from strategy
        try:
            default_strategy = strategy_class()
            default_params = {
                param: getattr(default_strategy, param, None)
                for param in param_config.keys()
            }
        except:
            # If can't instantiate, use config mins or medians as defaults
            default_params = {}
            for param_name, param_spec in param_config.items():
                if 'values' in param_spec:
                    default_params[param_name] = param_spec['values'][0]
                else:
                    min_val = param_spec['min']
                    max_val = param_spec['max']
                    default_params[param_name] = (min_val + max_val) / 2

        space = []
        param_names = []
        fixed_params = {}

        for param_name, param_spec in param_config.items():
            # Check if this parameter should be optimized
            should_optimize = True
            if selected_params is not None:
                should_optimize = selected_params.get(param_name, True)

            if not should_optimize:
                # Fixed parameter - use default value
                fixed_params[param_name] = default_params.get(param_name)
                logger.info(f"Fixed parameter: {param_name} = {fixed_params[param_name]}")
                continue

            # Parameter to optimize
            param_type = param_spec['type']

            if 'values' in param_spec:
                # Discrete values
                values = param_spec['values']
                if param_type == 'int':
                    space.append(Integer(min(values), max(values), name=param_name))
                else:
                    space.append(Real(min(values), max(values), name=param_name))
            else:
                # Range with min/max
                min_val = param_spec['min']
                max_val = param_spec['max']

                if param_type == 'int':
                    space.append(Integer(min_val, max_val, name=param_name))
                else:
                    space.append(Real(min_val, max_val, name=param_name))

            param_names.append(param_name)

        logger.info(f"Optimizing {len(param_names)} parameters, fixing {len(fixed_params)} parameters")
        return space, param_names, fixed_params

    def _check_constraints(self, metrics: Dict[str, Any],
                          constraints: OptimizationConstraints,
                          num_years: float) -> bool:
        """Check if backtest results satisfy constraints."""
        # Check profit factor
        profit_factor = metrics.get('profit_factor', 0.0)
        if profit_factor < constraints.min_profit_factor:
            return False

        # Check max drawdown
        max_dd_pct = abs(metrics.get('max_drawdown_pct', 100.0))
        if max_dd_pct > constraints.max_drawdown_percent:
            return False

        # Check minimum trades per year
        num_trades = metrics.get('num_trades', 0)
        trades_per_year = num_trades / num_years if num_years > 0 else 0
        if trades_per_year < constraints.min_trades_per_year:
            return False

        return True

    def _evaluate_parameters(self, strategy_class: Type[BaseStrategy],
                            params: Dict[str, Any],
                            data: pd.DataFrame,
                            symbol: str,
                            constraints: OptimizationConstraints) -> float:
        """
        Evaluate a set of parameters on given data.

        Returns:
            Objective value (Sortino ratio, or penalty if constraints violated)
        """
        try:
            # Create strategy with parameters
            strategy = strategy_class(**params)

            # Run backtest
            result = self.engine.run(
                symbol=symbol,
                data=data,
                strategy=strategy
            )

            # Calculate metrics
            metrics = PerformanceMetrics.calculate_metrics(result)

            # Check constraints
            num_years = (data['date'].max() - data['date'].min()).days / 365.25
            if not self._check_constraints(metrics, constraints, num_years):
                return self.config['optimization']['constraint_violation_penalty']

            # Return primary metric (Sortino ratio)
            sortino = metrics.get('sortino_ratio', -999999.0)

            # Handle invalid values
            if np.isnan(sortino) or np.isinf(sortino):
                return self.config['optimization']['constraint_violation_penalty']

            return sortino

        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return self.config['optimization']['constraint_violation_penalty']

    def _optimize_window(self, strategy_class: Type[BaseStrategy],
                        train_data: pd.DataFrame,
                        symbol: str,
                        constraints: OptimizationConstraints,
                        selected_params: Optional[Dict[str, bool]] = None,
                        progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Optimize parameters on a single training window using Bayesian optimization.

        Args:
            strategy_class: Strategy class
            train_data: Training data
            symbol: Security symbol
            constraints: Optimization constraints
            selected_params: Which parameters to optimize (None = all)
            progress_callback: Progress callback

        Returns:
            Best parameters found (includes both optimized and fixed parameters)
        """
        space, param_names, fixed_params = self._get_parameter_space(strategy_class, selected_params)

        bayesian_config = self.config['bayesian_optimization']

        # Apply speed mode
        speed_mode = bayesian_config.get('speed_mode', 'full')
        if speed_mode == 'quick':
            n_iterations = 25
            n_initial = 10
        elif speed_mode == 'fast':
            n_iterations = 50
            n_initial = 15
        else:  # full
            n_iterations = bayesian_config['n_iterations']
            n_initial = bayesian_config['n_initial_points']

        n_jobs = bayesian_config.get('n_jobs', 1)

        # Track progress
        iteration_counter = [0]

        @use_named_args(space)
        def objective(**params):
            """Objective function for Bayesian optimization."""
            if self.should_cancel:
                return 0.0

            iteration_counter[0] += 1
            if progress_callback:
                progress_callback(iteration_counter[0], n_iterations)

            # Merge optimized params with fixed params
            full_params = {**fixed_params, **params}

            # Evaluate parameters (return negative because gp_minimize minimizes)
            score = self._evaluate_parameters(
                strategy_class, full_params, train_data, symbol, constraints
            )
            return -score  # Negative because we want to maximize Sortino

        # Run Bayesian optimization with parallel jobs
        try:
            result = gp_minimize(
                objective,
                space,
                n_calls=n_iterations,
                n_initial_points=n_initial,
                n_jobs=n_jobs,
                random_state=bayesian_config.get('random_state'),
                verbose=False
            )
        except (ImportError, OSError, AttributeError) as e:
            # Multiprocessing not available in this environment
            if n_jobs != 1:
                logger.warning(f"Parallel processing not available ({e}). Falling back to serial processing (n_jobs=1)")
                # Retry with serial processing
                result = gp_minimize(
                    objective,
                    space,
                    n_calls=n_iterations,
                    n_initial_points=n_initial,
                    n_jobs=1,
                    random_state=bayesian_config.get('random_state'),
                    verbose=False
                )
            else:
                raise

        # Extract best parameters and merge with fixed parameters
        best_params = {name: value for name, value in zip(param_names, result.x)}
        best_params.update(fixed_params)

        logger.info(f"Optimization complete. Best parameters: {best_params}")
        return best_params

    def _backtest_with_params(self, strategy_class: Type[BaseStrategy],
                             params: Dict[str, Any],
                             data: pd.DataFrame,
                             symbol: str) -> Dict[str, Any]:
        """Run backtest with given parameters and return metrics."""
        strategy = strategy_class(**params)
        result = self.engine.run(symbol, data, strategy)
        metrics = PerformanceMetrics.calculate_metrics(result)
        metrics['trades'] = result.trades
        return metrics

    def optimize(self,
                strategy_class: Type[BaseStrategy],
                symbol: str,
                data: pd.DataFrame,
                selected_params: Optional[Dict[str, bool]] = None,
                progress_callback: Optional[Callable[[str, int, int], None]] = None) -> WalkForwardResults:
        """
        Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            symbol: Security symbol
            data: Historical price data
            selected_params: Dict of {param_name: bool} indicating which parameters to optimize
            progress_callback: Callback(stage_name, current, total)

        Returns:
            WalkForwardResults with complete optimization results
        """
        logger.info(f"Starting walk-forward optimization for {strategy_class.__name__} on {symbol}")

        # Reset cancellation flag
        self.should_cancel = False

        # Load constraints
        constraint_config = self.config['optimization']['constraints']
        constraints = OptimizationConstraints(
            min_profit_factor=constraint_config['min_profit_factor'],
            max_drawdown_percent=constraint_config['max_drawdown_percent'],
            min_trades_per_year=constraint_config['min_trades_per_year']
        )

        # Split data into windows
        if progress_callback:
            progress_callback("Splitting data into windows", 0, 1)

        windows = self._split_into_windows(data)

        if len(windows) == 0:
            raise ValueError("Not enough data to create walk-forward windows")

        # Optimize each window
        window_results = []
        # Get all parameter names (both optimized and fixed) for tracking
        _, param_names, fixed_params = self._get_parameter_space(strategy_class, selected_params)
        all_param_names = list(param_names) + list(fixed_params.keys())
        all_params = {param: [] for param in all_param_names}

        for window_id, (train_data, test_data) in enumerate(windows):
            if self.should_cancel:
                logger.info("Optimization cancelled by user")
                break

            logger.info(f"Processing window {window_id + 1}/{len(windows)}")

            if progress_callback:
                progress_callback(f"Optimizing window {window_id + 1}/{len(windows)}",
                                window_id, len(windows))

            try:
                # Optimize on training data
                def window_progress(current, total):
                    if progress_callback:
                        progress_callback(
                            f"Window {window_id + 1}/{len(windows)} - Iteration {current}/{total}",
                            current, total
                        )

                best_params = self._optimize_window(
                    strategy_class, train_data, symbol, constraints, selected_params, window_progress
                )
            except Exception as e:
                logger.error(f"Error optimizing window {window_id + 1}: {e}")
                logger.exception("Window optimization failed")
                # Skip this window and continue with the next one
                continue

            # Track parameter values
            for param_name, param_value in best_params.items():
                all_params[param_name].append(param_value)

            # Evaluate on training data (in-sample)
            in_sample_metrics = self._backtest_with_params(
                strategy_class, best_params, train_data, symbol
            )

            # Evaluate on testing data (out-of-sample)
            out_sample_metrics = self._backtest_with_params(
                strategy_class, best_params, test_data, symbol
            )

            # Calculate degradation
            in_sortino = in_sample_metrics.get('sortino_ratio', 0.0)
            out_sortino = out_sample_metrics.get('sortino_ratio', 0.0)
            sortino_degradation = ((in_sortino - out_sortino) / abs(in_sortino) * 100
                                  if in_sortino != 0 else 0.0)

            in_sharpe = in_sample_metrics.get('sharpe_ratio', 0.0)
            out_sharpe = out_sample_metrics.get('sharpe_ratio', 0.0)
            sharpe_degradation = ((in_sharpe - out_sharpe) / abs(in_sharpe) * 100
                                 if in_sharpe != 0 else 0.0)

            # Create window result
            window_result = WindowResult(
                window_id=window_id,
                train_start=train_data['date'].min(),
                train_end=train_data['date'].max(),
                test_start=test_data['date'].min(),
                test_end=test_data['date'].max(),
                best_params=best_params,
                in_sample_sortino=in_sortino,
                in_sample_sharpe=in_sharpe,
                in_sample_profit_factor=in_sample_metrics.get('profit_factor', 0.0),
                in_sample_max_drawdown_pct=abs(in_sample_metrics.get('max_drawdown_pct', 0.0)),
                in_sample_num_trades=in_sample_metrics.get('num_trades', 0),
                in_sample_total_return_pct=in_sample_metrics.get('total_return_pct', 0.0),
                in_sample_calmar=in_sample_metrics.get('calmar_ratio', 0.0),
                out_sample_sortino=out_sortino,
                out_sample_sharpe=out_sharpe,
                out_sample_profit_factor=out_sample_metrics.get('profit_factor', 0.0),
                out_sample_max_drawdown_pct=abs(out_sample_metrics.get('max_drawdown_pct', 0.0)),
                out_sample_num_trades=out_sample_metrics.get('num_trades', 0),
                out_sample_total_return_pct=out_sample_metrics.get('total_return_pct', 0.0),
                out_sample_calmar=out_sample_metrics.get('calmar_ratio', 0.0),
                sortino_degradation_pct=sortino_degradation,
                sharpe_degradation_pct=sharpe_degradation,
                in_sample_trades=in_sample_metrics.get('trades', []),
                out_sample_trades=out_sample_metrics.get('trades', [])
            )

            window_results.append(window_result)

        # Calculate aggregate statistics
        if len(window_results) == 0:
            raise ValueError("No windows were successfully processed")

        avg_in_sortino = np.mean([w.in_sample_sortino for w in window_results])
        avg_out_sortino = np.mean([w.out_sample_sortino for w in window_results])
        avg_sortino_degradation = np.mean([w.sortino_degradation_pct for w in window_results])

        avg_in_sharpe = np.mean([w.in_sample_sharpe for w in window_results])
        avg_out_sharpe = np.mean([w.out_sample_sharpe for w in window_results])
        avg_sharpe_degradation = np.mean([w.sharpe_degradation_pct for w in window_results])

        # Analyze parameter stability
        parameter_ranges = {}
        parameter_std = {}
        most_common_params = {}

        for param_name, values in all_params.items():
            if len(values) > 0:
                parameter_ranges[param_name] = (min(values), max(values))
                parameter_std[param_name] = np.std(values)
                # For most common, use median for continuous, mode for discrete
                most_common_params[param_name] = np.median(values)

        # Count windows that passed constraints
        windows_passed = sum(1 for w in window_results
                           if w.out_sample_profit_factor >= constraints.min_profit_factor
                           and w.out_sample_max_drawdown_pct <= constraints.max_drawdown_percent)

        results = WalkForwardResults(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            windows=window_results,
            avg_in_sample_sortino=avg_in_sortino,
            avg_out_sample_sortino=avg_out_sortino,
            avg_sortino_degradation_pct=avg_sortino_degradation,
            avg_in_sample_sharpe=avg_in_sharpe,
            avg_out_sample_sharpe=avg_out_sharpe,
            avg_sharpe_degradation_pct=avg_sharpe_degradation,
            parameter_ranges=parameter_ranges,
            parameter_std=parameter_std,
            most_common_params=most_common_params,
            windows_passed_constraints=windows_passed,
            total_windows=len(window_results)
        )

        logger.info(f"Walk-forward optimization complete. {len(window_results)} windows processed.")
        return results
