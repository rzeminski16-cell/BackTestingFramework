"""
Walk-Forward Optimization Engine with Bayesian Optimization

This module implements walk-forward optimization to prevent overfitting:
1. Divide data into rolling or anchored training/testing windows
2. Optimize parameters on in-sample (training) data using Bayesian optimization
3. Test optimized parameters on out-of-sample (testing) data
4. Roll/expand forward and repeat

Walk-Forward Modes:
- ROLLING: Both training start and end dates roll forward (good for frequent trading)
- ANCHORED: Training always starts from the same date, only end expands (good for low-frequency trading)

This approach ensures parameters are validated on truly unseen data.

NOTE: All metric calculations are delegated to CentralizedPerformanceMetrics
(through the Classes.Analysis.performance_metrics wrapper) to ensure consistency
across the framework with standardized risk-free rate (3.5%) and trading days (252).
"""

import logging
import os
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import yaml
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# PerformanceMetrics now delegates to CentralizedPerformanceMetrics
from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Config.config import BacktestConfig, PortfolioConfig, CommissionConfig
from Classes.Config.capital_contention import CapitalContentionConfig, CapitalContentionMode, VulnerabilityScoreConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine, PortfolioBacktestResult
from Classes.Models.trade import Trade
from Classes.Strategy.base_strategy import BaseStrategy

# Import centralized strategy configuration
from config.strategy_config import StrategyConfig

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward optimization mode."""
    ROLLING = "rolling"  # Both train start and end roll forward (sliding window)
    ANCHORED = "anchored"  # Train start fixed, end expands forward (expanding window)


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


@dataclass
class MultiSecurityResults:
    """
    Combined results from walk-forward optimization across multiple securities.

    This class aggregates individual security results to provide:
    1. Per-security performance metrics
    2. Combined/average metrics across all securities
    3. Parameter consistency analysis (which parameters work well across all)
    4. Robustness assessment across different market instruments
    """
    strategy_name: str
    securities: List[str]
    individual_results: Dict[str, WalkForwardResults]  # symbol -> results

    # Aggregate metrics (averaged across all securities)
    combined_avg_in_sample_sortino: float
    combined_avg_out_sample_sortino: float
    combined_avg_sortino_degradation_pct: float

    combined_avg_in_sample_sharpe: float
    combined_avg_out_sample_sharpe: float
    combined_avg_sharpe_degradation_pct: float

    # Parameter consistency analysis
    # Parameters that performed well across ALL securities
    consistent_params: Dict[str, Any]  # Median of medians
    param_consistency_scores: Dict[str, float]  # 0-100, how consistent across securities

    # Overall success metrics
    total_windows_all_securities: int
    total_passed_all_securities: int
    securities_with_positive_oos: int  # Securities where OOS Sortino > 0

    # Best performing security and worst
    best_security: str
    worst_security: str

    @staticmethod
    def from_individual_results(
        strategy_name: str,
        results_dict: Dict[str, WalkForwardResults]
    ) -> 'MultiSecurityResults':
        """
        Create MultiSecurityResults from individual security results.

        Args:
            strategy_name: Name of the strategy
            results_dict: Dict mapping symbol to WalkForwardResults

        Returns:
            MultiSecurityResults with aggregated metrics
        """
        if not results_dict:
            raise ValueError("No results provided")

        securities = list(results_dict.keys())

        # Calculate combined averages
        all_in_sortinos = [r.avg_in_sample_sortino for r in results_dict.values()]
        all_out_sortinos = [r.avg_out_sample_sortino for r in results_dict.values()]
        all_sortino_deg = [r.avg_sortino_degradation_pct for r in results_dict.values()]

        all_in_sharpes = [r.avg_in_sample_sharpe for r in results_dict.values()]
        all_out_sharpes = [r.avg_out_sample_sharpe for r in results_dict.values()]
        all_sharpe_deg = [r.avg_sharpe_degradation_pct for r in results_dict.values()]

        # Total windows
        total_windows = sum(r.total_windows for r in results_dict.values())
        total_passed = sum(r.windows_passed_constraints for r in results_dict.values())

        # Securities with positive OOS performance
        positive_oos = sum(1 for r in results_dict.values() if r.avg_out_sample_sortino > 0)

        # Find best and worst securities by OOS Sortino
        best_sec = max(results_dict.keys(), key=lambda s: results_dict[s].avg_out_sample_sortino)
        worst_sec = min(results_dict.keys(), key=lambda s: results_dict[s].avg_out_sample_sortino)

        # Calculate parameter consistency
        # Gather all parameter names from first result
        first_result = next(iter(results_dict.values()))
        param_names = list(first_result.most_common_params.keys())

        # For each parameter, collect values across all securities
        param_values_by_security = {param: [] for param in param_names}
        for symbol, result in results_dict.items():
            for param, value in result.most_common_params.items():
                param_values_by_security[param].append(value)

        # Calculate consistent params (median of medians) and consistency scores
        consistent_params = {}
        param_consistency_scores = {}
        for param, values in param_values_by_security.items():
            if len(values) > 0:
                consistent_params[param] = float(np.median(values))
                # Consistency score: 100 - (coefficient of variation * 100), capped at 0-100
                mean_val = np.mean(values)
                if mean_val != 0:
                    cv = (np.std(values) / abs(mean_val)) * 100
                    consistency_score = max(0, min(100, 100 - cv))
                else:
                    consistency_score = 100 if np.std(values) == 0 else 0
                param_consistency_scores[param] = consistency_score

        return MultiSecurityResults(
            strategy_name=strategy_name,
            securities=securities,
            individual_results=results_dict,
            combined_avg_in_sample_sortino=float(np.mean(all_in_sortinos)),
            combined_avg_out_sample_sortino=float(np.mean(all_out_sortinos)),
            combined_avg_sortino_degradation_pct=float(np.mean(all_sortino_deg)),
            combined_avg_in_sample_sharpe=float(np.mean(all_in_sharpes)),
            combined_avg_out_sample_sharpe=float(np.mean(all_out_sharpes)),
            combined_avg_sharpe_degradation_pct=float(np.mean(all_sharpe_deg)),
            consistent_params=consistent_params,
            param_consistency_scores=param_consistency_scores,
            total_windows_all_securities=total_windows,
            total_passed_all_securities=total_passed,
            securities_with_positive_oos=positive_oos,
            best_security=best_sec,
            worst_security=worst_sec
        )


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
        """Load optimization configuration from YAML file.

        Strategy parameters are loaded from the centralized config/strategy_parameters.json,
        then any overrides from the YAML are applied on top.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded optimization config from {config_path}")

            # Merge strategy parameters from centralized config
            # YAML values override centralized config for customization
            yaml_strategy_params = config.get('strategy_parameters', {})
            merged_strategy_params = {}

            for strategy_name in StrategyConfig.get_strategies():
                # Get centralized optimization params
                central_params = StrategyConfig.get_optimization_params(strategy_name)
                yaml_params = yaml_strategy_params.get(strategy_name, {})

                # Start with centralized params
                merged_strategy_params[strategy_name] = {}
                for param_name, param_info in central_params.items():
                    merged_strategy_params[strategy_name][param_name] = {
                        'min': param_info.get('min'),
                        'max': param_info.get('max'),
                        'type': param_info.get('type'),
                        'step': param_info.get('step')
                    }

                # Override with YAML params if present
                for param_name, yaml_param_info in yaml_params.items():
                    if param_name not in merged_strategy_params[strategy_name]:
                        merged_strategy_params[strategy_name][param_name] = {}
                    merged_strategy_params[strategy_name][param_name].update(yaml_param_info)

            config['strategy_parameters'] = merged_strategy_params
            logger.info(f"Merged strategy parameters from centralized config")

            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def cancel(self):
        """Cancel the optimization process."""
        self.should_cancel = True
        logger.info("Optimization cancellation requested")

    def _split_into_windows(self, data: pd.DataFrame,
                            mode: Optional[WalkForwardMode] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into training/testing windows based on the selected mode.

        Modes:
        - ROLLING: Both training start and end dates roll forward (sliding window)
                   Good for higher-frequency trading strategies
        - ANCHORED: Training always starts from same date, only end expands
                   Good for lower-frequency trading strategies that need more data

        Args:
            data: Historical price data
            mode: Walk-forward mode (defaults to config setting or ROLLING)

        Returns:
            List of (train_data, test_data) tuples
        """
        wf_config = self.config['walk_forward']
        train_days = wf_config['training_period_days']
        test_days = wf_config['testing_period_days']
        step_min = int(wf_config['step_size_min_days'])
        step_max = int(wf_config['step_size_max_days'])

        # Get mode from config if not specified
        if mode is None:
            mode_str = wf_config.get('mode', 'rolling').lower()
            mode = WalkForwardMode(mode_str)

        windows = []
        data = data.sort_values('date').reset_index(drop=True)

        # Get data date range
        data_start_date = data.iloc[0]['date']
        data_end_date = data.iloc[-1]['date']

        # Use deterministic step based on average (for reproducibility)
        # Using a seed based on data hash would vary with data, so use fixed average
        avg_step = (step_min + step_max) // 2

        # Calculate total available days
        total_days = (data_end_date - data_start_date).days
        window_size_days = train_days + test_days

        # Log the effective parameters for debugging
        logger.info(f"Window splitting: data from {data_start_date.date() if hasattr(data_start_date, 'date') else data_start_date} "
                   f"to {data_end_date.date() if hasattr(data_end_date, 'date') else data_end_date} ({total_days} days, {len(data)} bars)")
        logger.info(f"Window params: train={train_days}d, test={test_days}d, step={avg_step}d (range {step_min}-{step_max})")

        if total_days < window_size_days:
            logger.warning(f"Not enough data ({total_days} days) for even one window ({window_size_days} days required)")
            return windows

        window_count = 0

        if mode == WalkForwardMode.ANCHORED:
            # ANCHORED MODE: Training always starts from data_start_date, expands forward
            anchor_start_date = data_start_date
            current_train_end_offset = train_days  # Days from anchor for training end

            while True:
                if self.should_cancel:
                    break

                # Training always starts from anchor, ends at offset
                train_start_date = anchor_start_date
                train_end_date = anchor_start_date + timedelta(days=current_train_end_offset)

                # Testing window follows training
                test_start_date = train_end_date
                test_end_date = test_start_date + timedelta(days=test_days)

                # Check if test window would exceed available data
                if test_end_date > data_end_date:
                    logger.debug(f"Window {window_count + 1}: Test end {test_end_date.date()} exceeds data end {data_end_date.date()}, stopping")
                    break

                # Extract training data (expands over time)
                train_mask = (data['date'] >= train_start_date) & (data['date'] < train_end_date)
                train_data = data[train_mask].copy()

                # Extract testing data
                test_mask = (data['date'] >= test_start_date) & (data['date'] < test_end_date)
                test_data = data[test_mask].copy()

                # Check if we have enough data bars (not just calendar days)
                if len(train_data) < 100:
                    logger.debug(f"Window {window_count + 1}: Training data has only {len(train_data)} bars (< 100 minimum), stopping")
                    break
                if len(test_data) < 20:
                    logger.debug(f"Window {window_count + 1}: Test data has only {len(test_data)} bars (< 20 minimum), stopping")
                    break

                windows.append((train_data, test_data))
                window_count += 1
                logger.debug(f"Created window {window_count}: Train {train_start_date.date()} to {train_end_date.date()} ({len(train_data)} bars), "
                           f"Test {test_start_date.date()} to {test_end_date.date()} ({len(test_data)} bars)")

                # Expand training window by step size
                current_train_end_offset += avg_step

        else:  # ROLLING MODE
            # ROLLING MODE: Both start and end roll forward (sliding window)
            current_train_start = data_start_date

            while True:
                if self.should_cancel:
                    break

                # Calculate window dates
                train_start_date = current_train_start
                train_end_date = train_start_date + timedelta(days=train_days)
                test_start_date = train_end_date
                test_end_date = test_start_date + timedelta(days=test_days)

                # Check if test window would exceed available data
                if test_end_date > data_end_date:
                    logger.debug(f"Window {window_count + 1}: Test end {test_end_date.date()} exceeds data end {data_end_date.date()}, stopping")
                    break

                # Extract training data
                train_mask = (data['date'] >= train_start_date) & (data['date'] < train_end_date)
                train_data = data[train_mask].copy()

                # Extract testing data
                test_mask = (data['date'] >= test_start_date) & (data['date'] < test_end_date)
                test_data = data[test_mask].copy()

                # Check if we have enough data bars
                if len(train_data) < 100:
                    logger.debug(f"Window {window_count + 1}: Training data has only {len(train_data)} bars (< 100 minimum), stopping")
                    break
                if len(test_data) < 20:
                    logger.debug(f"Window {window_count + 1}: Test data has only {len(test_data)} bars (< 20 minimum), stopping")
                    break

                windows.append((train_data, test_data))
                window_count += 1
                logger.debug(f"Created window {window_count}: Train {train_start_date.date()} to {train_end_date.date()} ({len(train_data)} bars), "
                           f"Test {test_start_date.date()} to {test_end_date.date()} ({len(test_data)} bars)")

                # Roll forward by step size (in calendar days)
                current_train_start = current_train_start + timedelta(days=avg_step)

        logger.info(f"Created {len(windows)} walk-forward windows using {mode.value.upper()} mode "
                   f"(train={train_days}d, test={test_days}d, step={avg_step}d)")
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

    def _calculate_constraint_penalty(self, metrics: Dict[str, Any],
                                       constraints: OptimizationConstraints) -> float:
        """
        Calculate a penalty value for constraint violations.

        Used in Bayesian optimization to penalize parameter sets that violate constraints.

        Args:
            metrics: Backtest metrics dictionary
            constraints: Optimization constraints

        Returns:
            Penalty value (0 if no violations, positive if violations)
        """
        penalty = 0.0

        # Check profit factor
        profit_factor = metrics.get('profit_factor', 0.0)
        if profit_factor < constraints.min_profit_factor:
            # Penalty proportional to how far below threshold
            shortfall = constraints.min_profit_factor - profit_factor
            penalty += shortfall * 10.0  # Scale factor

        # Check max drawdown
        max_dd_pct = abs(metrics.get('max_drawdown_pct', 100.0))
        if max_dd_pct > constraints.max_drawdown_percent:
            # Penalty proportional to how far above threshold
            excess = max_dd_pct - constraints.max_drawdown_percent
            penalty += excess * 0.5  # Scale factor

        # Check minimum trades
        num_trades = metrics.get('num_trades', 0)
        if num_trades < constraints.min_trades_per_year:
            # Penalty for too few trades
            shortfall = constraints.min_trades_per_year - num_trades
            penalty += shortfall * 5.0  # Scale factor

        return penalty

    def _calculate_degradation(self, in_sample: float, out_sample: float) -> float:
        """
        Calculate performance degradation from in-sample to out-of-sample.

        Positive value = OOS performance is worse than IS (degradation)
        Negative value = OOS performance is better than IS (improvement)

        Uses a robust formula that handles edge cases:
        - When IS metric is zero, returns 0
        - When both are negative, correctly shows degradation direction
        - Caps extreme values to prevent misleading percentages

        Args:
            in_sample: In-sample metric value
            out_sample: Out-of-sample metric value

        Returns:
            Degradation percentage (capped at +/- 200% for readability)
        """
        if in_sample == 0:
            # Cannot calculate percentage change from zero
            if out_sample == 0:
                return 0.0
            elif out_sample > 0:
                return -100.0  # Improvement from nothing to positive
            else:
                return 100.0  # Degradation from nothing to negative

        # Standard degradation formula
        degradation = ((in_sample - out_sample) / abs(in_sample)) * 100

        # Cap extreme values for readability (e.g., when IS is very small)
        return max(min(degradation, 200.0), -200.0)

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

        # Adjust for Windows compatibility
        if platform.system() == 'Windows' and n_jobs != 1:
            logger.info(f"Windows detected: Parallel processing (n_jobs={n_jobs}) may not be available. Will attempt and fall back to serial if needed.")
            # Note: We'll still try, but expect it to fall back

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
                # Provide platform-specific message
                if platform.system() == 'Windows':
                    logger.info(f"Parallel processing not available on Windows (expected). Using serial processing (n_jobs=1).")
                else:
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

        # Ensure proper type casting (Bayesian optimizer returns numpy types)
        strategy_name = strategy_class.__name__
        param_config = self.config['strategy_parameters'].get(strategy_name, {})
        for param_name, param_value in best_params.items():
            if param_name in param_config:
                param_type = param_config[param_name].get('type', 'float')
                if param_type == 'int':
                    best_params[param_name] = int(round(param_value))
                else:
                    best_params[param_name] = float(param_value)

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
                progress_callback: Optional[Callable[[str, int, int], None]] = None,
                walk_forward_mode: Optional[WalkForwardMode] = None) -> WalkForwardResults:
        """
        Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            symbol: Security symbol
            data: Historical price data
            selected_params: Dict of {param_name: bool} indicating which parameters to optimize
            progress_callback: Callback(stage_name, current, total)
            walk_forward_mode: Walk-forward mode (ROLLING or ANCHORED). If None, uses config.

        Returns:
            WalkForwardResults with complete optimization results
        """
        mode_desc = walk_forward_mode.value if walk_forward_mode else "config default"
        logger.info(f"Starting walk-forward optimization for {strategy_class.__name__} on {symbol} (mode: {mode_desc})")

        # Reset cancellation flag
        self.should_cancel = False

        # Load constraints
        constraint_config = self.config['optimization']['constraints']
        constraints = OptimizationConstraints(
            min_profit_factor=constraint_config['min_profit_factor'],
            max_drawdown_percent=constraint_config['max_drawdown_percent'],
            min_trades_per_year=constraint_config['min_trades_per_year']
        )

        # Split data into windows based on mode
        if progress_callback:
            progress_callback("Splitting data into windows", 0, 1)

        windows = self._split_into_windows(data, mode=walk_forward_mode)

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

            # Calculate degradation (positive = OOS worse than IS, negative = OOS better)
            # Use improved formula that handles edge cases better
            in_sortino = in_sample_metrics.get('sortino_ratio', 0.0)
            out_sortino = out_sample_metrics.get('sortino_ratio', 0.0)
            sortino_degradation = self._calculate_degradation(in_sortino, out_sortino)

            in_sharpe = in_sample_metrics.get('sharpe_ratio', 0.0)
            out_sharpe = out_sample_metrics.get('sharpe_ratio', 0.0)
            sharpe_degradation = self._calculate_degradation(in_sharpe, out_sharpe)

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

    def optimize_portfolio(self,
                          strategy_class: Type[BaseStrategy],
                          data_dict: Dict[str, pd.DataFrame],
                          capital_contention: Optional[CapitalContentionConfig] = None,
                          initial_capital: float = 100000.0,
                          selected_params: Optional[Dict[str, bool]] = None,
                          progress_callback: Optional[Callable[[str, int, int], None]] = None,
                          walk_forward_mode: Optional[WalkForwardMode] = None) -> WalkForwardResults:
        """
        Run walk-forward optimization for a portfolio of securities.

        Unlike single-security optimization, this optimizes parameters for the
        combined portfolio performance with shared capital.

        Args:
            strategy_class: Strategy class to optimize
            data_dict: Dict mapping symbol to historical price data
            capital_contention: Capital contention configuration for portfolio
            initial_capital: Initial capital for portfolio
            selected_params: Dict of {param_name: bool} indicating which parameters to optimize
            progress_callback: Callback(stage_name, current, total)
            walk_forward_mode: Walk-forward mode (ROLLING or ANCHORED)

        Returns:
            WalkForwardResults with combined portfolio optimization results
        """
        symbols = list(data_dict.keys())
        mode_desc = walk_forward_mode.value if walk_forward_mode else "config default"
        logger.info(f"Starting PORTFOLIO walk-forward optimization for {strategy_class.__name__} "
                   f"on {len(symbols)} securities (mode: {mode_desc})")

        # Reset cancellation flag
        self.should_cancel = False

        # Use default capital contention if not specified
        if capital_contention is None:
            capital_contention = CapitalContentionConfig.default_mode()

        # Load constraints
        constraint_config = self.config['optimization']['constraints']
        constraints = OptimizationConstraints(
            min_profit_factor=constraint_config['min_profit_factor'],
            max_drawdown_percent=constraint_config['max_drawdown_percent'],
            min_trades_per_year=constraint_config['min_trades_per_year']
        )

        # Find common date range across all securities
        min_date = max(df['date'].min() for df in data_dict.values())
        max_date = min(df['date'].max() for df in data_dict.values())

        # Create combined data for window splitting (use any security for dates)
        reference_data = list(data_dict.values())[0]
        reference_data = reference_data[(reference_data['date'] >= min_date) &
                                        (reference_data['date'] <= max_date)].copy()

        if progress_callback:
            progress_callback("Splitting data into windows", 0, 1)

        windows = self._split_into_windows(reference_data, mode=walk_forward_mode)

        if len(windows) == 0:
            raise ValueError("Not enough data to create walk-forward windows")

        # Optimize each window
        window_results = []
        _, param_names, fixed_params = self._get_parameter_space(strategy_class, selected_params)
        all_param_names = list(param_names) + list(fixed_params.keys())
        all_params = {param: [] for param in all_param_names}

        for window_id, (train_ref, test_ref) in enumerate(windows):
            if self.should_cancel:
                logger.info("Optimization cancelled by user")
                break

            logger.info(f"Processing portfolio window {window_id + 1}/{len(windows)}")

            if progress_callback:
                progress_callback(f"Optimizing window {window_id + 1}/{len(windows)}",
                                window_id, len(windows))

            # Get train/test date ranges
            train_start = train_ref['date'].min()
            train_end = train_ref['date'].max()
            test_start = test_ref['date'].min()
            test_end = test_ref['date'].max()

            # Split all securities' data for this window
            train_data_dict = {}
            test_data_dict = {}
            for symbol, df in data_dict.items():
                train_data_dict[symbol] = df[(df['date'] >= train_start) &
                                             (df['date'] <= train_end)].copy()
                test_data_dict[symbol] = df[(df['date'] >= test_start) &
                                            (df['date'] <= test_end)].copy()

            try:
                # Optimize on training data using portfolio
                def window_progress(current, total):
                    if progress_callback:
                        progress_callback(
                            f"Window {window_id + 1}/{len(windows)} - Iteration {current}/{total}",
                            current, total
                        )

                best_params = self._optimize_portfolio_window(
                    strategy_class, train_data_dict, constraints,
                    capital_contention, initial_capital, selected_params, window_progress
                )
            except Exception as e:
                logger.error(f"Error optimizing portfolio window {window_id + 1}: {e}")
                logger.exception("Portfolio window optimization failed")
                continue

            # Track parameter values
            for param_name, param_value in best_params.items():
                all_params[param_name].append(param_value)

            # Evaluate on training data (in-sample)
            in_sample_metrics = self._backtest_portfolio_with_params(
                strategy_class, best_params, train_data_dict,
                capital_contention, initial_capital
            )

            # Evaluate on testing data (out-of-sample)
            out_sample_metrics = self._backtest_portfolio_with_params(
                strategy_class, best_params, test_data_dict,
                capital_contention, initial_capital
            )

            # Calculate degradation
            in_sortino = in_sample_metrics.get('sortino_ratio', 0.0)
            out_sortino = out_sample_metrics.get('sortino_ratio', 0.0)
            sortino_degradation = self._calculate_degradation(in_sortino, out_sortino)

            in_sharpe = in_sample_metrics.get('sharpe_ratio', 0.0)
            out_sharpe = out_sample_metrics.get('sharpe_ratio', 0.0)
            sharpe_degradation = self._calculate_degradation(in_sharpe, out_sharpe)

            # Create window result
            window_result = WindowResult(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                in_sample_sortino=in_sortino,
                in_sample_sharpe=in_sharpe,
                in_sample_profit_factor=in_sample_metrics.get('profit_factor', 0.0),
                in_sample_max_drawdown_pct=in_sample_metrics.get('max_drawdown_pct', 0.0),
                in_sample_num_trades=in_sample_metrics.get('num_trades', 0),
                in_sample_total_return_pct=in_sample_metrics.get('total_return_pct', 0.0),
                in_sample_calmar=in_sample_metrics.get('calmar_ratio', 0.0),
                out_sample_sortino=out_sortino,
                out_sample_sharpe=out_sharpe,
                out_sample_profit_factor=out_sample_metrics.get('profit_factor', 0.0),
                out_sample_max_drawdown_pct=out_sample_metrics.get('max_drawdown_pct', 0.0),
                out_sample_num_trades=out_sample_metrics.get('num_trades', 0),
                out_sample_total_return_pct=out_sample_metrics.get('total_return_pct', 0.0),
                out_sample_calmar=out_sample_metrics.get('calmar_ratio', 0.0),
                sortino_degradation_pct=sortino_degradation,
                sharpe_degradation_pct=sharpe_degradation
            )
            window_results.append(window_result)

        if not window_results:
            raise ValueError("No windows completed successfully")

        # Calculate aggregate statistics
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
                most_common_params[param_name] = np.median(values)

        # Count windows that passed constraints
        windows_passed = sum(1 for w in window_results
                           if w.out_sample_profit_factor >= constraints.min_profit_factor
                           and w.out_sample_max_drawdown_pct <= constraints.max_drawdown_percent)

        results = WalkForwardResults(
            strategy_name=strategy_class.__name__,
            symbol=f"PORTFOLIO({','.join(symbols)})",
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

        logger.info(f"Portfolio walk-forward optimization complete. {len(window_results)} windows processed.")
        return results

    def _optimize_portfolio_window(self,
                                   strategy_class: Type[BaseStrategy],
                                   data_dict: Dict[str, pd.DataFrame],
                                   constraints: OptimizationConstraints,
                                   capital_contention: CapitalContentionConfig,
                                   initial_capital: float,
                                   selected_params: Optional[Dict[str, bool]] = None,
                                   progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Optimize parameters for a single window using portfolio backtesting.

        Uses Bayesian optimization with the same approach as single-security,
        but evaluates the objective using combined portfolio performance.
        """
        # Get parameter space
        space, param_names, fixed_params = self._get_parameter_space(strategy_class, selected_params)

        if len(space) == 0:
            return fixed_params

        # Get Bayesian settings
        bayes_config = self.config['bayesian_optimization']
        speed_mode = bayes_config.get('speed_mode', 'full')
        n_jobs = bayes_config.get('n_jobs', 1)

        if speed_mode == 'quick':
            n_calls = 25
            n_random = 8
        elif speed_mode == 'fast':
            n_calls = 50
            n_random = 15
        else:
            n_calls = 100
            n_random = 30

        # Adjust for Windows compatibility
        if platform.system() == 'Windows' and n_jobs != 1:
            logger.info(f"Windows detected: Parallel processing (n_jobs={n_jobs}) may not be available. Will attempt and fall back to serial if needed.")

        # Track progress
        iteration = [0]

        @use_named_args(space)
        def objective(**params):
            iteration[0] += 1

            if self.should_cancel:
                return 0.0

            if progress_callback:
                progress_callback(iteration[0], n_calls)

            # Combine with fixed params
            all_params = {**fixed_params, **params}

            try:
                metrics = self._backtest_portfolio_with_params(
                    strategy_class, all_params, data_dict,
                    capital_contention, initial_capital
                )
            except Exception as e:
                logger.warning(f"Backtest failed with params {all_params}: {e}")
                return 0.0

            # Calculate penalty for constraint violations
            penalty = self._calculate_constraint_penalty(metrics, constraints)

            # Objective: maximize Sortino (minimize negative Sortino)
            sortino = metrics.get('sortino_ratio', 0.0)
            objective_value = -sortino + penalty

            return objective_value

        try:
            result = gp_minimize(
                objective,
                space,
                n_calls=n_calls,
                n_initial_points=n_random,
                random_state=bayes_config.get('random_state'),
                n_jobs=n_jobs
            )
        except (ImportError, OSError, AttributeError) as e:
            # Multiprocessing not available in this environment
            if n_jobs != 1:
                # Provide platform-specific message
                if platform.system() == 'Windows':
                    logger.info(f"Parallel processing not available on Windows (expected). Using serial processing (n_jobs=1).")
                else:
                    logger.warning(f"Parallel processing not available ({e}). Falling back to serial processing (n_jobs=1)")

                # Retry with serial processing
                result = gp_minimize(
                    objective,
                    space,
                    n_calls=n_calls,
                    n_initial_points=n_random,
                    random_state=bayes_config.get('random_state'),
                    n_jobs=1
                )
            else:
                raise

        try:
            # Extract best parameters
            best_params = {name: value for name, value in zip(param_names, result.x)}
            best_params.update(fixed_params)

            return best_params

        except Exception as e:
            logger.error(f"Portfolio window optimization failed: {e}")
            default_params = {}
            for dim in space:
                if hasattr(dim, 'bounds'):
                    default_params[dim.name] = (dim.bounds[0] + dim.bounds[1]) / 2
                else:
                    default_params[dim.name] = dim.categories[0]
            default_params.update(fixed_params)
            return default_params

    def _backtest_portfolio_with_params(self,
                                        strategy_class: Type[BaseStrategy],
                                        params: Dict[str, Any],
                                        data_dict: Dict[str, pd.DataFrame],
                                        capital_contention: CapitalContentionConfig,
                                        initial_capital: float) -> Dict[str, Any]:
        """Run portfolio backtest with given parameters and return metrics."""
        strategy = strategy_class(**params)

        config = PortfolioConfig(
            initial_capital=initial_capital,
            commission=CommissionConfig(),
            capital_contention=capital_contention
        )

        engine = PortfolioEngine(config=config)
        result = engine.run(data_dict, strategy)

        # Calculate metrics from portfolio result
        all_trades = []
        for sym_result in result.symbol_results.values():
            all_trades.extend(sym_result.trades)

        metrics = PerformanceMetrics.calculate_from_trades(all_trades, initial_capital)

        # Add portfolio-specific metrics
        metrics['total_return'] = result.total_return
        metrics['total_return_pct'] = result.total_return_pct
        metrics['final_equity'] = result.final_equity
        metrics['num_swaps'] = len(result.vulnerability_swaps)
        metrics['num_rejections'] = len(result.signal_rejections)

        # Calculate Sharpe, Sortino, and other metrics from equity curve if available
        if not result.portfolio_equity_curve.empty:
            metrics['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(result.portfolio_equity_curve)
            metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(result.portfolio_equity_curve)
            max_dd, max_dd_pct = PerformanceMetrics.calculate_max_drawdown(result.portfolio_equity_curve)
            metrics['max_drawdown'] = max_dd
            metrics['max_drawdown_pct'] = max_dd_pct

            # Calculate CAGR and Calmar ratio
            equity_curve = result.portfolio_equity_curve
            if len(equity_curve) >= 2:
                start_date = pd.Timestamp(equity_curve['date'].iloc[0])
                end_date = pd.Timestamp(equity_curve['date'].iloc[-1])
                years = (end_date - start_date).days / 365.25
                initial_equity = equity_curve['equity'].iloc[0]
                final_equity = equity_curve['equity'].iloc[-1]

                if years > 0 and initial_equity > 0:
                    cagr = (pow(final_equity / initial_equity, 1 / years) - 1) * 100
                else:
                    cagr = 0.0
            else:
                cagr = 0.0

            metrics['cagr'] = cagr
            metrics['calmar_ratio'] = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

        return metrics
