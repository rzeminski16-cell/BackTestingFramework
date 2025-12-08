"""
Sensitivity Analysis Module

Tests parameter robustness by varying parameters around optimal values
and measuring performance changes. Identifies:
- Robust parameter ranges (performance plateau)
- Sensitive parameters (large performance swings)
- Interaction effects between parameters
"""

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from Classes.Analysis.performance_metrics import PerformanceMetrics
from Classes.Config.config import BacktestConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class SensitivityTest:
    """Single sensitivity test result."""
    test_id: int
    parameters: Dict[str, Any]
    parameter_variations: Dict[str, str]  # "base", "+10%", "-10%", "+1", "-1"

    # Performance metrics
    sortino_ratio: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    total_return_pct: float
    calmar_ratio: float
    num_trades: int

    # Comparison to base
    sortino_change_pct: float
    sharpe_change_pct: float
    return_change_pct: float


@dataclass
class ParameterSensitivity:
    """Sensitivity analysis for a single parameter."""
    parameter_name: str
    base_value: Any
    variation_type: str  # "percentage" or "absolute"

    # Tested values and results
    tested_values: List[Any] = field(default_factory=list)
    sortino_values: List[float] = field(default_factory=list)
    sharpe_values: List[float] = field(default_factory=list)

    # Sensitivity metrics
    sortino_range: float = 0.0  # max - min
    sortino_std: float = 0.0
    is_robust: bool = False  # True if variation < threshold
    is_unstable: bool = False  # True if variation > threshold


@dataclass
class SensitivityResults:
    """Complete sensitivity analysis results."""
    strategy_name: str
    symbol: str
    base_parameters: Dict[str, Any]

    # Individual parameter sensitivities
    parameter_sensitivities: Dict[str, ParameterSensitivity]

    # All sensitivity tests
    all_tests: List[SensitivityTest]

    # Overall robustness
    overall_sortino_range_pct: float
    overall_sharpe_range_pct: float
    is_overall_robust: bool

    # Most and least robust parameters
    most_robust_params: List[str]
    least_robust_params: List[str]


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on optimized parameters.

    Tests parameter robustness by:
    1. Varying each parameter by ±10% (float) or ±1 (int)
    2. Testing all combinations (if configured)
    3. Measuring performance changes
    4. Identifying robust parameter ranges
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize sensitivity analyzer."""
        self.config = config
        self.backtest_config = BacktestConfig()
        self.engine = SingleSecurityEngine(config=self.backtest_config)
        self.should_cancel = False

        # Sensitivity settings
        sens_config = config.get('sensitivity_analysis', {})
        self.float_variation_pct = sens_config.get('float_variation_percent', 10.0)
        self.int_variation = sens_config.get('int_variation_amount', 1)
        self.test_mode = sens_config.get('test_mode', 'combinations')
        self.max_tests = sens_config.get('max_sensitivity_tests', 1000)

        # Reporting thresholds
        report_config = config.get('reporting', {})
        self.robust_threshold = report_config.get('robust_threshold_percent', 15.0)
        self.unstable_threshold = report_config.get('unstable_threshold_percent', 50.0)

        logger.info("Sensitivity analyzer initialized")

    def cancel(self):
        """Cancel the sensitivity analysis."""
        self.should_cancel = True

    def _generate_variations(self, base_params: Dict[str, Any],
                           strategy_class: Type[BaseStrategy]) -> List[Dict[str, Any]]:
        """
        Generate parameter variations for sensitivity testing.

        Returns:
            List of parameter dictionaries to test
        """
        strategy_name = strategy_class.__name__
        param_config = self.config['strategy_parameters'].get(strategy_name, {})

        variations = [base_params.copy()]  # Start with base parameters
        param_variation_map = [{param: "base" for param in base_params}]

        if self.test_mode == 'individual':
            # Test each parameter individually
            for param_name, base_value in base_params.items():
                param_spec = param_config.get(param_name, {})
                param_type = param_spec.get('type', 'float')

                if param_type == 'float':
                    # ±10% variation
                    variation_amount = base_value * (self.float_variation_pct / 100.0)
                    plus_value = base_value + variation_amount
                    minus_value = base_value - variation_amount

                    # Respect bounds
                    min_val = param_spec.get('min', -np.inf)
                    max_val = param_spec.get('max', np.inf)
                    plus_value = min(max_val, plus_value)
                    minus_value = max(min_val, minus_value)

                    # Add variations
                    for val, label in [(plus_value, f"+{self.float_variation_pct}%"),
                                      (minus_value, f"-{self.float_variation_pct}%")]:
                        variant = base_params.copy()
                        variant[param_name] = val
                        variations.append(variant)

                        var_map = {p: "base" for p in base_params}
                        var_map[param_name] = label
                        param_variation_map.append(var_map)

                else:  # int
                    # ±1 variation
                    min_val = param_spec.get('min', -np.inf)
                    max_val = param_spec.get('max', np.inf)

                    for delta, label in [(self.int_variation, f"+{self.int_variation}"),
                                        (-self.int_variation, f"-{self.int_variation}")]:
                        new_val = int(base_value + delta)
                        if min_val <= new_val <= max_val:
                            variant = base_params.copy()
                            variant[param_name] = new_val
                            variations.append(variant)

                            var_map = {p: "base" for p in base_params}
                            var_map[param_name] = label
                            param_variation_map.append(var_map)

        else:  # combinations
            # Generate all combinations of variations
            param_names = list(base_params.keys())
            all_variations = []

            for param_name in param_names:
                base_value = base_params[param_name]
                param_spec = param_config.get(param_name, {})
                param_type = param_spec.get('type', 'float')

                if param_type == 'float':
                    variation_amount = base_value * (self.float_variation_pct / 100.0)
                    min_val = param_spec.get('min', -np.inf)
                    max_val = param_spec.get('max', np.inf)

                    values = [
                        (base_value, "base"),
                        (min(max_val, base_value + variation_amount), f"+{self.float_variation_pct}%"),
                        (max(min_val, base_value - variation_amount), f"-{self.float_variation_pct}%")
                    ]
                else:  # int
                    min_val = param_spec.get('min', -np.inf)
                    max_val = param_spec.get('max', np.inf)

                    values = [(base_value, "base")]
                    if base_value + self.int_variation <= max_val:
                        values.append((int(base_value + self.int_variation), f"+{self.int_variation}"))
                    if base_value - self.int_variation >= min_val:
                        values.append((int(base_value - self.int_variation), f"-{self.int_variation}"))

                all_variations.append(values)

            # Create all combinations
            combinations = list(product(*all_variations))

            # Limit number of tests
            if len(combinations) > self.max_tests:
                logger.warning(f"Too many combinations ({len(combinations)}). Sampling {self.max_tests}.")
                import random
                combinations = random.sample(combinations, self.max_tests)

            variations = []
            param_variation_map = []

            for combo in combinations:
                variant = {}
                var_map = {}
                for param_name, (value, label) in zip(param_names, combo):
                    variant[param_name] = value
                    var_map[param_name] = label
                variations.append(variant)
                param_variation_map.append(var_map)

        logger.info(f"Generated {len(variations)} parameter variations for sensitivity testing")
        return variations, param_variation_map

    def _evaluate_parameters(self, strategy_class: Type[BaseStrategy],
                           params: Dict[str, Any],
                           data: pd.DataFrame,
                           symbol: str) -> Dict[str, Any]:
        """Evaluate parameters and return metrics."""
        try:
            # Ensure proper type casting (parameters may be numpy types)
            strategy_name = strategy_class.__name__
            param_config = self.config['strategy_parameters'].get(strategy_name, {})
            typed_params = {}
            for param_name, param_value in params.items():
                if param_name in param_config:
                    param_type = param_config[param_name].get('type', 'float')
                    if param_type == 'int':
                        typed_params[param_name] = int(round(param_value))
                    else:
                        typed_params[param_name] = float(param_value)
                else:
                    # Parameter not in config, keep as is
                    typed_params[param_name] = param_value

            strategy = strategy_class(**typed_params)
            result = self.engine.run(symbol, data, strategy)
            metrics = PerformanceMetrics.calculate_metrics(result)
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return {
                'sortino_ratio': -999999.0,
                'sharpe_ratio': -999999.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 100.0,
                'total_return_pct': -100.0,
                'calmar_ratio': 0.0,
                'num_trades': 0
            }

    def analyze(self,
               strategy_class: Type[BaseStrategy],
               base_parameters: Dict[str, Any],
               data: pd.DataFrame,
               symbol: str,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> SensitivityResults:
        """
        Perform sensitivity analysis on the given parameters.

        Args:
            strategy_class: Strategy class
            base_parameters: Optimized parameters to test around
            data: Historical data
            symbol: Security symbol
            progress_callback: Progress callback(current, total)

        Returns:
            SensitivityResults
        """
        logger.info(f"Starting sensitivity analysis for {strategy_class.__name__} on {symbol}")

        self.should_cancel = False

        # Generate parameter variations
        variations, variation_maps = self._generate_variations(base_parameters, strategy_class)

        # Evaluate base parameters first
        base_metrics = self._evaluate_parameters(strategy_class, base_parameters, data, symbol)
        base_sortino = base_metrics.get('sortino_ratio', 0.0)
        base_sharpe = base_metrics.get('sharpe_ratio', 0.0)
        base_return = base_metrics.get('total_return_pct', 0.0)

        # Test all variations
        all_tests = []
        param_test_results = {param: {'values': [], 'sortino': [], 'sharpe': []}
                             for param in base_parameters.keys()}

        for test_id, (params, var_map) in enumerate(zip(variations, variation_maps)):
            if self.should_cancel:
                logger.info("Sensitivity analysis cancelled")
                break

            if progress_callback:
                progress_callback(test_id + 1, len(variations))

            # Evaluate parameters
            metrics = self._evaluate_parameters(strategy_class, params, data, symbol)

            # Calculate changes from base
            sortino = metrics.get('sortino_ratio', 0.0)
            sharpe = metrics.get('sharpe_ratio', 0.0)
            total_return = metrics.get('total_return_pct', 0.0)

            sortino_change = ((sortino - base_sortino) / abs(base_sortino) * 100
                            if base_sortino != 0 else 0.0)
            sharpe_change = ((sharpe - base_sharpe) / abs(base_sharpe) * 100
                           if base_sharpe != 0 else 0.0)
            return_change = ((total_return - base_return) / abs(base_return) * 100
                           if base_return != 0 else 0.0)

            # Create test result
            test = SensitivityTest(
                test_id=test_id,
                parameters=params,
                parameter_variations=var_map,
                sortino_ratio=sortino,
                sharpe_ratio=sharpe,
                profit_factor=metrics.get('profit_factor', 0.0),
                max_drawdown_pct=abs(metrics.get('max_drawdown_pct', 0.0)),
                total_return_pct=total_return,
                calmar_ratio=metrics.get('calmar_ratio', 0.0),
                num_trades=metrics.get('num_trades', 0),
                sortino_change_pct=sortino_change,
                sharpe_change_pct=sharpe_change,
                return_change_pct=return_change
            )

            all_tests.append(test)

            # Track individual parameter results
            for param_name, param_value in params.items():
                param_test_results[param_name]['values'].append(param_value)
                param_test_results[param_name]['sortino'].append(sortino)
                param_test_results[param_name]['sharpe'].append(sharpe)

        # Analyze parameter sensitivities
        parameter_sensitivities = {}

        for param_name, results in param_test_results.items():
            if len(results['sortino']) == 0:
                continue

            sortino_values = results['sortino']
            sharpe_values = results['sharpe']

            sortino_range = max(sortino_values) - min(sortino_values)
            sortino_std = np.std(sortino_values)

            # Determine if robust or unstable
            sortino_variation_pct = (sortino_range / abs(base_sortino) * 100
                                   if base_sortino != 0 else 0.0)

            is_robust = sortino_variation_pct < self.robust_threshold
            is_unstable = sortino_variation_pct > self.unstable_threshold

            param_spec = self.config['strategy_parameters'][strategy_class.__name__].get(param_name, {})
            variation_type = "percentage" if param_spec.get('type') == 'float' else "absolute"

            sensitivity = ParameterSensitivity(
                parameter_name=param_name,
                base_value=base_parameters[param_name],
                variation_type=variation_type,
                tested_values=results['values'],
                sortino_values=sortino_values,
                sharpe_values=sharpe_values,
                sortino_range=sortino_range,
                sortino_std=sortino_std,
                is_robust=is_robust,
                is_unstable=is_unstable
            )

            parameter_sensitivities[param_name] = sensitivity

        # Calculate overall robustness
        all_sortinos = [t.sortino_ratio for t in all_tests]
        all_sharpes = [t.sharpe_ratio for t in all_tests]

        overall_sortino_range = max(all_sortinos) - min(all_sortinos)
        overall_sharpe_range = max(all_sharpes) - min(all_sharpes)

        overall_sortino_range_pct = (overall_sortino_range / abs(base_sortino) * 100
                                    if base_sortino != 0 else 0.0)
        overall_sharpe_range_pct = (overall_sharpe_range / abs(base_sharpe) * 100
                                   if base_sharpe != 0 else 0.0)

        is_overall_robust = overall_sortino_range_pct < self.robust_threshold

        # Identify most and least robust parameters
        param_variations = [(name, sens.sortino_range / abs(base_sortino) * 100 if base_sortino != 0 else 0)
                          for name, sens in parameter_sensitivities.items()]
        param_variations.sort(key=lambda x: x[1])

        most_robust = [name for name, _ in param_variations[:3]]
        least_robust = [name for name, _ in param_variations[-3:]]

        results = SensitivityResults(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            base_parameters=base_parameters,
            parameter_sensitivities=parameter_sensitivities,
            all_tests=all_tests,
            overall_sortino_range_pct=overall_sortino_range_pct,
            overall_sharpe_range_pct=overall_sharpe_range_pct,
            is_overall_robust=is_overall_robust,
            most_robust_params=most_robust,
            least_robust_params=least_robust
        )

        logger.info(f"Sensitivity analysis complete. {len(all_tests)} tests performed.")
        return results
