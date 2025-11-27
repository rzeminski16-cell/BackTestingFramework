"""
Strategy parameter optimization.
"""
import itertools
import pandas as pd
from typing import Dict, List, Any, Type
from ..Strategy.base_strategy import BaseStrategy
from ..Engine.single_security_engine import SingleSecurityEngine
from ..Engine.backtest_result import BacktestResult
from ..Config.config import BacktestConfig, OptimizationConfig
from .optimization_result import OptimizationResult, OptimizationResults


class StrategyOptimizer:
    """
    Optimizes strategy parameters using grid search.

    Supports:
    - Global optimization (find best parameters across all securities)
    - Per-security optimization (find best parameters for each security)
    - Customizable metrics
    """

    def __init__(self, config: OptimizationConfig, backtest_config: BacktestConfig):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
            backtest_config: Backtest configuration
        """
        self.config = config
        self.backtest_config = backtest_config

    def optimize(self, strategy_class: Type[BaseStrategy],
                param_grid: Dict[str, List[Any]],
                symbol: str, data: pd.DataFrame) -> OptimizationResults:
        """
        Optimize strategy parameters for a single security.

        Args:
            strategy_class: Strategy class (not instance)
            param_grid: Parameter grid {param_name: [value1, value2, ...]}
            symbol: Security symbol
            data: Price/indicator data

        Returns:
            OptimizationResults with ranked results

        Example:
            param_grid = {
                'ma_period': [20, 50, 100],
                'position_size': [0.1, 0.2, 0.3]
            }
            results = optimizer.optimize(MyStrategy, param_grid, 'AAPL', data)
            best_params = results.get_best_parameters()
        """
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(param_grid)

        # Test each combination
        optimization_results = []

        for params in param_combinations:
            # Create strategy instance with these parameters
            strategy = strategy_class(**params)

            # Run backtest
            engine = SingleSecurityEngine(self.backtest_config)
            backtest_result = engine.run(symbol, data, strategy)

            # Extract metric value
            metric_value = self._extract_metric(backtest_result)

            # Skip if insufficient trades
            if backtest_result.num_trades < self.config.min_trades:
                continue

            # Store result
            opt_result = OptimizationResult(
                parameters=params,
                backtest_result=backtest_result,
                metric_value=metric_value
            )
            optimization_results.append(opt_result)

        # Create results collection
        results = OptimizationResults(
            results=optimization_results,
            metric=self.config.metric,
            maximize=self.config.maximize
        )

        return results

    def optimize_multiple(self, strategy_class: Type[BaseStrategy],
                         param_grid: Dict[str, List[Any]],
                         data_dict: Dict[str, pd.DataFrame]) -> Dict[str, OptimizationResults]:
        """
        Optimize parameters for multiple securities.

        If config.per_security is True, finds best parameters for each security.
        If False, finds single best parameters across all securities.

        Args:
            strategy_class: Strategy class
            param_grid: Parameter grid
            data_dict: Dictionary mapping symbol to data

        Returns:
            Dictionary mapping symbol to OptimizationResults
        """
        if self.config.per_security:
            # Optimize separately for each security
            results = {}
            for symbol, data in data_dict.items():
                print(f"Optimizing {symbol}...")
                results[symbol] = self.optimize(
                    strategy_class, param_grid, symbol, data
                )
            return results
        else:
            # Find global best parameters
            return self._optimize_global(strategy_class, param_grid, data_dict)

    def _optimize_global(self, strategy_class: Type[BaseStrategy],
                        param_grid: Dict[str, List[Any]],
                        data_dict: Dict[str, pd.DataFrame]) -> Dict[str, OptimizationResults]:
        """
        Find single best parameter set across all securities.

        Tests each parameter combination on all securities and
        ranks by average/total performance.

        Args:
            strategy_class: Strategy class
            param_grid: Parameter grid
            data_dict: Dictionary mapping symbol to data

        Returns:
            Dictionary mapping symbol to OptimizationResults (same best params for all)
        """
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(param_grid)

        # Track results for each combination
        combination_scores = []

        for params in param_combinations:
            # Test this combination on all securities
            total_metric = 0
            valid_count = 0

            for symbol, data in data_dict.items():
                strategy = strategy_class(**params)
                engine = SingleSecurityEngine(self.backtest_config)

                try:
                    backtest_result = engine.run(symbol, data, strategy)

                    if backtest_result.num_trades >= self.config.min_trades:
                        metric_value = self._extract_metric(backtest_result)
                        total_metric += metric_value
                        valid_count += 1
                except Exception as e:
                    print(f"Error testing {params} on {symbol}: {e}")
                    continue

            # Calculate average metric across securities
            if valid_count > 0:
                avg_metric = total_metric / valid_count
                combination_scores.append((params, avg_metric))

        # Sort by average metric
        combination_scores.sort(
            key=lambda x: x[1],
            reverse=self.config.maximize
        )

        # Get best parameters
        if not combination_scores:
            raise ValueError("No valid parameter combinations found")

        best_params = combination_scores[0][0]

        print(f"Best global parameters: {best_params}")

        # Run final backtest with best parameters for each security
        results = {}
        for symbol, data in data_dict.items():
            strategy = strategy_class(**best_params)
            engine = SingleSecurityEngine(self.backtest_config)
            backtest_result = engine.run(symbol, data, strategy)

            metric_value = self._extract_metric(backtest_result)

            opt_result = OptimizationResult(
                parameters=best_params,
                backtest_result=backtest_result,
                metric_value=metric_value
            )

            results[symbol] = OptimizationResults(
                results=[opt_result],
                metric=self.config.metric,
                maximize=self.config.maximize
            )

        return results

    def _generate_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from grid.

        Args:
            param_grid: Parameter grid

        Returns:
            List of parameter dictionaries
        """
        keys = param_grid.keys()
        values = param_grid.values()

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def _extract_metric(self, result: BacktestResult) -> float:
        """
        Extract metric value from backtest result.

        Args:
            result: Backtest result

        Returns:
            Metric value
        """
        metric = self.config.metric

        if metric == 'total_return':
            return result.total_return
        elif metric == 'total_return_pct':
            return result.total_return_pct
        elif metric == 'num_trades':
            return result.num_trades
        elif metric == 'win_rate':
            return result.win_rate
        elif metric == 'sharpe_ratio':
            # Calculate Sharpe ratio from equity curve
            if len(result.equity_curve) > 1:
                returns = result.equity_curve['equity'].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    return (returns.mean() / returns.std()) * (252 ** 0.5)  # Annualized
            return 0.0
        elif metric == 'profit_factor':
            wins = [t.pl for t in result.winning_trades]
            losses = [abs(t.pl) for t in result.losing_trades]
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 0
            if total_losses > 0:
                return total_wins / total_losses
            return 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
