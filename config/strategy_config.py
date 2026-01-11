"""
Centralized Strategy Parameter Configuration Loader.

This module provides a single source of truth for strategy parameters.
All systems (backtesting, optimization, walk-forward) should use this
module to get strategy parameter definitions.

Usage:
    from config.strategy_config import StrategyConfig

    # Get default parameters for a strategy
    defaults = StrategyConfig.get_defaults('AlphaTrendStrategy')

    # Get optimization ranges
    opt_params = StrategyConfig.get_optimization_params('AlphaTrendStrategy')

    # Get all configurable parameters
    configurable = StrategyConfig.get_configurable_params('AlphaTrendStrategy')
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class StrategyConfig:
    """Centralized strategy parameter configuration."""

    _config: Optional[Dict[str, Any]] = None
    _config_path: Path = Path(__file__).parent / 'strategy_parameters.json'

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if cls._config is None:
            with open(cls._config_path, 'r') as f:
                cls._config = json.load(f)
        return cls._config

    @classmethod
    def reload(cls) -> None:
        """Force reload of configuration from file."""
        cls._config = None
        cls._load_config()

    @classmethod
    def get_strategies(cls) -> List[str]:
        """Get list of all available strategy names."""
        config = cls._load_config()
        return list(config.get('strategies', {}).keys())

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """Get full strategy information including display name and description."""
        config = cls._load_config()
        strategy = config.get('strategies', {}).get(strategy_name, {})
        return {
            'name': strategy_name,
            'display_name': strategy.get('display_name', strategy_name),
            'description': strategy.get('description', ''),
            'trade_direction': strategy.get('trade_direction', 'LONG')
        }

    @classmethod
    def get_parameters(cls, strategy_name: str) -> Dict[str, Dict[str, Any]]:
        """Get all parameter definitions for a strategy."""
        config = cls._load_config()
        strategy = config.get('strategies', {}).get(strategy_name, {})
        return strategy.get('parameters', {})

    @classmethod
    def get_defaults(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get default parameter values for a strategy.

        Returns:
            Dict mapping parameter name to default value.
        """
        params = cls.get_parameters(strategy_name)
        return {
            name: param['default']
            for name, param in params.items()
        }

    @classmethod
    def get_configurable_params(cls, strategy_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get only configurable parameters (those that can be changed by user).

        Returns:
            Dict of parameter definitions that have configurable=True.
        """
        params = cls.get_parameters(strategy_name)
        return {
            name: param
            for name, param in params.items()
            if param.get('configurable', True)
        }

    @classmethod
    def get_optimization_params(cls, strategy_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get parameters with optimization ranges.

        Returns:
            Dict mapping parameter name to optimization config (min, max, step, type).
        """
        params = cls.get_parameters(strategy_name)
        result = {}
        for name, param in params.items():
            if param.get('optimization') and param.get('configurable', True):
                opt = param['optimization']
                result[name] = {
                    'type': param['type'],
                    'default': param['default'],
                    'min': opt.get('min'),
                    'max': opt.get('max'),
                    'step': opt.get('step'),
                    'description': param.get('description', '')
                }
        return result

    @classmethod
    def get_param_info(cls, strategy_name: str, param_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific parameter."""
        params = cls.get_parameters(strategy_name)
        return params.get(param_name)

    @classmethod
    def get_params_by_category(cls, strategy_name: str) -> Dict[str, List[str]]:
        """
        Get parameters grouped by category.

        Returns:
            Dict mapping category name to list of parameter names.
        """
        params = cls.get_parameters(strategy_name)
        categories: Dict[str, List[str]] = {}
        for name, param in params.items():
            category = param.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories

    @classmethod
    def validate_params(cls, strategy_name: str, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameter values against their definitions.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        param_defs = cls.get_parameters(strategy_name)

        for name, value in params.items():
            if name not in param_defs:
                errors.append(f"Unknown parameter: {name}")
                continue

            param_def = param_defs[name]
            param_type = param_def['type']

            # Type check
            if param_type == 'int' and not isinstance(value, int):
                try:
                    int(value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter {name} must be an integer")
                    continue
            elif param_type == 'float' and not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter {name} must be a number")
                    continue

            # Range check if optimization bounds exist
            opt = param_def.get('optimization')
            if opt:
                num_value = float(value) if param_type == 'float' else int(value)
                if opt.get('min') is not None and num_value < opt['min']:
                    errors.append(f"Parameter {name} must be >= {opt['min']}")
                if opt.get('max') is not None and num_value > opt['max']:
                    errors.append(f"Parameter {name} must be <= {opt['max']}")

        return errors

    @classmethod
    def get_optimization_grid(cls, strategy_name: str, param_name: str) -> List[Union[int, float]]:
        """
        Generate a grid of values for optimization.

        Returns:
            List of parameter values from min to max with step.
        """
        params = cls.get_optimization_params(strategy_name)
        if param_name not in params:
            return []

        opt = params[param_name]
        param_type = opt['type']
        min_val = opt['min']
        max_val = opt['max']
        step = opt['step']

        if min_val is None or max_val is None or step is None:
            return [opt['default']]

        values = []
        current = min_val
        while current <= max_val:
            if param_type == 'int':
                values.append(int(current))
            else:
                values.append(round(current, 6))
            current += step

        return values


# Convenience functions for quick access
def get_strategy_defaults(strategy_name: str) -> Dict[str, Any]:
    """Get default parameters for a strategy."""
    return StrategyConfig.get_defaults(strategy_name)


def get_strategy_opt_params(strategy_name: str) -> Dict[str, Dict[str, Any]]:
    """Get optimization parameters for a strategy."""
    return StrategyConfig.get_optimization_params(strategy_name)


def get_available_strategies() -> List[str]:
    """Get list of available strategies."""
    return StrategyConfig.get_strategies()
