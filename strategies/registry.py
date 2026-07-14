"""
Canonical strategy registry: one place that maps strategy names to classes.

The GUIs and the ``btf`` CLI both need "which strategies exist"; keeping the
mapping here means adding a strategy is one import + one list entry.
"""

from typing import Dict, List, Type

from Classes.Strategy.base_strategy import BaseStrategy

from strategies.alpha_trend_v1_strategy import AlphaTrendV1Strategy
from strategies.alpha_trend_v2_strategy import AlphaTrendV2Strategy
from strategies.alpha_trend_v2c2_strategy import AlphaTrendV2C2Strategy
from strategies.alpha_trend_v2c3_strategy import AlphaTrendV2C3Strategy
from strategies.alpha_trend_v3c1_strategy import AlphaTrendV3C1Strategy
from strategies.random_control_strategy import RandomControlStrategy
from strategies.short_only_base_strategy import ShortOnlyBaseStrategy

_STRATEGY_CLASSES: List[Type[BaseStrategy]] = [
    AlphaTrendV1Strategy,
    AlphaTrendV2Strategy,
    AlphaTrendV2C2Strategy,
    AlphaTrendV2C3Strategy,
    AlphaTrendV3C1Strategy,
    RandomControlStrategy,
    ShortOnlyBaseStrategy,
]

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    cls.__name__: cls for cls in _STRATEGY_CLASSES
}


def get_strategy_class(name: str) -> Type[BaseStrategy]:
    """
    Look up a strategy class by name.

    Raises:
        KeyError: with the list of available strategies when unknown.
    """
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {available}") from None
