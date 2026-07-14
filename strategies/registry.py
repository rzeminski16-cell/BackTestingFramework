"""
Canonical strategy registry with plugin-style discovery.

Strategies are discovered automatically, so adding one is just dropping a
module into ``strategies/`` — no registry edit required:

1. **Package scan** — every module in the ``strategies`` package is imported
   and searched for concrete :class:`BaseStrategy` subclasses.
2. **Entry points** — external packages can contribute strategies without
   touching this repository by declaring::

       [project.entry-points."btf.strategies"]
       MyStrategy = "my_package.my_module:MyStrategy"

Scaffold a new strategy with ``python -m btf new-strategy MyStrategy``.

The public API is stable: ``STRATEGY_REGISTRY`` (name -> class) and
``get_strategy_class(name)``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Type

from Classes.Strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "btf.strategies"


def _discover_package_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Import every module in the strategies package and collect concrete
    BaseStrategy subclasses (abstract intermediates are skipped)."""
    import strategies as pkg

    found: Dict[str, Type[BaseStrategy]] = {}
    for module_info in pkgutil.iter_modules(pkg.__path__):
        name = module_info.name
        if name.startswith("_") or name == "registry":
            continue
        try:
            module = importlib.import_module(f"strategies.{name}")
        except Exception as exc:
            logger.warning("Skipping strategies.%s (import failed: %s)", name, exc)
            continue
        for attr_name, obj in vars(module).items():
            if (inspect.isclass(obj)
                    and issubclass(obj, BaseStrategy)
                    and obj is not BaseStrategy
                    and not inspect.isabstract(obj)
                    # only classes defined in this module, not re-imports
                    and obj.__module__ == module.__name__):
                found[obj.__name__] = obj
    return found


def _discover_entry_point_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Load strategies contributed by installed packages via entry points."""
    found: Dict[str, Type[BaseStrategy]] = {}
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:
        return found
    for ep in eps:
        try:
            obj = ep.load()
            if (inspect.isclass(obj) and issubclass(obj, BaseStrategy)
                    and not inspect.isabstract(obj)):
                found[ep.name] = obj
            else:
                logger.warning("Entry point %s (%s) is not a concrete "
                               "BaseStrategy; skipped.", ep.name, ep.value)
        except Exception as exc:
            logger.warning("Failed loading strategy entry point %s: %s",
                           ep.name, exc)
    return found


def build_registry() -> Dict[str, Type[BaseStrategy]]:
    """(Re)build the full registry: package scan first, entry points can
    add — but not silently replace — repo strategies."""
    registry = _discover_package_strategies()
    for name, cls in _discover_entry_point_strategies().items():
        if name in registry and registry[name] is not cls:
            logger.warning("Entry-point strategy %s shadows a repo strategy; "
                           "keeping the repo version.", name)
            continue
        registry[name] = cls
    return registry


STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = build_registry()


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
