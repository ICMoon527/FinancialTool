# -*- coding: utf-8 -*-
"""
Python Code Strategy Loader.

Loads strategies from Python modules with automatic discovery.
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

try:
    import importlib.util
    HAS_IMPORTLIB_UTIL = True
except ImportError:
    HAS_IMPORTLIB_UTIL = False

from stock_selector.base import StockSelectorStrategy

logger = logging.getLogger(__name__)

# Global registry for Python strategies
_python_strategy_registry: Dict[str, Type[StockSelectorStrategy]] = {}


def register_strategy(strategy_class: Type[StockSelectorStrategy]) -> Type[StockSelectorStrategy]:
    """
    Decorator to register a Python strategy class.

    Args:
        strategy_class: Strategy class to register

    Returns:
        The same strategy class (for decorator chaining)
    """
    # Create a temporary instance to get the strategy ID
    try:
        temp_instance = strategy_class()
        strategy_id = temp_instance.id
        _python_strategy_registry[strategy_id] = strategy_class
        logger.debug(f"Registered Python strategy: {strategy_id}")
    except Exception as e:
        logger.warning(f"Failed to register strategy {strategy_class.__name__}: {e}")
    return strategy_class


def get_registered_strategies() -> Dict[str, Type[StockSelectorStrategy]]:
    """
    Get all registered Python strategies.

    Returns:
        Dictionary of {strategy_id: strategy_class}
    """
    return _python_strategy_registry.copy()


def get_strategy_class(strategy_id: str) -> Optional[Type[StockSelectorStrategy]]:
    """
    Get a registered strategy class by ID.

    Args:
        strategy_id: Strategy ID to look up

    Returns:
        Strategy class or None if not found
    """
    return _python_strategy_registry.get(strategy_id)


def load_python_strategies_from_dir(directory: Path) -> List[StockSelectorStrategy]:
    """
    Load all Python strategies from a directory.

    Args:
        directory: Directory containing Python strategy modules

    Returns:
        List of strategy instances
    """
    strategies = []
    if not directory.is_dir():
        logger.warning(f"Python strategy directory not found: {directory}")
        return strategies

    # Add directory to sys.path for imports
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

    # Discover and load Python files
    py_files = sorted(directory.glob("*.py"))
    for py_path in py_files:
        if py_path.name.startswith("_"):
            continue

        try:
            module_name = py_path.stem
            if HAS_IMPORTLIB_UTIL:
                spec = importlib.util.spec_from_file_location(module_name, py_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.debug(f"Loaded Python strategy module: {module_name}")
            else:
                logger.warning(f"importlib.util not available, skipping {module_name}")
        except Exception as e:
            logger.warning(f"Failed to load Python strategy from {py_path}: {e}")

    # Create instances of all registered strategies
    for strategy_id, strategy_class in _python_strategy_registry.items():
        try:
            strategy = strategy_class()
            strategies.append(strategy)
            logger.debug(f"Created instance of Python strategy: {strategy_id}")
        except Exception as e:
            logger.warning(f"Failed to create instance of strategy {strategy_id}: {e}")

    logger.info(f"Loaded {len(strategies)} Python strategies from {directory}")
    return strategies
