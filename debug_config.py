#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script to check configuration loading and strategy filtering.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from stock_selector.config import get_config
from stock_selector.manager import StrategyManager


def debug_config():
    """Debug configuration loading and strategy filtering."""
    print("=== Debug Configuration ===")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    print("- STOCK_SELECTOR_PREFERRED_STRATEGY_TYPE:", os.getenv("STOCK_SELECTOR_PREFERRED_STRATEGY_TYPE"))
    print("- STOCK_SELECTOR_NL_STRATEGY_DIR:", os.getenv("STOCK_SELECTOR_NL_STRATEGY_DIR"))
    print("- STOCK_SELECTOR_PYTHON_STRATEGY_DIR:", os.getenv("STOCK_SELECTOR_PYTHON_STRATEGY_DIR"))
    
    # Get config
    config = get_config()
    print("\nLoaded Config:")
    print(f"- preferred_strategy_type: {config.preferred_strategy_type}")
    print(f"- nl_strategy_dir: {config.nl_strategy_dir}")
    print(f"- python_strategy_dir: {config.python_strategy_dir}")
    
    # Create strategy manager
    project_root = Path(__file__).parent
    nl_dir = project_root / "stock_selector" / "strategies"
    python_dir = project_root / "stock_selector" / "strategies"
    
    print("\nStrategy Directories:")
    print(f"- NL strategies dir: {nl_dir} (exists: {nl_dir.exists()})")
    print(f"- Python strategies dir: {python_dir} (exists: {python_dir.exists()})")
    
    manager = StrategyManager(
        nl_strategy_dir=nl_dir,
        python_strategy_dir=python_dir,
        config=config
    )
    
    print("\nStrategies After Filtering:")
    strategies = manager.get_all_strategies()
    print(f"Total strategies after filtering: {len(strategies)}")
    
    for strategy in strategies:
        print(f"- {strategy.display_name} (ID: {strategy.id}, Type: {strategy.metadata.strategy_type.name})")
    
    print("\nActive Strategies:")
    active_strategies = manager.get_active_strategies()
    print(f"Total active strategies: {len(active_strategies)}")
    
    for strategy in active_strategies:
        print(f"- {strategy.display_name} (ID: {strategy.id}, Type: {strategy.metadata.strategy_type.name})")


if __name__ == "__main__":
    import sys
    debug_config()
