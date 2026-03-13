# -*- coding: utf-8 -*-
"""
Stock Selector Module - A scalable stock screening system.
"""

from .base import (
    StockCandidate,
    StrategyMatch,
    StockSelectorStrategy,
    StrategyMetadata,
)
from .config import (
    StockSelectorConfig,
    get_config,
    set_config,
)
from .manager import StrategyManager
from .service import StockSelectorService

__all__ = [
    "StockCandidate",
    "StrategyMatch",
    "StockSelectorStrategy",
    "StrategyMetadata",
    "StockSelectorConfig",
    "get_config",
    "set_config",
    "StrategyManager",
    "StockSelectorService",
]
