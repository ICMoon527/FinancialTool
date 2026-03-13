# -*- coding: utf-8 -*-
"""
Watchdog Module - A股智能盯盘助手

Allow users to select stocks, set trading strategies, and monitor market dynamics automatically.
Sends alerts and provides decision recommendations when market conditions trigger.
"""

__version__ = "1.0.0"
__author__ = "Daily Stock Analysis Team"

from watchdog.base import (
    ActionType,
    AlertLevel,
    ConditionType,
    StrategyType,
    WatchdogAlert,
    WatchdogCondition,
    WatchdogDecision,
    WatchdogStrategy,
    WatchdogWatchlist,
)
from watchdog.config import WatchdogConfig, WatchdogPersistence, get_config, set_config
from watchdog.monitor import MarketDataProvider, WatchdogMonitor
from watchdog.notifier import WatchdogNotifier
from watchdog.service import WatchdogService

__all__ = [
    "ActionType",
    "AlertLevel",
    "ConditionType",
    "StrategyType",
    "WatchdogAlert",
    "WatchdogCondition",
    "WatchdogDecision",
    "WatchdogStrategy",
    "WatchdogWatchlist",
    "WatchdogConfig",
    "WatchdogPersistence",
    "get_config",
    "set_config",
    "MarketDataProvider",
    "WatchdogMonitor",
    "WatchdogNotifier",
    "WatchdogService",
]
