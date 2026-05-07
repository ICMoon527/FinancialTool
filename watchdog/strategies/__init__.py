# -*- coding: utf-8 -*-
"""
Watchdog Strategies Module

Built-in watch strategies and intraday T0 strategies.
"""

from typing import List

from watchdog.base import (
    AlertLevel,
    ConditionType,
    StrategyType,
    StrategyMetadata,
    WatchdogCondition,
    WatchdogDecision,
    WatchdogStrategy,
)

from watchdog.strategies.intraday_t0_strategy import (
    IndicatorSnapshot,
    IntradayDataBuffer,
    IntradayIndicatorEngine,
    IntradayT0Strategy,
    SignalEvaluator,
    T0Signal,
)
from watchdog.strategies.simulator import (
    SimulationReport,
    T0Simulator,
    TradeRecord,
    generate_simulated_intraday_data,
)


def get_builtin_strategies() -> List[WatchdogStrategy]:
    """
    Get all built-in watch strategies.

    Returns:
        List of WatchdogStrategy
    """
    strategies: List[WatchdogStrategy] = []

    # 注释：由于 WatchdogStrategy 接口已变更（需 StrategyMetadata），
    # 原有策略实例化代码需要后续适配。
    # 当前保留该函数框架，实际策略通过 IntradayT0Strategy 独立使用。

    return strategies
