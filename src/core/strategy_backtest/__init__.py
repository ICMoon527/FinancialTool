# -*- coding: utf-8 -*-
"""
策略回测系统。

为六维选股策略提供完整的历史数据回测功能。
"""

from .data_access import TimeIsolatedDataProvider
from .engine import (
    StrategyBacktestEngine,
    Portfolio,
    Position,
    Trade,
    Order,
    OrderType,
    OrderStatus,
)
from .metrics import PerformanceMetrics
from .visualization import BacktestVisualizer
from .report import BacktestReportGenerator
from .sensitivity import ParameterSensitivityTest
from .orchestrator import BacktestOrchestrator

__all__ = [
    "TimeIsolatedDataProvider",
    "StrategyBacktestEngine",
    "Portfolio",
    "Position",
    "Trade",
    "Order",
    "OrderType",
    "OrderStatus",
    "PerformanceMetrics",
    "BacktestVisualizer",
    "BacktestReportGenerator",
    "ParameterSensitivityTest",
    "BacktestOrchestrator",
]
