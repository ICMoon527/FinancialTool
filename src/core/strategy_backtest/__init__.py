# -*- coding: utf-8 -*-
"""
策略回测系统 (Deprecated)

⚠️  此模块已迁移到统一的回测框架：
  - src/core/backtest/

请使用以下新导入方式：
  from src.core.backtest.core import StrategyBacktestEngine, Portfolio
  from src.core.backtest.metrics import PerformanceMetrics
  from src.core.backtest.visualization import BacktestVisualizer
"""

import warnings
warnings.warn(
    "strategy_backtest 已合并为统一的 backtest 模块，"
    "请使用 src.core.backtest 替代。",
    DeprecationWarning,
    stacklevel=2
)

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
