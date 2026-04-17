# -*- coding: utf-8 -*-
"""
===================================
统一回测框架
===================================

这是 strategy_backtest 和 advanced_backtest 合并后的统一回测框架。

导入示例：
    from src.core.backtest.core import (
        StrategyBacktestEngine,
        Portfolio,
        Position,
        Order,
        Trade,
    )
    
    from src.core.backtest.factors import (
        Factor,
        FactorCombination,
        MomentumFactor,
    )
    
    from src.core.backtest.strategies import (
        MultiFactorStrategy,
    )
    
    from src.core.backtest.metrics import (
        PerformanceMetrics,
    )
    
    from src.core.backtest.optimization import (
        GridSearchOptimizer,
        OverfitDetector,
    )
    
    from src.core.backtest.visualization import (
        BacktestVisualizer,
    )
    
    from src.core.backtest.orchestration import (
        BacktestOrchestrator,
    )
"""

from . import core
from . import factors
from . import strategies
from . import metrics
from . import optimization
from . import visualization
from . import orchestration
from . import sensitivity
from . import reporting

__all__ = [
    "core",
    "factors",
    "strategies",
    "metrics",
    "optimization",
    "visualization",
    "orchestration",
    "sensitivity",
    "reporting",
]
