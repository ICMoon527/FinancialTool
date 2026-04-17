# -*- coding: utf-8 -*-
"""
高级回测系统 (Deprecated)

⚠️  此模块已迁移到统一的回测框架：
  - src/core/backtest/

请使用以下新导入方式：
  from src.core.backtest.factors import Factor, FactorCombination
  from src.core.backtest.optimization import GridSearchOptimizer
  from src.core.backtest.strategies import MultiFactorStrategy
"""

import warnings
warnings.warn(
    "advanced_backtest 已合并为统一的 backtest 模块，"
    "请使用 src.core.backtest 替代。",
    DeprecationWarning,
    stacklevel=2
)

# -*- coding: utf-8 -*-
"""
===================================
高级回测系统模块
===================================

提供多因子回测、参数优化、过拟合检测等高级功能。
"""

from .factor import (
    Factor,
    FactorConfig,
    FactorType,
    FactorDirection,
    FactorCombination,
    MomentumFactor,
    ValueFactor,
    VolatilityFactor,
    VolumeFactor,
    create_factor,
)
from .strategy import (
    MultiFactorStrategy,
    MultiFactorStrategyConfig,
    RebalanceFrequency,
    PositionSizingMethod,
)
from .engine import (
    BacktestConfig,
    BacktestResult,
    AdvancedBacktestEngine,
)
from .visualizer import Visualizer
from .optimization import (
    ParameterRange,
    OptimizationResult,
    WalkForwardResult,
    OptimizationMethod,
    OverfitCheckMethod,
    ParameterOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    OverfitDetector,
    create_optimizer,
)

__all__ = [
    "Factor",
    "FactorConfig",
    "FactorType",
    "FactorDirection",
    "FactorCombination",
    "MomentumFactor",
    "ValueFactor",
    "VolatilityFactor",
    "VolumeFactor",
    "create_factor",
    "MultiFactorStrategy",
    "MultiFactorStrategyConfig",
    "RebalanceFrequency",
    "PositionSizingMethod",
    "BacktestConfig",
    "BacktestResult",
    "AdvancedBacktestEngine",
    "Visualizer",
    "ParameterRange",
    "OptimizationResult",
    "WalkForwardResult",
    "OptimizationMethod",
    "OverfitCheckMethod",
    "ParameterOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "OverfitDetector",
    "create_optimizer",
]

