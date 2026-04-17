# -*- coding: utf-8 -*-
"""
回测模块合并迁移脚本

将 strategy_backtest 和 advanced_backtest 合并为统一的 backtest 模块。
"""
import os
import shutil
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_step(step_num: int, description: str):
    """打印步骤信息"""
    print(f"\n{'=' * 80}")
    print(f"步骤 {step_num}: {description}")
    print(f"{'=' * 80}")


def copy_file(src: Path, dst: Path, update_imports: bool = True, old_prefix: str = "", new_prefix: str = ""):
    """复制文件并更新导入"""
    print(f"  复制: {src.name} -> {dst.name}")
    
    content = src.read_text(encoding='utf-8')
    
    if update_imports and old_prefix and new_prefix:
        # 更新导入语句
        old_import1 = f"from {old_prefix}"
        new_import1 = f"from {new_prefix}"
        old_import2 = f"from .{old_prefix.split('.')[-1]}"
        new_import2 = f"from .{new_prefix.split('.')[-1]}"
        
        content = content.replace(old_import1, new_import1)
        content = content.replace(old_import2, new_import2)
    
    # 确保目标目录存在
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    dst.write_text(content, encoding='utf-8')
    print(f"  ✓ 已写入: {dst}")


def main():
    """主迁移函数"""
    print_step(1, "检查源目录")
    
    strategy_backtest_dir = project_root / "src" / "core" / "strategy_backtest"
    advanced_backtest_dir = project_root / "src" / "core" / "advanced_backtest"
    backtest_dir = project_root / "src" / "core" / "backtest"
    
    print(f"  strategy_backtest: {strategy_backtest_dir.exists()}")
    print(f"  advanced_backtest: {advanced_backtest_dir.exists()}")
    print(f"  目标目录: {backtest_dir}")
    
    if not strategy_backtest_dir.exists() or not advanced_backtest_dir.exists():
        print("\n❌ 源目录不存在，无法迁移！")
        return
    
    print_step(2, "复制 strategy_backtest 核心模块")
    
    # 创建子目录
    (backtest_dir / "core").mkdir(exist_ok=True)
    (backtest_dir / "factors").mkdir(exist_ok=True)
    (backtest_dir / "strategies").mkdir(exist_ok=True)
    (backtest_dir / "metrics").mkdir(exist_ok=True)
    (backtest_dir / "optimization").mkdir(exist_ok=True)
    (backtest_dir / "visualization").mkdir(exist_ok=True)
    (backtest_dir / "orchestration").mkdir(exist_ok=True)
    (backtest_dir / "sensitivity").mkdir(exist_ok=True)
    (backtest_dir / "reporting").mkdir(exist_ok=True)
    
    # 复制 strategy_backtest 文件
    print("\n  从 strategy_backtest 复制:")
    
    # core/
    copy_file(strategy_backtest_dir / "engine.py", backtest_dir / "core" / "engine.py")
    copy_file(strategy_backtest_dir / "data_access.py", backtest_dir / "core" / "data_access.py")
    copy_file(strategy_backtest_dir / "log_capture.py", backtest_dir / "core" / "log_capture.py")
    copy_file(strategy_backtest_dir / "task_manager.py", backtest_dir / "core" / "task_manager.py")
    
    # metrics/
    copy_file(strategy_backtest_dir / "metrics.py", backtest_dir / "metrics" / "performance.py")
    
    # visualization/
    copy_file(strategy_backtest_dir / "visualization.py", backtest_dir / "visualization" / "core.py")
    
    # orchestration/
    copy_file(strategy_backtest_dir / "orchestrator.py", backtest_dir / "orchestration" / "orchestrator.py")
    copy_file(strategy_backtest_dir / "smart_data_preloader.py", backtest_dir / "orchestration" / "preloader.py")
    
    # sensitivity/
    copy_file(strategy_backtest_dir / "sensitivity.py", backtest_dir / "sensitivity" / "analysis.py")
    
    # reporting/
    copy_file(strategy_backtest_dir / "report.py", backtest_dir / "reporting" / "generator.py")
    
    print_step(3, "复制 advanced_backtest 模块")
    
    print("\n  从 advanced_backtest 复制:")
    
    # factors/
    copy_file(advanced_backtest_dir / "factor.py", backtest_dir / "factors" / "base.py")
    
    # strategies/
    copy_file(advanced_backtest_dir / "strategy.py", backtest_dir / "strategies" / "multi_factor.py")
    
    # optimization/
    copy_file(advanced_backtest_dir / "optimization.py", backtest_dir / "optimization" / "optimization.py")
    
    # visualization/
    copy_file(advanced_backtest_dir / "visualizer.py", backtest_dir / "visualization" / "advanced.py")
    
    # engine/ - 但 strategy_backtest 已经有 engine 了
    # 我们不复制，以 strategy_backtest 的 engine 为主
    # 但复制 advanced_backtest 的 engine 为 reference
    copy_file(advanced_backtest_dir / "engine.py", backtest_dir / "strategies" / "advanced_engine.py", update_imports=False)
    
    print_step(4, "创建统一的 __init__.py")
    
    # 创建统一的 __init__.py
    init_content = '''# -*- coding: utf-8 -*-
"""
统一回测框架模块 (Deprecated)

⚠️  此模块已迁移到新的统一结构：
  - src/core/backtest/

旧的导入方式会继续工作，但建议迁移到新结构。

请使用以下新导入方式：
  from src.core.backtest.core import StrategyBacktestEngine, Portfolio
  from src.core.backtest.factors import Factor, FactorCombination
  from src.core.backtest.optimization import GridSearchOptimizer
  from src.core.backtest.visualization import BacktestVisualizer
"""

import warnings
warnings.warn(
    "strategy_backtest 和 advanced_backtest 已合并为统一的 backtest 模块，"
    "请使用 src.core.backtest 替代。",
    DeprecationWarning,
    stacklevel=2
)

# 从新模块重新导出，保持向后兼容
try:
    from src.core.backtest.core import (
        StrategyBacktestEngine,
        Portfolio,
        Position,
        Trade,
        Order,
        OrderType,
        OrderStatus,
    )
    from src.core.backtest.core.data_access import TimeIsolatedDataProvider
    from src.core.backtest.metrics import PerformanceMetrics
    from src.core.backtest.visualization import BacktestVisualizer
    from src.core.backtest.orchestration import BacktestOrchestrator
    from src.core.backtest.sensitivity import ParameterSensitivityTest
    
    # 同时从 advanced_backtest 导入
    from src.core.backtest.factors import (
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
    from src.core.backtest.strategies import (
        MultiFactorStrategy,
        MultiFactorStrategyConfig,
        RebalanceFrequency,
        PositionSizingMethod,
    )
    from src.core.backtest.optimization import (
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
        # 来自 strategy_backtest
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
        
        # 来自 advanced_backtest
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
except ImportError as e:
    print(f"警告：部分模块导入失败: {e}")
    __all__ = []
'''
    
    # 同时为三个目录创建 __init__.py
    # 1. 新的 backtest 模块
    backtest_init = '''# -*- coding: utf-8 -*-
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
'''
    
    (backtest_dir / "__init__.py").write_text(backtest_init, encoding='utf-8')
    print(f"  ✓ 已创建: {backtest_dir / '__init__.py'}")
    
    # 2. 各个子目录的 __init__.py
    # core/__init__.py
    core_init = '''# -*- coding: utf-8 -*-
"""
回测核心引擎模块。
"""

from .engine import (
    StrategyBacktestEngine,
    Portfolio,
    Position,
    Trade,
    Order,
    OrderType,
    OrderStatus,
)
from .data_access import TimeIsolatedDataProvider

__all__ = [
    "StrategyBacktestEngine",
    "Portfolio",
    "Position",
    "Trade",
    "Order",
    "OrderType",
    "OrderStatus",
    "TimeIsolatedDataProvider",
]
'''
    (backtest_dir / "core" / "__init__.py").write_text(core_init, encoding='utf-8')
    
    # factors/__init__.py
    factors_init = '''# -*- coding: utf-8 -*-
"""
多因子框架模块。
"""

from .base import (
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
]
'''
    (backtest_dir / "factors" / "__init__.py").write_text(factors_init, encoding='utf-8')
    
    # strategies/__init__.py
    strategies_init = '''# -*- coding: utf-8 -*-
"""
策略实现模块。
"""

from .multi_factor import (
    MultiFactorStrategy,
    MultiFactorStrategyConfig,
    RebalanceFrequency,
    PositionSizingMethod,
)

__all__ = [
    "MultiFactorStrategy",
    "MultiFactorStrategyConfig",
    "RebalanceFrequency",
    "PositionSizingMethod",
]
'''
    (backtest_dir / "strategies" / "__init__.py").write_text(strategies_init, encoding='utf-8')
    
    # metrics/__init__.py
    metrics_init = '''# -*- coding: utf-8 -*-
"""
绩效指标模块。
"""

from .performance import PerformanceMetrics

__all__ = ["PerformanceMetrics"]
'''
    (backtest_dir / "metrics" / "__init__.py").write_text(metrics_init, encoding='utf-8')
    
    # optimization/__init__.py
    optimization_init = '''# -*- coding: utf-8 -*-
"""
参数优化模块。
"""

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
'''
    (backtest_dir / "optimization" / "__init__.py").write_text(optimization_init, encoding='utf-8')
    
    # visualization/__init__.py
    visualization_init = '''# -*- coding: utf-8 -*-
"""
可视化模块。
"""

from .core import BacktestVisualizer
from .advanced import Visualizer as AdvancedVisualizer

__all__ = ["BacktestVisualizer", "AdvancedVisualizer"]
'''
    (backtest_dir / "visualization" / "__init__.py").write_text(visualization_init, encoding='utf-8')
    
    # orchestration/__init__.py
    orchestration_init = '''# -*- coding: utf-8 -*-
"""
回测编排模块。
"""

from .orchestrator import BacktestOrchestrator
from .preloader import SmartDataPreloader

__all__ = ["BacktestOrchestrator", "SmartDataPreloader"]
'''
    (backtest_dir / "orchestration" / "__init__.py").write_text(orchestration_init, encoding='utf-8')
    
    # sensitivity/__init__.py
    sensitivity_init = '''# -*- coding: utf-8 -*-
"""
参数敏感性分析模块。
"""

from .analysis import ParameterSensitivityTest

__all__ = ["ParameterSensitivityTest"]
'''
    (backtest_dir / "sensitivity" / "__init__.py").write_text(sensitivity_init, encoding='utf-8')
    
    # reporting/__init__.py
    reporting_init = '''# -*- coding: utf-8 -*-
"""
报告生成模块。
"""

from .generator import BacktestReportGenerator

__all__ = ["BacktestReportGenerator"]
'''
    (backtest_dir / "reporting" / "__init__.py").write_text(reporting_init, encoding='utf-8')
    
    print_step(5, "标记旧模块为 @deprecated")
    
    # 更新 strategy_backtest/__init__.py
    old_strategy_init = strategy_backtest_dir / "__init__.py"
    if old_strategy_init.exists():
        content = old_strategy_init.read_text(encoding='utf-8')
        warning_comment = '''# -*- coding: utf-8 -*-
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

'''
        # 读取原内容，在前面添加警告
        original_content = old_strategy_init.read_text(encoding='utf-8')
        new_content = warning_comment + original_content
        old_strategy_init.write_text(new_content, encoding='utf-8')
        print(f"  ✓ 已更新: {old_strategy_init}")
    
    # 更新 advanced_backtest/__init__.py
    old_advanced_init = advanced_backtest_dir / "__init__.py"
    if old_advanced_init.exists():
        warning_comment = '''# -*- coding: utf-8 -*-
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

'''
        original_content = old_advanced_init.read_text(encoding='utf-8')
        new_content = warning_comment + original_content
        old_advanced_init.write_text(new_content, encoding='utf-8')
        print(f"  ✓ 已更新: {old_advanced_init}")
    
    print_step(6, "验证迁移结果")
    
    print("\n  新目录结构:")
    for item in sorted(backtest_dir.rglob("*")):
        if item.is_file() and item.name != "__pycache__":
            relative = item.relative_to(project_root)
            print(f"    {relative}")
    
    print_step(7, "完成！")
    
    print("\n" + "=" * 80)
    print("✓ 迁移完成！")
    print("=" * 80)
    print("\n新的统一回测框架位于: src/core/backtest/")
    print("\n导入示例:")
    print("  from src.core.backtest.core import StrategyBacktestEngine, Portfolio")
    print("  from src.core.backtest.factors import Factor, FactorCombination")
    print("  from src.core.backtest.optimization import GridSearchOptimizer")
    print("  from src.core.backtest.visualization import BacktestVisualizer")
    print("\n旧模块仍可使用，但会显示 DeprecationWarning")


if __name__ == "__main__":
    main()
