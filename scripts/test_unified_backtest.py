# -*- coding: utf-8 -*-
"""
测试统一回测框架的导入和基本功能
"""
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("测试 1: 从新的统一模块导入")
print("=" * 80)

try:
    print("\n1.1 测试 core 模块导入...")
    from src.core.backtest.core import (
        StrategyBacktestEngine,
        Portfolio,
        Position,
        Trade,
        Order,
        OrderType,
        OrderStatus,
        TimeIsolatedDataProvider,
    )
    print("  ✓ core 模块导入成功")
    
    print("\n1.2 测试 factors 模块导入...")
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
    print("  ✓ factors 模块导入成功")
    
    print("\n1.3 测试 strategies 模块导入...")
    from src.core.backtest.strategies import (
        MultiFactorStrategy,
        MultiFactorStrategyConfig,
        RebalanceFrequency,
        PositionSizingMethod,
    )
    print("  ✓ strategies 模块导入成功")
    
    print("\n1.4 测试 metrics 模块导入...")
    from src.core.backtest.metrics import PerformanceMetrics
    print("  ✓ metrics 模块导入成功")
    
    print("\n1.5 测试 optimization 模块导入...")
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
    print("  ✓ optimization 模块导入成功")
    
    print("\n1.6 测试 visualization 模块导入...")
    from src.core.backtest.visualization import (
        BacktestVisualizer,
    )
    print("  ✓ visualization 模块导入成功")
    
    print("\n1.7 测试 orchestration 模块导入...")
    from src.core.backtest.orchestration import (
        BacktestOrchestrator,
        SmartDataPreloader,
    )
    print("  ✓ orchestration 模块导入成功")
    
    print("\n1.8 测试 sensitivity 模块导入...")
    from src.core.backtest.sensitivity import ParameterSensitivityTest
    print("  ✓ sensitivity 模块导入成功")
    
    print("\n1.9 测试 reporting 模块导入...")
    from src.core.backtest.reporting import BacktestReportGenerator
    print("  ✓ reporting 模块导入成功")
    
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 2: 从旧模块导入（向后兼容）")
print("=" * 80)

try:
    print("\n2.1 测试 strategy_backtest 模块导入...")
    from src.core.strategy_backtest import (
        StrategyBacktestEngine as OldStrategyEngine,
        Portfolio as OldPortfolio,
        PerformanceMetrics as OldMetrics,
    )
    print("  ✓ strategy_backtest 模块导入成功（有 DeprecationWarning 是正常的）")
    
    print("\n2.2 测试 advanced_backtest 模块导入...")
    from src.core.advanced_backtest import (
        Factor as OldFactor,
        GridSearchOptimizer as OldGridSearch,
    )
    print("  ✓ advanced_backtest 模块导入成功（有 DeprecationWarning 是正常的）")
    
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("测试 3: 基本功能测试")
print("=" * 80)

try:
    print("\n3.1 测试 Portfolio 初始化...")
    portfolio = Portfolio(initial_capital=1000000.0)
    print(f"  ✓ Portfolio 创建成功: 初始资金 = {portfolio.initial_capital:,.2f}")
    print(f"    当前现金 = {portfolio.cash:,.2f}")
    
    print("\n3.2 测试 PerformanceMetrics...")
    metrics = PerformanceMetrics(portfolio)
    print("  ✓ PerformanceMetrics 创建成功")
    
    print("\n3.3 测试 FactorConfig...")
    factor_config = FactorConfig(
        name="test_factor",
        factor_type=FactorType.TECHNICAL,
        direction=FactorDirection.LONG,
    )
    print(f"  ✓ FactorConfig 创建成功: {factor_config.name}")
    
    print("\n3.4 测试 ParameterRange...")
    param_range = ParameterRange(
        name="lookback",
        param_type="int",
        min_value=10,
        max_value=30,
        step=5,
    )
    print(f"  ✓ ParameterRange 创建成功: {param_range.name}")
    
except Exception as e:
    print(f"  ✗ 功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)

print("\n📊 统一回测框架结构:")
print("  src/core/backtest/")
print("  ├── core/")
print("  │   ├── engine.py           (回测引擎、组合管理)")
print("  │   └── data_access.py      (时间隔离数据提供器)")
print("  ├── factors/")
print("  │   └── base.py             (因子定义、因子组合)")
print("  ├── strategies/")
print("  │   └── multi_factor.py     (多因子策略)")
print("  ├── metrics/")
print("  │   └── performance.py      (绩效指标计算)")
print("  ├── optimization/")
print("  │   └── optimization.py     (参数优化、过拟合检测)")
print("  ├── visualization/")
print("  │   ├── core.py             (基础可视化)")
print("  │   └── advanced.py         (高级可视化)")
print("  ├── orchestration/")
print("  │   └── orchestrator.py     (回测编排器)")
print("  ├── sensitivity/")
print("  │   └── analysis.py         (参数敏感性分析)")
print("  └── reporting/")
print("      └── generator.py        (报告生成)")
