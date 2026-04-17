# -*- coding: utf-8 -*-
"""
参数优化模块测试脚本
"""
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.core.advanced_backtest import (
    ParameterRange,
    OptimizationMethod,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    OverfitDetector,
    create_optimizer,
)


def test_parameter_range():
    """测试参数范围定义"""
    print("=" * 80)
    print("测试 1: 参数范围定义")
    print("=" * 80)
    
    # 整数参数
    int_param = ParameterRange(
        name="lookback_period",
        param_type="int",
        min_value=10,
        max_value=30,
        step=5,
        description="回看周期",
    )
    print(f"✅ 整数参数创建成功: {int_param.name}")
    print(f"   范围: {int_param.min_value} - {int_param.max_value}")
    print(f"   步长: {int_param.step}")
    
    # 浮点数参数
    float_param = ParameterRange(
        name="stop_loss_pct",
        param_type="float",
        min_value=0.02,
        max_value=0.10,
        step=0.02,
        description="止损比例",
    )
    print(f"\n✅ 浮点数参数创建成功: {float_param.name}")
    print(f"   范围: {float_param.min_value} - {float_param.max_value}")
    
    # 分类参数
    cat_param = ParameterRange(
        name="position_sizing",
        param_type="categorical",
        values=["equal_weight", "risk_parity"],
        description="仓位管理方法",
    )
    print(f"\n✅ 分类参数创建成功: {cat_param.name}")
    print(f"   可选值: {cat_param.values}")
    
    print("\n✅ 参数范围定义测试通过!")


def create_test_objective_function(params: dict) -> dict:
    """
    测试用目标函数
    
    一个简单的模拟函数，用于测试优化器
    
    Args:
        params: 参数字典
    
    Returns:
        指标字典
    """
    lookback = params.get("lookback_period", 20)
    stop_loss = params.get("stop_loss_pct", 0.05)
    position_method = params.get("position_sizing", "equal_weight")
    
    # 模拟夏普比率
    # 假设最佳参数在中间值附近
    base_sharpe = 1.5
    
    # 距离最佳值越近，夏普比率越高
    lookback_error = abs(lookback - 20)
    stop_loss_error = abs(stop_loss - 0.05)
    
    noise = np.random.normal(0, 0.1)
    
    sharpe = base_sharpe - lookback_error * 0.05 - stop_loss_error * 5 + noise
    
    # 其他指标
    total_return = 0.15 + (sharpe - 1.0) * 0.1
    max_drawdown = -0.10 - (sharpe - 1.0) * 0.02
    win_rate = 0.55 + (sharpe - 1.0) * 0.05
    
    return {
        "sharpe_ratio": max(0.1, sharpe),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": min(win_rate, 0.9),
    }


def test_grid_search():
    """测试网格搜索优化"""
    print("\n" + "=" * 80)
    print("测试 2: 网格搜索优化")
    print("=" * 80)
    
    # 定义参数范围
    param_ranges = [
        ParameterRange(
            name="lookback_period",
            param_type="int",
            min_value=10,
            max_value=30,
            step=10,
        ),
        ParameterRange(
            name="stop_loss_pct",
            param_type="float",
            min_value=0.03,
            max_value=0.07,
            step=0.02,
        ),
        ParameterRange(
            name="position_sizing",
            param_type="categorical",
            values=["equal_weight", "risk_parity"],
        ),
    ]
    
    # 创建网格搜索优化器
    optimizer = GridSearchOptimizer(
        objective_function=create_test_objective_function,
        param_ranges=param_ranges,
        metric_name="sharpe_ratio",
        maximize=True,
    )
    
    # 运行优化
    print("开始网格搜索优化...")
    result = optimizer.optimize()
    
    print(f"\n✅ 网格搜索完成!")
    print(f"   总试验次数: {result.total_trials}")
    print(f"   耗时: {result.duration_seconds:.2f} 秒")
    print(f"   最佳参数: {result.best_params}")
    print(f"   最佳夏普比率: {result.best_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"   最佳总收益率: {result.best_metrics.get('total_return', 0):.2%}")
    
    print("\n✅ 网格搜索测试通过!")


def test_random_search():
    """测试随机搜索优化"""
    print("\n" + "=" * 80)
    print("测试 3: 随机搜索优化")
    print("=" * 80)
    
    # 定义参数范围
    param_ranges = [
        ParameterRange(
            name="lookback_period",
            param_type="int",
            min_value=10,
            max_value=30,
        ),
        ParameterRange(
            name="stop_loss_pct",
            param_type="float",
            min_value=0.02,
            max_value=0.10,
        ),
        ParameterRange(
            name="position_sizing",
            param_type="categorical",
            values=["equal_weight", "risk_parity"],
        ),
    ]
    
    # 创建随机搜索优化器
    optimizer = RandomSearchOptimizer(
        objective_function=create_test_objective_function,
        param_ranges=param_ranges,
        metric_name="sharpe_ratio",
        maximize=True,
        random_seed=42,
    )
    
    # 运行优化
    print("开始随机搜索优化...")
    result = optimizer.optimize(max_trials=20)
    
    print(f"\n✅ 随机搜索完成!")
    print(f"   总试验次数: {result.total_trials}")
    print(f"   耗时: {result.duration_seconds:.2f} 秒")
    print(f"   最佳参数: {result.best_params}")
    print(f"   最佳夏普比率: {result.best_metrics.get('sharpe_ratio', 0):.4f}")
    
    print("\n✅ 随机搜索测试通过!")


def test_factory_function():
    """测试工厂函数"""
    print("\n" + "=" * 80)
    print("测试 4: 工厂函数")
    print("=" * 80)
    
    # 定义参数范围
    param_ranges = [
        ParameterRange(
            name="lookback_period",
            param_type="int",
            min_value=10,
            max_value=30,
        ),
    ]
    
    # 使用工厂函数创建网格搜索优化器
    optimizer1 = create_optimizer(
        method=OptimizationMethod.GRID_SEARCH,
        objective_function=create_test_objective_function,
        param_ranges=param_ranges,
        metric_name="sharpe_ratio",
        maximize=True,
    )
    
    print(f"✅ 工厂函数创建网格搜索优化器成功: {type(optimizer1).__name__}")
    
    # 使用工厂函数创建随机搜索优化器
    optimizer2 = create_optimizer(
        method=OptimizationMethod.RANDOM_SEARCH,
        objective_function=create_test_objective_function,
        param_ranges=param_ranges,
        metric_name="sharpe_ratio",
        maximize=True,
        random_seed=123,
    )
    
    print(f"✅ 工厂函数创建随机搜索优化器成功: {type(optimizer2).__name__}")
    
    print("\n✅ 工厂函数测试通过!")


def test_overfit_detector():
    """测试过拟合检测器"""
    print("\n" + "=" * 80)
    print("测试 5: 过拟合检测器")
    print("=" * 80)
    
    # 创建过拟合检测器
    detector = OverfitDetector(
        metric_name="sharpe_ratio",
        threshold=0.3,
    )
    
    # 测试 1: 不过拟合的情况
    train_metrics_good = {
        "sharpe_ratio": 1.5,
        "total_return": 0.20,
    }
    test_metrics_good = {
        "sharpe_ratio": 1.4,
        "total_return": 0.18,
    }
    
    is_overfit, score = detector.check_train_test(
        train_metrics_good, test_metrics_good)
    
    print(f"测试场景 1:")
    print(f"   训练夏普比率: {train_metrics_good['sharpe_ratio']}")
    print(f"   测试夏普比率: {test_metrics_good['sharpe_ratio']}")
    print(f"   过拟合分数: {score:.4f}")
    print(f"   是否过拟合: {is_overfit}")
    assert not is_overfit, "不应该检测到过拟合"
    
    # 测试 2: 过拟合的情况
    train_metrics_bad = {
        "sharpe_ratio": 2.0,
        "total_return": 0.30,
    }
    test_metrics_bad = {
        "sharpe_ratio": 1.0,
        "total_return": 0.10,
    }
    
    is_overfit, score = detector.check_train_test(
        train_metrics_bad, test_metrics_bad)
    
    print(f"\n测试场景 2:")
    print(f"   训练夏普比率: {train_metrics_bad['sharpe_ratio']}")
    print(f"   测试夏普比率: {test_metrics_bad['sharpe_ratio']}")
    print(f"   过拟合分数: {score:.4f}")
    print(f"   是否过拟合: {is_overfit}")
    assert is_overfit, "应该检测到过拟合"
    
    print("\n✅ 过拟合检测器测试通过!")


def main():
    """主函数"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "参数优化模块测试" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_parameter_range()
        test_grid_search()
        test_random_search()
        test_factory_function()
        test_overfit_detector()
        
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "✅ 所有测试通过!" + " " * 40 + "║")
        print("╚" + "=" * 78 + "╝")
        
        print("\n📝 参数优化模块功能:")
        print("   - 参数范围定义（整数、浮点数、分类")
        print("   - 网格搜索优化")
        print("   - 随机搜索优化")
        print("   - 过拟合检测（训练测试对比）")
        print("   - 工厂函数创建优化器")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
