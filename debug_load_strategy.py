
# -*- coding: utf-8 -*-
"""
调试脚本：测试主力建仓策略是否能被正确加载
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from stock_selector.strategies.python_strategy_loader import load_python_strategies_from_dir, get_registered_strategies

print("=" * 80)
print("开始测试策略加载")
print("=" * 80)

# 策略目录
strategy_dir = Path("stock_selector/strategies/python")

print(f"\n策略目录: {strategy_dir.absolute()}")
print(f"目录存在: {strategy_dir.exists()}")
print(f"是目录: {strategy_dir.is_dir()}")

if strategy_dir.exists() and strategy_dir.is_dir():
    print(f"\n目录下的文件:")
    for f in strategy_dir.glob("*.py"):
        print(f"  - {f.name}")

    print("\n" + "=" * 80)
    print("尝试加载策略...")
    print("=" * 80)

    strategies = load_python_strategies_from_dir(strategy_dir)
    
    print(f"\n成功加载的策略数量: {len(strategies)}")
    print("\n策略列表:")
    for s in strategies:
        print(f"  - ID: {s.id}")
        print(f"    Display Name: {s.display_name}")
        print(f"    Type: {s.metadata.strategy_type}")
        print(f"    Category: {s.metadata.category}")
        print()

    print("=" * 80)
    print("已注册的策略字典:")
    print("=" * 80)
    registered = get_registered_strategies()
    for strategy_id, strategy_class in registered.items():
        print(f"  - {strategy_id}: {strategy_class.__name__}")

    # 检查我们的策略是否在其中
    our_strategy_found = any(s.id == "main_capital_position" for s in strategies)
    print(f"\n主力建仓策略是否加载: {'✓' if our_strategy_found else '✗'}")
    
    if not our_strategy_found:
        print("\n尝试单独加载主力建仓策略文件...")
        strategy_file = strategy_dir / "main_capital_position_strategy.py"
        if strategy_file.exists():
            print(f"策略文件存在: {strategy_file}")
            
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("main_capital_position_strategy", strategy_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    print("执行模块...")
                    spec.loader.exec_module(module)
                    
                    # 检查模块内容
                    print(f"\n模块中的内容: {dir(module)}")
                    
                    if hasattr(module, "MainCapitalPositionStrategy"):
                        print("✓ 找到策略类 MainCapitalPositionStrategy")
                        strategy_class = module.MainCapitalPositionStrategy
                        strategy = strategy_class()
                        print(f"✓ 成功实例化策略: {strategy.display_name} (ID: {strategy.id})")
                    else:
                        print("✗ 找不到策略类 MainCapitalPositionStrategy")
            except Exception as e:
                print(f"✗ 加载失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ 策略文件不存在")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

