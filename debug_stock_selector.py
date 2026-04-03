# -*- coding: utf-8 -*-
"""
调试选股器 API
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    print("=" * 60)
    print("测试 1: 导入配置")
    print("=" * 60)
    
    from api.v1.schemas.stock_selector import StockSelectorConfigResponse
    from stock_selector.config import get_config
    
    print("✓ 导入成功")
    
    config = get_config()
    print(f"✓ 配置读取成功，default_top_n = {config.default_top_n}")
    
    response = StockSelectorConfigResponse(
        success=True,
        default_top_n=config.default_top_n,
    )
    
    print(f"✓ Response 创建成功: {response.model_dump()}")
    print(f"✓ Response JSON: {response.model_dump_json()}")
    
    print("\n" + "=" * 60)
    print("测试 2: 导入策略相关")
    print("=" * 60)
    
    from stock_selector import StockSelectorService
    
    print("✓ StockSelectorService 导入成功")
    
    service = StockSelectorService()
    print("✓ StockSelectorService 初始化成功")
    
    strategies = service.get_available_strategies()
    print(f"✓ 获取到 {len(strategies)} 个策略")
    
    for i, s in enumerate(strategies[:3]):
        print(f"  策略 {i+1}: {s.display_name} ({s.id})")
    
    active_ids = service.get_active_strategy_ids()
    print(f"✓ 激活的策略数: {len(active_ids)}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
