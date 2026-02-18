#!/usr/bin/env python3
"""
测试数据源最新交易日获取
"""

import pandas as pd
from datetime import datetime
from data.data_fetcher import data_fetcher

print("="*80)
print("测试数据源最新交易日获取")
print("="*80)

print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n正在获取数据源最新交易日...")
available_latest = data_fetcher.get_available_latest_trading_day()
print(f"数据源最新交易日: {available_latest}")

print("\n正在获取数据库最新交易日...")
db_latest = data_fetcher.get_latest_trading_day()
print(f"数据库最新交易日: {db_latest}")

print(f"\n对比结果:")
if available_latest == db_latest:
    print("  ✓ 数据源和数据库最新交易日一致，无需更新")
else:
    print("  ✗ 数据源和数据库最新交易日不一致，需要更新")

print("\n" + "="*80)
print("测试完成")
print("="*80)
