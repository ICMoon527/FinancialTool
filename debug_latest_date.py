#!/usr/bin/env python3
"""
调试数据源最新交易日获取
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

print("="*80)
print("调试数据源最新交易日获取")
print("="*80)

print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from data.data_fetcher import data_fetcher, HAS_AKSHARE, HAS_TUSHARE
    print(f"\n数据源状态:")
    print(f"  AKShare: {HAS_AKSHARE}")
    print(f"  Tushare: {data_fetcher.ts_pro is not None}")
    
    print("\n" + "-"*80)
    print("测试1: 直接用Tushare获取平安银行数据")
    print("-"*80)
    
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    print(f"查询范围: {start_date} - {end_date}")
    
    if data_fetcher.ts_pro:
        try:
            df = data_fetcher.ts_pro.daily(ts_code='000001.SZ', start_date=start_date, end_date=end_date)
            if not df.empty:
                print(f"✓ 获取到 {len(df)} 条数据")
                print(f"  最新日期: {df['trade_date'].iloc[0]}")
                print(f"  最早日期: {df['trade_date'].iloc[-1]}")
                print(f"\n前5条数据:")
                print(df.head())
            else:
                print("✗ 数据为空")
        except Exception as e:
            print(f"✗ Tushare获取失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ Tushare不可用")
    
    print("\n" + "-"*80)
    print("测试2: 直接用AKShare获取平安银行数据")
    print("-"*80)
    
    if HAS_AKSHARE:
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol='000001', period='daily', 
                                     start_date=start_date, end_date=end_date, adjust='qfq')
            if not df.empty:
                print(f"✓ 获取到 {len(df)} 条数据")
                print(f"  最新日期: {pd.to_datetime(df['日期'].iloc[-1]).strftime('%Y%m%d')}")
                print(f"  最早日期: {pd.to_datetime(df['日期'].iloc[0]).strftime('%Y%m%d')}")
                print(f"\n最后5条数据:")
                print(df.tail())
            else:
                print("✗ 数据为空")
        except Exception as e:
            print(f"✗ AKShare获取失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ AKShare不可用")
    
    print("\n" + "-"*80)
    print("测试3: 调用get_available_latest_trading_day()")
    print("-"*80)
    
    result = data_fetcher.get_available_latest_trading_day()
    print(f"返回结果: {result}")
    
    print("\n" + "-"*80)
    print("测试4: 获取数据库最新交易日")
    print("-"*80)
    
    db_result = data_fetcher.get_latest_trading_day()
    print(f"返回结果: {db_result}")
    
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)
    
except Exception as e:
    print(f"\n发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
