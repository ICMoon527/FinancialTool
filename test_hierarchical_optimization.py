#!/usr/bin/env python3
"""
分层优化性能测试脚本
测试分层优化的性能改进
"""

import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.stock_universe import stock_universe
from data.data_fetcher import data_fetcher
from strategy.optimizer import strategy_optimizer


def test_hierarchical_optimization():
    print("=" * 80)
    print("分层优化性能测试")
    print("=" * 80)
    
    try:
        # 获取股票池
        print("\n1. 获取股票池...")
        optimize_stock_count = 10  # 减少股票数量，加快测试
        stock_pool = stock_universe.get_stock_pool(
            pool_type='medium',
            size=optimize_stock_count
        )
        
        if not stock_pool:
            print("错误: 无法获取股票池")
            return
        
        print(f"  获取到 {len(stock_pool)} 支股票")
        
        # 获取历史数据
        print("\n2. 获取历史数据...")
        data_days = 180  # 减少数据天数，加快测试
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=data_days + 60)).strftime('%Y%m%d')
        
        ts_code_list = [ts_code for ts_code, _ in stock_pool]
        print(f"  使用 {len(ts_code_list)} 支股票进行测试")
        
        batch_result = data_fetcher.fetch_stock_daily_batch(
            ts_code_list=ts_code_list,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        # 准备测试数据
        stock_data = {}
        for ts_code, df in batch_result.items():
            if not df.empty and len(df) >= 60:
                stock_data[ts_code] = df
        
        print(f"  成功加载 {len(stock_data)} 支股票数据")
        
        if not stock_data:
            print("错误: 没有足够的测试数据")
            return
        
        # 测试短期策略的分层优化
        print("\n3. 测试短期策略的分层优化...")
        short_param_ranges = {
            'ma5_weight': [10, 15, 20],
            'ma5_ma10_weight': [5, 10, 15],
            'ma10_ma20_weight': [5, 10, 15],
            'macd_positive_weight': [15, 20, 25],
            'macd_cross_weight': [5, 10, 15],
            'rsi_low_weight': [10, 15, 20],
            'rsi_mid_weight': [5, 10, 15],
            'kdj_weight': [10, 15, 20],
            'entry_threshold': [60, 70, 80]
        }
        
        start_time = time.time()
        
        best_params, best_perf = strategy_optimizer.grid_search_optimization(
            stock_data,
            short_param_ranges,
            horizon='short',
            objective='sharpe'
        )
        
        total_time = time.time() - start_time
        
        print(f"\n测试结果:")
        print(f"  总执行时间: {total_time:.2f} 秒")
        
        if best_perf:
            print(f"\n最优参数:")
            print(f"  {best_params}")
            print(f"\n最优参数表现:")
            print(f"  平均收益率: {best_perf['avg_return']*100:.2f}%")
            print(f"  平均胜率: {best_perf['avg_win_rate']*100:.2f}%")
            print(f"  平均盈亏比: {best_perf['avg_profit_factor']:.2f}")
            print(f"  夏普比率: {best_perf['sharpe_ratio']:.2f}")
        else:
            print(f"\n警告: 没有获得有效性能数据")
        
        # 计算参数组合数量
        total_combinations = 1
        for key, values in short_param_ranges.items():
            total_combinations *= len(values)
        
        # 计算分层优化的参数组合数量
        importance_groups = strategy_optimizer._get_importance_groups('short')
        hierarchical_combinations = 0
        for level_name, params in importance_groups.items():
            level_combinations = 1
            for param in params:
                if param in short_param_ranges:
                    level_combinations *= len(short_param_ranges[param])
            hierarchical_combinations += level_combinations
        
        print(f"\n性能对比:")
        print(f"  传统网格搜索参数组合: {total_combinations}")
        print(f"  分层优化参数组合: {hierarchical_combinations}")
        print(f"  计算量减少: {100 - (hierarchical_combinations/total_combinations)*100:.1f}%")
        print(f"  平均每个组合时间: {total_time/hierarchical_combinations:.2f} 秒")
        
        print("\n" + "=" * 80)
        print("分层优化测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_hierarchical_optimization()
