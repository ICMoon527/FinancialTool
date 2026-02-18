#!/usr/bin/env python3
"""
策略优化性能测试脚本
测试优化后的性能改进
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


def test_optimization_performance():
    print("=" * 80)
    print("策略优化性能测试")
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
        
        # 测试参数组合
        print("\n3. 测试参数组合排序...")
        test_param_ranges = {
            'ma5_weight': [10, 15, 20],
            'ma5_ma10_weight': [5, 10],  # 减少参数值，加快测试
            'ma10_ma20_weight': [5, 10],
            'macd_positive_weight': [15, 20],
            'entry_threshold': [60, 70]
        }
        
        # 测试单个参数组合的执行时间
        print("\n4. 测试单个参数组合执行时间...")
        test_params = {
            'ma5_weight': 15,
            'ma5_ma10_weight': 10,
            'ma10_ma20_weight': 10,
            'macd_positive_weight': 20,
            'entry_threshold': 70
        }
        
        start_time = time.time()
        result = strategy_optimizer.evaluate_strategy_on_multiple_stocks(
            stock_data, test_params, 'short'
        )
        single_combination_time = time.time() - start_time
        
        print(f"  单个参数组合执行时间: {single_combination_time:.2f} 秒")
        print(f"  每只股票平均时间: {single_combination_time/len(stock_data):.3f} 秒")
        
        # 测试参数组合排序和评估
        print("\n5. 测试参数组合排序和评估...")
        start_time = time.time()
        
        best_params, best_perf = strategy_optimizer.grid_search_optimization(
            stock_data,
            test_param_ranges,
            horizon='short',
            objective='sharpe'
        )
        
        total_time = time.time() - start_time
        
        # 计算参数组合数量
        param_combinations = 1
        for key, values in test_param_ranges.items():
            param_combinations *= len(values)
        
        print(f"\n测试结果:")
        print(f"  参数组合数量: {param_combinations}")
        print(f"  总执行时间: {total_time:.2f} 秒")
        print(f"  平均每个组合时间: {total_time/param_combinations:.2f} 秒")
        
        if best_perf:
            print(f"\n最优参数表现:")
            print(f"  平均收益率: {best_perf['avg_return']*100:.2f}%")
            print(f"  平均胜率: {best_perf['avg_win_rate']*100:.2f}%")
            print(f"  夏普比率: {best_perf['sharpe_ratio']:.2f}")
        
        # 估算完整优化时间
        print(f"\n6. 估算完整优化时间...")
        full_stock_count = 25
        full_param_combinations = 19683  # 3^9
        
        estimated_per_stock = single_combination_time / len(stock_data)
        estimated_full_per_combination = estimated_per_stock * full_stock_count
        estimated_full_total = estimated_full_per_combination * full_param_combinations
        
        print(f"  完整优化配置:")
        print(f"    股票数量: {full_stock_count}")
        print(f"    参数组合数量: {full_param_combinations}")
        print(f"  估算时间:")
        print(f"    每个参数组合: ~{estimated_full_per_combination:.1f}秒")
        print(f"    总优化时间: ~{estimated_full_total/60:.1f}分钟 ({estimated_full_total/3600:.1f}小时)")
        
        print("\n" + "=" * 80)
        print("性能测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_optimization_performance()
