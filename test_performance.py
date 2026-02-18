#!/usr/bin/env python3
"""
性能测试脚本
测试策略评估的执行时间
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
from config import user_config


def test_strategy_evaluation():
    print("=" * 80)
    print("策略评估性能测试")
    print("=" * 80)
    
    try:
        # 获取股票池
        print("\n1. 获取股票池...")
        stock_count = 20  # 减少股票数量，加快测试
        stock_pool = stock_universe.get_stock_pool(
            pool_type='medium',
            size=stock_count
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
        
        ts_code_list = [ts_code for ts_code, _ in stock_pool[:5]]  # 只使用前5只股票
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
        
        # 测试单个股票的评估时间
        print("\n3. 测试单个股票评估时间...")
        test_stock_code, test_df = list(stock_data.items())[0]
        test_config = {
            'ma5_weight': 15,
            'ma5_ma10_weight': 10,
            'ma10_ma20_weight': 10,
            'macd_positive_weight': 20,
            'macd_cross_weight': 10,
            'rsi_low_weight': 15,
            'rsi_mid_weight': 10,
            'kdj_weight': 15,
            'entry_threshold': 70
        }
        
        start_time = time.time()
        result = strategy_optimizer.evaluate_strategy_on_stock(
            test_df, test_config, 'short'
        )
        single_stock_time = time.time() - start_time
        
        print(f"  单只股票评估时间: {single_stock_time:.2f} 秒")
        
        # 测试多只股票的评估时间
        print("\n4. 测试多只股票评估时间...")
        start_time = time.time()
        result_multi = strategy_optimizer.evaluate_strategy_on_multiple_stocks(
            stock_data, test_config, 'short'
        )
        multi_stock_time = time.time() - start_time
        
        print(f"  {len(stock_data)} 只股票评估时间: {multi_stock_time:.2f} 秒")
        print(f"  平均每只股票: {multi_stock_time / len(stock_data):.2f} 秒")
        
        # 估算完整优化时间
        print("\n5. 估算完整优化时间...")
        total_stocks = user_config.stock_count
        param_combinations = 19683  # 3^9
        
        estimated_per_combination = multi_stock_time * (total_stocks / len(stock_data))
        estimated_total = estimated_per_combination * param_combinations
        
        print(f"  配置股票数量: {total_stocks}")
        print(f"  参数组合数量: {param_combinations}")
        print(f"  估计每个参数组合时间: {estimated_per_combination:.2f} 秒")
        print(f"  估计总优化时间: {estimated_total:.2f} 秒")
        print(f"  估计总优化时间: {estimated_total / 60:.2f} 分钟")
        print(f"  估计总优化时间: {estimated_total / 3600:.2f} 小时")
        
        print("\n" + "=" * 80)
        print("性能测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_strategy_evaluation()
