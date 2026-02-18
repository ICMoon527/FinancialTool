#!/usr/bin/env python3
"""
策略优化程序
使用历史数据迭代优化短、中、长期策略参数
"""

import sys
import os
from datetime import datetime, timedelta

from logger import log
from data.stock_universe import stock_universe
from data.data_fetcher import data_fetcher
from strategy.optimizer import strategy_optimizer
from strategy.strategy_config import strategy_config
from config import user_config
import pandas as pd


def main():
    print("\n" + "="*80)
    print("策略优化程序")
    print("="*80)
    
    try:
        print("\n正在准备优化数据...")
        
        # 让策略优化的股票数量读取配置文件，与主程序保持一致
        optimize_stock_count = user_config.stock_count
        data_days = user_config.data_days
        
        print(f"\n优化配置:")
        print(f"  策略优化股票数量: {optimize_stock_count}")
        print(f"  数据天数: {data_days}")
        
        print(f"\n正在获取股票池...")
        stock_pool = stock_universe.get_stock_pool(
            pool_type='medium',
            size=optimize_stock_count
        )
        
        if not stock_pool:
            print("\n错误: 无法获取股票池")
            return
        
        print(f"  获取到 {len(stock_pool)} 支股票")
        
        print(f"\n正在获取历史数据（{data_days}天）...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=data_days + 100)).strftime('%Y%m%d')
        
        ts_code_list = [ts_code for ts_code, _ in stock_pool]
        name_map = {ts_code: name for ts_code, name in stock_pool}
        
        batch_result = data_fetcher.fetch_stock_daily_batch(
            ts_code_list=ts_code_list,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        stock_data = {}
        success_count = 0
        total = len(batch_result)
        
        from utils.progress_bar import create_progress_bar
        pb = create_progress_bar(total, '处理优化数据')
        
        for i, (ts_code, df) in enumerate(batch_result.items()):
            if not df.empty and len(df) >= 120:
                df = df.copy()
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
                if 'vol' in df.columns and 'volume' not in df.columns:
                    df['volume'] = df['vol']
                
                df['name'] = name_map.get(ts_code, ts_code)
                stock_data[ts_code] = df
                success_count += 1
            pb.update(i + 1)
        
        print(f"  成功加载 {success_count}/{len(stock_pool)} 支股票数据")
        
        if success_count == 0:
            print("\n错误: 没有足够的历史数据")
            return
        
        print(f"\n开始优化所有周期策略...")
        import time
        start_time = time.time()
        
        # 完整的参数范围，进行全面优化
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
        
        # 估算分层优化时间
        estimated_per_stock = 0.5  # 每只股票的估算时间（秒）
        estimated_per_combination = estimated_per_stock * len(stock_data)
        
        # 计算分层优化的参数组合数量
        from strategy.optimizer import strategy_optimizer
        importance_groups = strategy_optimizer._get_importance_groups('short')
        hierarchical_combinations = 0
        level_combinations = {}
        
        for level_name, params in importance_groups.items():
            level_count = 1
            for param in params:
                if param in short_param_ranges:
                    level_count *= len(short_param_ranges[param])
            level_combinations[level_name] = level_count
            hierarchical_combinations += level_count
        
        # 计算传统网格搜索的参数组合数量
        total_combinations = 1
        for key, values in short_param_ranges.items():
            total_combinations *= len(values)
        
        # 计算分层优化的预估时间
        estimated_total = estimated_per_combination * hierarchical_combinations
        
        print(f"\n时间预估:")
        print(f"  传统网格搜索参数组合: {total_combinations}")
        print(f"  分层优化参数组合: {hierarchical_combinations}")
        print(f"  计算量减少: {100 - (hierarchical_combinations/total_combinations)*100:.1f}%")
        print(f"  每只股票评估时间: ~{estimated_per_stock:.1f}秒")
        print(f"  每个参数组合时间: ~{estimated_per_combination:.1f}秒")
        
        print(f"\n层级预估时间:")
        for level_name, count in level_combinations.items():
            level_time = estimated_per_combination * count
            print(f"  {level_name}: ~{level_time/60:.1f}分钟 ({count}个组合)")
        
        print(f"\n总优化时间: ~{estimated_total/60:.1f}分钟 ({estimated_total/3600:.1f}小时)")
        print(f"  注: 实际时间可能因硬件性能而异")
        
        print(f"\n使用完整参数范围进行优化...")
        param_combinations = 3 ** 9
        print(f"参数组合数量: {param_combinations}")
        
        print(f"\n开始优化所有周期策略...")
        results = strategy_optimizer.optimize_all_horizons(stock_data)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n所有周期策略优化完成！")
        print(f"总执行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
        
        # 打印各周期的最优参数
        for horizon, result in results.items():
            if result and result.get('best_params'):
                print(f"\n{horizon} 周期最优参数:")
                print(f"  {result['best_params']}")
                if result.get('best_performance'):
                    perf = result['best_performance']
                    print(f"  表现:")
                    print(f"    平均收益率: {perf['avg_return']*100:.2f}%")
                    print(f"    平均胜率: {perf['avg_win_rate']*100:.2f}%")
                    print(f"    平均盈亏比: {perf['avg_profit_factor']:.2f}")
                    print(f"    夏普比率: {perf['sharpe_ratio']:.2f}")
        
        print(f"\n保存优化结果...")
        strategy_config.save_optimization_results(results)
        
        print(f"\n" + "="*80)
        print("策略优化完成！")
        print("="*80)
        print(f"\n优化结果已保存到: strategy_config.json")
        print(f"\n下次运行推荐程序时，将使用优化后的参数。")
        
    except Exception as e:
        print(f"\n错误: {e}")
        log.error(f"策略优化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import pandas as pd
    main()
