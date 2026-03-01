#!/usr/bin/env python3
"""
单周期策略优化模块
支持对单个周期（短/中/长）进行策略优化
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


def print_optimization_method_menu():
    """打印优化方法选择菜单"""
    print("\n" + "="*60)
    print("请选择优化方法:")
    print("  1. 增强型随机搜索（推荐 - 带早停、并行、小数支持）")
    print("  2. 分层优化（平衡效率和效果）")
    print("  0. 返回")
    print("="*60)

def run_single_horizon_optimization(horizon: str = 'short'):
    """
    运行单个周期的策略优化
    
    Args:
        horizon: 周期 ('short', 'medium', 'long')
    """
    horizon_names = {
        'short': '短线',
        'medium': '中期',
        'long': '长期'
    }
    
    horizon_name = horizon_names.get(horizon, '未知')
    
    # 如果是短线策略，让用户选择优化方法
    optimization_method = 'enhanced_random'
    if horizon == 'short':
        while True:
            print_optimization_method_menu()
            method_choice = input("\n请选择 (0-2): ").strip()
            
            if method_choice == '0':
                return
            elif method_choice == '1':
                optimization_method = 'enhanced_random'
                break
            elif method_choice == '2':
                optimization_method = 'hierarchical'
                break
            else:
                print("\n无效选项")
    
    print(f"\n" + "="*80)
    print(f"{horizon_name}策略优化程序")
    print("="*80)
    
    try:
        print(f"\n正在准备优化数据...")
        
        optimize_stock_count = user_config.stock_count
        data_days = user_config.data_days
        
        print(f"\n优化配置:")
        print(f"  优化周期: {horizon_name}")
        print(f"  优化方法: {optimization_method if horizon == 'short' else '分层优化'}")
        print(f"  策略优化股票数量: {optimize_stock_count}")
        print(f"  数据天数: {data_days}")
        
        print(f"\n正在获取股票池...")
        stock_pool = stock_universe.get_stock_pool(
            pool_type='medium',
            size=optimize_stock_count
        )
        
        if not stock_pool:
            print(f"\n错误: 无法获取股票池")
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
            print(f"\n错误: 没有足够的历史数据")
            return
        
        print(f"\n开始优化{horizon_name}策略...")
        import time
        start_time = time.time()
        
        # 获取参数范围
        param_ranges = strategy_config.get_param_ranges(horizon)
        
        print(f"\n使用参数范围进行优化...")
        print(f"  参数: {list(param_ranges.keys())}")
        
        # 运行优化
        if horizon == 'short':
            from strategy.advanced_optimizer import advanced_optimizer
            best_params, best_perf = advanced_optimizer.optimize_short_term_with_multiple_methods(
                stock_data,
                method=optimization_method
            )
        else:
            # 对于中长线，使用旧的参数范围格式
            old_param_ranges = {}
            if horizon == 'medium':
                old_param_ranges = {
                    'ma20_weight': [5, 10, 15],
                    'ma20_ma60_weight': [10, 15, 20],
                    'macd_weight': [10, 15, 20],
                    'trend_weight': [15, 20, 25],
                    'obv_weight': [10, 15, 20],
                    'volatility_weight': [10, 15, 20],
                    'entry_threshold': [60, 70, 80]
                }
            elif horizon == 'long':
                old_param_ranges = {
                    'ma20_weight': [5, 10, 15],
                    'ma20_ma60_weight': [10, 15, 20],
                    'ma60_ma120_weight': [15, 20, 25],
                    'trend_weight': [20, 25, 30],
                    'near_high_weight': [10, 15, 20],
                    'obv_weight': [10, 15, 20],
                    'entry_threshold': [60, 70, 80]
                }
            
            best_params, best_perf = strategy_optimizer.grid_search_optimization(
                stock_data,
                old_param_ranges,
                horizon=horizon,
                objective='sharpe'
            )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 如果不是增强型随机搜索，打印结果并保存配置
        if horizon != 'short' or optimization_method != 'enhanced_random':
            print(f"\n{horizon_name}策略优化完成！")
            print(f"执行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
            
            # 打印最优参数
            print(f"\n{horizon_name}周期最优参数:")
            print(f"  {best_params}")
            if best_perf:
                print(f"\n表现:")
                print(f"  平均收益率: {best_perf['avg_return']*100:.2f}%")
                print(f"  平均胜率: {best_perf['avg_win_rate']*100:.2f}%")
                print(f"  平均盈亏比: {best_perf['avg_profit_factor']:.2f}")
                print(f"  夏普比率: {best_perf['sharpe_ratio']:.2f}")
            
            print(f"\n保存优化结果...")
            
            # 更新配置
            if best_params:
                weights = {k: v for k, v in best_params.items() if k != 'entry_threshold'}
                strategy_config.update_weights(horizon, weights)
                if 'entry_threshold' in best_params:
                    strategy_config.update_entry_threshold(horizon, best_params['entry_threshold'])
            
            print(f"\n" + "="*80)
            print(f"{horizon_name}策略优化完成！")
            print("="*80)
            print(f"\n优化结果已保存到: strategy_config.json")
            print(f"\n下次运行推荐程序时，将使用优化后的参数。")
        
    except Exception as e:
        print(f"\n错误: {e}")
        log.error(f"{horizon_name}策略优化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
