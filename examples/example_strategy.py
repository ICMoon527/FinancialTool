import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import data_fetcher
from strategy.factors import factor_library
from strategy.signals import signal_generator
from backtest.engine import backtest_engine
from backtest.analytics import performance_analyzer
from logger import log

def run_ma_cross_strategy(ts_code: str = '000001.SZ', 
                          start_date: str = '20200101',
                          end_date: str = '20250101',
                          short_period: int = 5,
                          long_period: int = 20):
    log.info(f"开始运行均线交叉策略回测: {ts_code}")
    
    df = data_fetcher.fetch_stock_daily(ts_code, start_date, end_date)
    
    if df.empty:
        log.error("获取数据失败")
        return None
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    df = factor_library.calculate_ma(df, periods=[short_period, long_period])
    
    df = signal_generator.ma_cross_signal(df, short_period=short_period, long_period=long_period)
    
    backtest_engine.reset()
    results = backtest_engine.run(df, ts_code=ts_code)
    
    print("\n" + performance_analyzer.generate_report(results))
    
    return results, df

def run_multi_strategy_comparison(ts_code: str = '000001.SZ',
                                   start_date: str = '20200101',
                                   end_date: str = '20250101'):
    log.info(f"开始多策略对比回测: {ts_code}")
    
    df = data_fetcher.fetch_stock_daily(ts_code, start_date, end_date)
    
    if df.empty:
        log.error("获取数据失败")
        return None
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    df = factor_library.calculate_all_factors(df)
    
    strategies = ['ma_cross', 'macd', 'rsi', 'bollinger', 'kdj', 'dual_thrust', 'turtle']
    results_dict = {}
    
    for strategy_name in strategies:
        try:
            log.info(f"运行策略: {strategy_name}")
            
            df_strategy = signal_generator.generate_signals(df.copy(), strategy_name=strategy_name)
            
            backtest_engine.reset()
            results = backtest_engine.run(df_strategy, ts_code=ts_code)
            
            results_dict[strategy_name] = results
            
            print(f"\n{'='*60}")
            print(f"策略: {strategy_name}")
            print(performance_analyzer.generate_report(results))
            
        except Exception as e:
            log.error(f"策略 {strategy_name} 运行失败: {e}")
            continue
    
    return results_dict

def optimize_strategy_params(ts_code: str = '000001.SZ',
                            start_date: str = '20200101',
                            end_date: str = '20250101'):
    log.info(f"开始策略参数优化: {ts_code}")
    
    df = data_fetcher.fetch_stock_daily(ts_code, start_date, end_date)
    
    if df.empty:
        log.error("获取数据失败")
        return None
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    def objective_func(data, short_period=5, long_period=20):
        df_local = data.copy()
        df_local = factor_library.calculate_ma(df_local, periods=[short_period, long_period])
        df_local = signal_generator.ma_cross_signal(df_local, short_period=short_period, long_period=long_period)
        backtest_engine.reset()
        result = backtest_engine.run(df_local, ts_code=ts_code)
        return result
    
    param_grid = {
        'short_period': [3, 5, 8, 10],
        'long_period': [15, 20, 30, 60]
    }
    
    from backtest.analytics import parameter_optimizer
    
    opt_result = parameter_optimizer.grid_search(
        objective_func,
        param_grid,
        df,
        objective='sharpe_ratio'
    )
    
    print(f"\n最优参数: {opt_result['best_params']}")
    print(f"最优夏普比率: {opt_result['best_score']:.4f}")
    
    return opt_result

if __name__ == '__main__':
    import pandas as pd
    
    print("股票推荐系统 - 策略回测示例")
    print("=" * 60)
    
    print("\n1. 均线交叉策略回测")
    results, df = run_ma_cross_strategy(
        ts_code='000001.SZ',
        start_date='20200101',
        end_date='20250101',
        short_period=5,
        long_period=20
    )
    
    print("\n2. 多策略对比")
    results_dict = run_multi_strategy_comparison(
        ts_code='000001.SZ',
        start_date='20200101',
        end_date='20250101'
    )
    
    print("\n3. 参数优化")
    opt_result = optimize_strategy_params(
        ts_code='000001.SZ',
        start_date='20200101',
        end_date='20250101'
    )
