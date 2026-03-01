import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from config import Config, user_config
from logger import log
from data.data_fetcher import data_fetcher, HAS_AKSHARE, HAS_TUSHARE
from data.stock_universe import stock_universe
from data.sample_data import sample_data_generator
from data.recommendation_cache import recommendation_cache
from strategy.recommendation_engine import recommendation_engine
from strategy.factors import factor_library
from strategy.signals import signal_generator
from strategy.single_horizon_optimizer import run_single_horizon_optimization
from backtest.engine import backtest_engine
from backtest.recommendation_backtest import recommendation_backtester

try:
    from data.database import db_manager, StockBasic, StockDaily, HAS_SQLALCHEMY
except ImportError:
    HAS_SQLALCHEMY = False
    db_manager = None
    StockBasic = None
    StockDaily = None


def print_welcome():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    股票推荐系统 (FinancialTools)               ║
║                      A股量化交易回测平台                        ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def get_pool_display_name(pool_type: str, risk_level: str) -> str:
    if pool_type == 'core':
        return '核心标的池'
    elif pool_type == 'risk':
        risk_names = {
            'low': '保守型 (上证+深证)',
            'medium': '稳健型 (上证+深证+创业板)',
            'high': '进取型 (上证+深证+创业板+科创板)',
            'all': '激进型 (全部板块)'
        }
        return risk_names.get(risk_level, '未知风险等级')
    else:
        return '未知股票池'


def print_menu(pool_type='core', risk_level='low', current_stock_count=80):
    pool_display = get_pool_display_name(pool_type, risk_level)
    print("\n" + "="*60)
    print("请选择操作:")
    print(f"  1. 获取推荐股票")
    print(f"  2. 推荐历史回测")
    print(f"  3. 历史数据回测验证")
    print(f"  4. 策略优化 - 使用历史数据优化策略参数")
    print(f"  5. 选择风险等级/股票池 (当前: {pool_display})")
    print(f"  6. 设置分析股票数量 (当前: {current_stock_count}支)")
    print(f"  0. 退出")
    print("="*60)

def print_recommendation_submenu():
    print("\n" + "="*60)
    print("请选择推荐周期:")
    print("  1. 短线推荐 (5支，预计1-5个交易日，明天上涨概率最大)")
    print("  2. 中长期推荐 (5支，预计1-3个月)")
    print("  3. 长期推荐 (5支，预计6个月以上)")
    print("  0. 返回")
    print("="*60)

def print_backtest_submenu():
    print("\n" + "="*60)
    print("请选择回测周期:")
    print("  1. 短线推荐历史回测 (10天前推荐，查看后5个交易日表现)")
    print("  2. 中线推荐历史回测 (45天前推荐，查看后30个交易日表现)")
    print("  3. 长线推荐历史回测 (210天前推荐，查看后180个交易日表现)")
    print("  0. 返回")
    print("="*60)


def print_risk_level_menu():
    print("\n" + "="*60)
    print("请选择风险等级:")
    print("  1. 保守型 (初入股市) - 仅上证+深证主板")
    print("  2. 稳健型 (有经验) - 上证+深证+创业板")
    print("  3. 进取型 (经验丰富) - 上证+深证+创业板+科创板")
    print("  4. 激进型 (高风险承受) - 所有板块 (含北交所)")
    print("  5. 核心标的池 (80支核心股票)")
    print("  0. 返回")
    print("="*60)

def print_strategy_optimization_menu():
    print("\n" + "="*60)
    print("【策略优化】")
    print("="*60)
    print("请选择优化周期:")
    print("  1. 短线策略优化")
    print("  2. 中期策略优化")
    print("  3. 长期策略优化")
    print("  4. 优化所有周期策略")
    print("  0. 返回")
    print("="*60)

def print_optimization_type_menu(horizon_name: str):
    print("\n" + "="*60)
    print(f"【{horizon_name}策略优化】")
    print("="*60)
    print("请选择优化类型:")
    print("  1. 经典指标策略优化")
    print("  2. 机器学习增量训练")
    print("  0. 返回")
    print("="*60)


def fetch_realtime_stock_data(pool_type='core', risk_level='low', stock_count=80):
    pool_display = get_pool_display_name(pool_type, risk_level)
    log.info(f"正在从{pool_display}获取 {stock_count} 支A股股票数据...")
    
    actual_pool_type = risk_level if pool_type == 'risk' else pool_type
    
    stock_data = stock_universe.fetch_all_stock_data(
        pool_type=actual_pool_type,
        stock_count=stock_count, 
        days=user_config.data_days, 
        show_progress=True
    )
    
    if not stock_data:
        log.warning("获取真实数据失败，使用样本数据")
        stock_data = sample_data_generator.generate_sample_stock_data(days=user_config.data_days)
    else:
        print("\n正在预计算并缓存推荐结果...")
        
        today_str = datetime.now().strftime('%Y%m%d')
        
        horizons = [
            ('short', '短线', user_config.short_term_top_n),
            ('medium', '中线', user_config.medium_term_top_n),
            ('long', '长线', user_config.long_term_top_n)
        ]
        
        for horizon, name, top_n in horizons:
            if not recommendation_cache.has_valid_cache(horizon, today_str):
                print(f"  生成{name}推荐...")
                if horizon == 'short':
                    recs = recommendation_engine.generate_short_term_recommendations(stock_data, top_n=top_n)
                elif horizon == 'medium':
                    recs = recommendation_engine.generate_medium_long_term_recommendations(stock_data, top_n=top_n)
                else:
                    recs = recommendation_engine.generate_long_term_recommendations(stock_data, top_n=top_n)
                
                recommendation_cache.save_recommendations(recs, horizon, today_str)
            else:
                print(f"  {name}推荐已有缓存，跳过")
    
    return stock_data


def print_recommendations(recommendations, period_type):
    period_names = {
        'short': '短线',
        'medium': '中长期',
        'long': '长期'
    }
    
    print(f"\n{'='*80}")
    print(f"【{period_names[period_type]}推荐】 - {len(recommendations)} 支股票")
    print(f"{'='*80}\n")
    
    for idx, rec in enumerate(recommendations, 1):
        print(f"【第 {idx} 名】")
        print(f"  股票代码: {rec['ts_code']}")
        print(f"  股票名称: {rec['name']}")
        print(f"  当前价格: {rec['current_price']:.2f} 元")
        print(f"  推荐评分: {rec['score']:.1f} 分")
        print(f"\n  推荐原因: {rec['reason']}")
        print(f"\n  买入分析:")
        print(f"    - 目标价位: {rec['analysis']['target_price']:.2f} 元")
        print(f"    - 止损价位: {rec['analysis']['stop_loss']:.2f} 元")
        print(f"    - 持仓周期: {rec['analysis']['holding_period']}")
        print(f"    - 进场建议: {rec['analysis']['entry_suggestion']}")
        print(f"    - 风险控制: {rec['analysis']['risk_control']}")
        print(f"\n{'-'*80}\n")


def get_stocks_with_data():
    """获取数据库中有数据的股票列表"""
    try:
        if not HAS_SQLALCHEMY or not db_manager:
            return []
        
        session = db_manager.get_session()
        try:
            stocks = session.query(StockDaily.ts_code).distinct().all()
            return [ts_code for (ts_code,) in stocks]
        finally:
            session.close()
    except Exception as e:
        log.warning(f"获取有数据的股票列表失败: {e}")
        return []


def run_backtest_validation():
    print("\n" + "="*80)
    print("【历史数据回测验证】")
    print("="*80)
    
    stocks_with_data = get_stocks_with_data()
    
    if not stocks_with_data:
        print("\n数据库中没有股票数据！")
        print("\n请先选择股票推荐选项（1/2/3）来获取数据，")
        print("或者运行 fetch_and_save_data.py 脚本来获取数据。")
        print("\n使用演示数据进行回测...\n")
        df = sample_data_generator.generate_backtest_demo(days=252)
        target_stocks = [('000000.SZ', '演示股票')]
    else:
        print(f"\n数据库中有 {len(stocks_with_data)} 支股票有数据")
        print("选择前3支股票进行回测...\n")
        
        target_stocks = []
        for ts_code in stocks_with_data[:3]:
            try:
                session = db_manager.get_session()
                try:
                    basic = session.query(StockBasic).filter(StockBasic.ts_code == ts_code).first()
                    name = basic.name if basic else ts_code
                    target_stocks.append((ts_code, name))
                finally:
                    session.close()
            except:
                target_stocks.append((ts_code, ts_code))
    
    strategies = [
        ('ma_cross', '均线交叉策略', {}),
        ('macd', 'MACD策略', {}),
        ('rsi', 'RSI策略', {}),
        ('bollinger', '布林带策略', {}),
        ('kdj', 'KDJ策略', {}),
    ]
    
    all_results = []
    
    for ts_code, stock_name in target_stocks:
        print(f"\n{'='*80}")
        print(f"【股票: {stock_name} ({ts_code})】")
        print(f"{'='*80}")
        
        if stocks_with_data:
            df = data_fetcher.load_stock_daily_from_db(ts_code)
            if df.empty:
                print(f"\n  警告: {ts_code} 数据为空，跳过")
                continue
        else:
            df = sample_data_generator.generate_backtest_demo(days=252)
        
        print(f"  数据时间范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
        print(f"  数据条数: {len(df)}")
        
        stock_results = []
        
        for strategy_name, strategy_desc, kwargs in strategies:
            print(f"\n  正在测试: {strategy_desc}")
            
            try:
                df_with_signals = signal_generator.generate_signals(df.copy(), strategy_name, **kwargs)
                results = backtest_engine.run(df_with_signals)
                
                if results:
                    summary = {
                        '股票': stock_name,
                        '代码': ts_code,
                        '策略': strategy_desc,
                        '总收益': f"{results.get('total_return', 0)*100:.2f}%",
                        '年化收益': f"{results.get('annual_return', 0)*100:.2f}%",
                        '夏普比率': f"{results.get('sharpe_ratio', 0):.2f}",
                        '最大回撤': f"{results.get('max_drawdown', 0)*100:.2f}%",
                        '交易次数': results.get('total_trades', 0),
                        '胜率': f"{results.get('win_rate', 0)*100:.2f}%",
                    }
                    stock_results.append(summary)
                    all_results.append(summary)
                    
                    print(f"    ✓ 总收益率: {summary['总收益']}")
                    print(f"    ✓ 年化收益: {summary['年化收益']}")
                    print(f"    ✓ 夏普比率: {summary['夏普比率']}")
                    print(f"    ✓ 最大回撤: {summary['最大回撤']}")
                    print(f"    ✓ 交易次数: {summary['交易次数']}")
                    print(f"    ✓ 胜率: {summary['胜率']}")
                else:
                    print(f"    ✗ 回测失败")
                    
            except Exception as e:
                print(f"    ✗ 错误: {e}")
                log.error(f"回测 {stock_name} {strategy_name} 失败: {e}")
        
        if stock_results:
            print(f"\n  {stock_name} 回测结果汇总:")
            result_df = pd.DataFrame(stock_results)
            print("\n" + result_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("【所有股票回测结果汇总】")
    print(f"{'='*80}")
    
    if all_results:
        result_df = pd.DataFrame(all_results)
        print("\n" + result_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("【系统可行性分析】")
        print(f"{'='*80}")
        print("\n✓ 系统已成功实现完整的量化交易回测框架")
        print("✓ 包含多种经典技术分析策略（均线、MACD、RSI、布林带、KDJ等）")
        print("✓ 支持滑点、手续费等真实交易成本模拟")
        print("✓ 提供完整的绩效指标（收益率、夏普比率、最大回撤、胜率等）")
        print("✓ 推荐系统基于多因子综合评分，包含趋势、动量、成交量等维度")
        
        if stocks_with_data:
            print("\n✓ 使用真实历史数据进行回测验证")
        else:
            print("\n注意: 以上回测使用演示数据，请先获取真实数据进行验证")
    else:
        print("\n未获得有效回测结果")


def main():
    print_welcome()
    log.info("股票推荐系统启动")
    
    print(f"\n数据源: {'AKShare' if HAS_AKSHARE else 'Tushare' if HAS_TUSHARE else '样本数据'}")
    
    stock_data = None
    
    while True:
        print_menu(user_config.pool_type, user_config.risk_level, user_config.stock_count)
        
        try:
            choice = input(f"\n请输入选项 (0-9): ").strip()
            
            if choice == '0':
                print("\n感谢使用，再见！")
                break
            
            elif choice == '1':
                print_recommendation_submenu()
                sub_choice = input("\n请选择推荐周期 (0-3): ").strip()
                
                if sub_choice == '0':
                    continue
                elif sub_choice in ['1', '2', '3']:
                    horizon_map = {
                        '1': ('short', '短线', user_config.short_term_top_n),
                        '2': ('medium', '中线', user_config.medium_term_top_n),
                        '3': ('long', '长线', user_config.long_term_top_n)
                    }
                    
                    horizon, name, top_n = horizon_map[sub_choice]
                    today_str = datetime.now().strftime('%Y%m%d')
                    
                    use_cache = False
                    recommendations = []
                    
                    if recommendation_cache.has_valid_cache(horizon, today_str):
                        print(f"\n从缓存加载{name}推荐...")
                        recommendations = recommendation_cache.load_recommendations(horizon, today_str, top_n=top_n)
                        if recommendations:
                            use_cache = True
                    
                    if not use_cache:
                        if stock_data is None:
                            pool_display = get_pool_display_name(user_config.pool_type, user_config.risk_level)
                            print(f"\n正在准备数据 ({pool_display} - {user_config.stock_count} 支)...")
                            stock_data = fetch_realtime_stock_data(
                                pool_type=user_config.pool_type,
                                risk_level=user_config.risk_level,
                                stock_count=user_config.stock_count
                            )
                        else:
                            print("\n正在预计算并缓存所有推荐结果...")
                            horizons = [
                                ('short', '短线', user_config.short_term_top_n),
                                ('medium', '中线', user_config.medium_term_top_n),
                                ('long', '长线', user_config.long_term_top_n)
                            ]
                            
                            for h, n, tn in horizons:
                                if not recommendation_cache.has_valid_cache(h, today_str):
                                    print(f"  生成{n}推荐...")
                                    if h == 'short':
                                        recs = recommendation_engine.generate_short_term_recommendations(stock_data, top_n=tn)
                                    elif h == 'medium':
                                        recs = recommendation_engine.generate_medium_long_term_recommendations(stock_data, top_n=tn)
                                    else:
                                        recs = recommendation_engine.generate_long_term_recommendations(stock_data, top_n=tn)
                                    
                                    recommendation_cache.save_recommendations(recs, h, today_str)
                                else:
                                    print(f"  {n}推荐已有缓存，跳过")
                    
                    recommendations = recommendation_cache.load_recommendations(horizon, today_str, top_n=top_n)
                    print_recommendations(recommendations, horizon)
                    input("\n按回车键继续...")
                else:
                    print("\n无效选项")
                    input("\n按回车键继续...")
            
            elif choice == '2':
                print_backtest_submenu()
                sub_choice = input("\n请选择回测周期 (0-3): ").strip()
                
                if sub_choice == '0':
                    continue
                elif sub_choice in ['1', '2', '3']:
                    horizon_map = {
                        '1': ('short', '短线'),
                        '2': ('medium', '中线'),
                        '3': ('long', '长线')
                    }
                    
                    horizon, name = horizon_map[sub_choice]
                    
                    print("\n" + "="*80)
                    print(f"【{name}推荐历史回测】")
                    print("="*80)
                    
                    if horizon == 'short':
                        print("\n说明: 寻找10天前的5个短线股票推荐，")
                        print("      查看后5个交易日的表现\n")
                        top_n = user_config.short_term_top_n
                    elif horizon == 'medium':
                        print("\n说明: 寻找45天前的5个中线股票推荐，")
                        print("      查看后30个交易日的表现\n")
                        top_n = user_config.medium_term_top_n
                    else:
                        print("\n说明: 寻找210天前的5个长期股票推荐，")
                        print("      查看后180个交易日的表现\n")
                        top_n = user_config.long_term_top_n
                    
                    recommendation_backtester.run_backtest(
                        horizon=horizon,
                        top_n=top_n,
                        stock_count=user_config.stock_count,
                        pool_type=user_config.pool_type,
                        risk_level=user_config.risk_level
                    )
                    input("\n按回车键继续...")
                else:
                    print("\n无效选项")
                    input("\n按回车键继续...")
            
            elif choice == '3':
                run_backtest_validation()
                input("\n按回车键继续...")
            
            elif choice == '4':
                while True:
                    print_strategy_optimization_menu()
                    opt_choice = input("\n请选择 (0-4): ").strip()
                    
                    if opt_choice == '0':
                        break
                    elif opt_choice == '4':
                        print("\n" + "="*80)
                        print("【优化所有周期策略】")
                        print("="*80)
                        print("\n说明: 使用历史数据迭代优化短、中、长期策略参数")
                        print("      优化目标: 夏普比率 (风险调整后收益)")
                        
                        confirm = input("\n是否开始优化所有周期策略？(y/n): ").strip().lower()
                        if confirm == 'y':
                            try:
                                import optimize_strategy
                                optimize_strategy.main()
                            except Exception as e:
                                print(f"\n策略优化失败: {e}")
                                log.error(f"策略优化失败: {e}")
                                import traceback
                                traceback.print_exc()
                        input("\n按回车键继续...")
                    elif opt_choice in ['1', '2', '3']:
                        horizon_map = {
                            '1': ('short', '短线'),
                            '2': ('medium', '中期'),
                            '3': ('long', '长期')
                        }
                        horizon, horizon_name = horizon_map[opt_choice]
                        
                        while True:
                            print_optimization_type_menu(horizon_name)
                            type_choice = input("\n请选择 (0-2): ").strip()
                            
                            if type_choice == '0':
                                break
                            elif type_choice == '1':
                                print(f"\n" + "="*80)
                                print(f"【{horizon_name}经典指标策略优化】")
                                print("="*80)
                                
                                confirm = input(f"\n是否开始{horizon_name}经典指标策略优化？(y/n): ").strip().lower()
                                if confirm == 'y':
                                    try:
                                        run_single_horizon_optimization(horizon)
                                    except Exception as e:
                                        print(f"\n策略优化失败: {e}")
                                        log.error(f"策略优化失败: {e}")
                                        import traceback
                                        traceback.print_exc()
                                input("\n按回车键继续...")
                            elif type_choice == '2':
                                print(f"\n" + "="*80)
                                print(f"【{horizon_name}机器学习增量训练】")
                                print("="*80)
                                print("\n此功能正在开发中...")
                                input("\n按回车键继续...")
                            else:
                                print("\n无效选项")
                                input("\n按回车键继续...")
                    else:
                        print("\n无效选项")
                        input("\n按回车键继续...")
            
            elif choice == '5':
                print_risk_level_menu()
                risk_choice = input("\n请选择 (0-5): ").strip()
                
                if risk_choice == '0':
                    pass
                elif risk_choice in ['1', '2', '3', '4', '5']:
                    print(f"\n正在清除推荐缓存...")
                    deleted_count = recommendation_cache.clear_all_recommendations()
                    print(f"✓ 已清除 {deleted_count} 条推荐缓存")
                    
                    if risk_choice == '1':
                        user_config.pool_type = 'risk'
                        user_config.risk_level = 'low'
                        user_config.stock_count = 100
                        print(f"正在获取保守型股票池信息...")
                        try:
                            pool_stocks = stock_universe.get_stock_pool(pool_type='low', basic_only=True, use_cache=True)
                            user_config.max_stock_count = len(pool_stocks)
                        except Exception as e:
                            log.warning(f"获取股票池数量失败，使用默认值: {e}")
                            user_config.max_stock_count = 500
                        stock_data = None
                        print(f"✓ 已设置为保守型，默认分析100支股票 (仅上证+深证)，股票池总数: {user_config.max_stock_count}支")
                    elif risk_choice == '2':
                        user_config.pool_type = 'risk'
                        user_config.risk_level = 'medium'
                        user_config.stock_count = 200
                        print(f"正在获取稳健型股票池信息...")
                        try:
                            pool_stocks = stock_universe.get_stock_pool(pool_type='medium', basic_only=True, use_cache=True)
                            user_config.max_stock_count = len(pool_stocks)
                        except Exception as e:
                            log.warning(f"获取股票池数量失败，使用默认值: {e}")
                            user_config.max_stock_count = 1000
                        stock_data = None
                        print(f"✓ 已设置为稳健型，默认分析200支股票 (上证+深证+创业板)，股票池总数: {user_config.max_stock_count}支")
                    elif risk_choice == '3':
                        user_config.pool_type = 'risk'
                        user_config.risk_level = 'high'
                        user_config.stock_count = 300
                        print(f"正在获取进取型股票池信息...")
                        try:
                            pool_stocks = stock_universe.get_stock_pool(pool_type='high', basic_only=True, use_cache=True)
                            user_config.max_stock_count = len(pool_stocks)
                        except Exception as e:
                            log.warning(f"获取股票池数量失败，使用默认值: {e}")
                            user_config.max_stock_count = 2000
                        stock_data = None
                        print(f"✓ 已设置为进取型，默认分析300支股票 (上证+深证+创业板+科创板)，股票池总数: {user_config.max_stock_count}支")
                    elif risk_choice == '4':
                        user_config.pool_type = 'risk'
                        user_config.risk_level = 'all'
                        user_config.stock_count = 500
                        print(f"正在获取激进型股票池信息...")
                        try:
                            pool_stocks = stock_universe.get_stock_pool(pool_type='all', basic_only=True, use_cache=True)
                            user_config.max_stock_count = len(pool_stocks)
                        except Exception as e:
                            log.warning(f"获取股票池数量失败，使用默认值: {e}")
                            user_config.max_stock_count = 5000
                        stock_data = None
                        print(f"✓ 已设置为激进型，默认分析500支股票 (所有板块)，股票池总数: {user_config.max_stock_count}支")
                    elif risk_choice == '5':
                        user_config.pool_type = 'core'
                        user_config.risk_level = 'low'
                        user_config.stock_count = 80
                        print(f"正在获取核心标的池信息...")
                        try:
                            pool_stocks = stock_universe.get_stock_pool(pool_type='core', basic_only=True, use_cache=True)
                            user_config.max_stock_count = len(pool_stocks)
                        except Exception as e:
                            log.warning(f"获取股票池数量失败，使用默认值: {e}")
                            user_config.max_stock_count = 80
                        stock_data = None
                        print(f"✓ 已切换到核心标的池，默认分析80支股票，股票池总数: {user_config.max_stock_count}支")
                else:
                    print("✗ 无效选项")
                
                input("\n按回车键继续...")
            
            elif choice == '6':
                print(f"\n正在获取当前股票池信息...")
                
                # 直接使用用户配置中存储的股票池最大数量
                max_count = user_config.max_stock_count
                
                print(f"\n当前分析股票数量: {user_config.stock_count}")
                print(f"当前股票池总数量: {max_count} 支")
                
                new_count = input(f"请输入新的股票数量 (10-{max_count}): ").strip()
                try:
                    new_count = int(new_count)
                    if 10 <= new_count <= max_count:
                        if new_count != user_config.stock_count:
                            print(f"\n正在清除推荐缓存...")
                            deleted_count = recommendation_cache.clear_all_recommendations()
                            print(f"✓ 已清除 {deleted_count} 条推荐缓存")
                        
                        user_config.stock_count = new_count
                        stock_data = None
                        print(f"✓ 已设置为 {user_config.stock_count} 支，下次获取数据时生效")
                    else:
                        print(f"✗ 请输入10-{max_count}之间的数字")
                except ValueError:
                    print("✗ 请输入有效的数字")
                input("\n按回车键继续...")
            
            else:
                print("\n无效选项，请重新输入")
        
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            log.error(f"主程序错误: {e}")
            input("\n按回车键继续...")


if __name__ == '__main__':
    main()
