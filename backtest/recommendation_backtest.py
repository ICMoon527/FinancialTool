import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from logger import log
from data.data_fetcher import data_fetcher
from data.stock_universe import stock_universe
from strategy.recommendation_engine import recommendation_engine
from strategy.factors import factor_library
from data.recommendation_cache import recommendation_cache
from utils.progress_bar import create_progress_bar


class RecommendationBacktester:
    def __init__(self):
        self._latest_trading_day = None
    
    def get_latest_trading_day(self) -> str:
        """获取数据库中最新的交易日"""
        if self._latest_trading_day:
            return self._latest_trading_day
        
        try:
            today = datetime.now().strftime('%Y%m%d')
            
            for offset in range(60):
                check_date = (datetime.strptime(today, '%Y%m%d') - timedelta(days=offset)).strftime('%Y%m%d')
                
                df = data_fetcher.load_stock_daily_from_db('000001.SZ', check_date, check_date)
                
                if not df.empty:
                    self._latest_trading_day = check_date
                    return check_date
            
            return today
        except Exception as e:
            log.warning(f"获取最新交易日失败: {e}")
            return datetime.now().strftime('%Y%m%d')
    
    def get_nearest_trading_day(self, target_date: str) -> str:
        """获取离目标日期最近的前一个交易日"""
        try:
            target_dt = datetime.strptime(target_date, '%Y%m%d')
            
            for offset in range(30):
                check_date = (target_dt - timedelta(days=offset)).strftime('%Y%m%d')
                
                df = data_fetcher.load_stock_daily_from_db('000001.SZ', check_date, check_date)
                
                if not df.empty:
                    return check_date
            
            return target_date
        except Exception as e:
            log.warning(f"获取最近交易日失败: {e}")
            return target_date
    
    def get_historical_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取某只股票的历史数据"""
        try:
            df = data_fetcher.load_stock_daily_from_db(ts_code, start_date, end_date)
            return df
        except Exception as e:
            log.warning(f"获取 {ts_code} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_price_at_date(self, ts_code: str, target_date: str) -> float:
        """获取某只股票在特定日期的收盘价"""
        try:
            target_dt = datetime.strptime(target_date, '%Y%m%d')
            
            for offset in range(30):
                check_date = (target_dt - timedelta(days=offset)).strftime('%Y%m%d')
                start_date = (target_dt - timedelta(days=60 + offset)).strftime('%Y%m%d')
                df = self.get_historical_data(ts_code, start_date, check_date)
                if not df.empty and 'close' in df.columns and 'trade_date' in df.columns:
                    for _, row in df.iterrows():
                        if row['trade_date'] == check_date:
                            return float(row['close'])
                    if len(df) > 0:
                        return float(df['close'].iloc[-1])
            
            return None
        except Exception as e:
            log.warning(f"获取 {ts_code} 在 {target_date} 的价格失败: {e}")
            return None
    
    def get_price_after_period(self, ts_code: str, start_date: str, days: int) -> Tuple[float, str]:
        """获取某只股票在一段时间后的价格"""
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = start_dt + timedelta(days=days)
            
            for offset in range(60):
                check_date = (end_dt + timedelta(days=offset)).strftime('%Y%m%d')
                search_start = start_date
                search_end = check_date
                df = self.get_historical_data(ts_code, search_start, search_end)
                if not df.empty and 'close' in df.columns and 'trade_date' in df.columns:
                    for _, row in df.iterrows():
                        row_date = row['trade_date']
                        if row_date == check_date and row_date > start_date:
                            return float(row['close']), row_date
                    for _, row in df.iterrows():
                        row_date = row['trade_date']
                        if row_date > start_date:
                            return float(row['close']), row_date
            
            return None, None
        except Exception as e:
            log.warning(f"获取 {ts_code} 后续价格失败: {e}")
            return None, None
    
    def generate_historical_recommendations(self, 
                                           recommendation_date: str,
                                           horizon: str = 'short',
                                           top_n: int = 5,
                                           stock_count: int = 80,
                                           pool_type: str = 'core',
                                           risk_level: str = 'low') -> List[Dict]:
        """
        在历史某个时间点生成推荐
        
        Args:
            recommendation_date: 推荐日期 (YYYYMMDD)
            horizon: 推荐周期 ('short', 'medium', 'long')
            top_n: 推荐股票数量
            stock_count: 分析股票数量
            pool_type: 股票池类型
            risk_level: 风险等级
        """
        try:
            log.info(f"正在生成 {recommendation_date} 的{horizon}推荐...")
            
            rec_dt = datetime.strptime(recommendation_date, '%Y%m%d')
            data_end_date = recommendation_date
            data_start_date = (rec_dt - timedelta(days=365)).strftime('%Y%m%d')
            
            actual_pool_type = risk_level if pool_type == 'risk' else pool_type
            stock_pool = stock_universe.get_stock_pool(pool_type=actual_pool_type, use_cache=True)
            
            if not stock_pool:
                log.warning("股票池为空")
                return []
            
            selected_stocks = stock_pool[:stock_count]
            
            stock_data = {}
            total = len(selected_stocks)
            pb = create_progress_bar(total, '加载历史数据')
            
            for i, (ts_code, name) in enumerate(selected_stocks):
                df = self.get_historical_data(ts_code, data_start_date, data_end_date)
                if not df.empty and len(df) >= 60:
                    df['name'] = name
                    df = factor_library.calculate_all_factors(df)
                    stock_data[ts_code] = df
                pb.update(i + 1)
            
            if not stock_data:
                log.warning("没有足够的历史数据")
                return []
            
            if horizon == 'short':
                recommendations = recommendation_engine.generate_short_term_recommendations(
                    stock_data, top_n=top_n
                )
            elif horizon == 'medium':
                recommendations = recommendation_engine.generate_medium_long_term_recommendations(
                    stock_data, top_n=top_n
                )
            else:
                recommendations = recommendation_engine.generate_long_term_recommendations(
                    stock_data, top_n=top_n
                )
            
            log.info(f"成功生成 {len(recommendations)} 支{horizon}推荐")
            return recommendations
            
        except Exception as e:
            log.error(f"生成历史推荐失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def evaluate_recommendation_performance(self,
                                            recommendations: List[Dict],
                                            recommendation_date: str,
                                            holding_days: int,
                                            horizon: str = 'short') -> Dict[str, Any]:
        """
        评估推荐的表现
        
        Args:
            recommendations: 推荐列表
            recommendation_date: 推荐日期 (YYYYMMDD)
            holding_days: 持有天数
            horizon: 推荐周期
        """
        results = {
            'recommendation_date': recommendation_date,
            'holding_days': holding_days,
            'stock_count': len(recommendations),
            'stocks': [],
            'win_count': 0,
            'total_return': 0.0,
            'avg_return': 0.0,
            'max_return': -float('inf'),
            'min_return': float('inf')
        }
        
        total = len(recommendations)
        pb = create_progress_bar(total, '评估推荐表现')
        
        for i, rec in enumerate(recommendations):
            ts_code = rec['ts_code']
            name = rec['name']
            
            buy_price = self.get_price_at_date(ts_code, recommendation_date)
            
            if buy_price is None:
                log.warning(f"无法获取 {ts_code} 在 {recommendation_date} 的买入价格")
                continue
            
            if horizon == 'short':
                sell_price, sell_date = self.get_next_trading_day_price(ts_code, recommendation_date)
            else:
                sell_price, sell_date, _ = self.get_exit_price_with_stop_loss(
                    ts_code, recommendation_date, holding_days, buy_price, 
                    buy_price * 1.20 if horizon == 'medium' else buy_price * 1.50,
                    buy_price * 0.90 if horizon == 'medium' else buy_price * 0.85
                )
            
            if sell_price is None:
                log.warning(f"无法获取 {ts_code} 的卖出价格")
                pb.update(i + 1)
                continue
            
            return_pct = (sell_price - buy_price) / buy_price
            
            stock_result = {
                'ts_code': ts_code,
                'name': name,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'buy_date': recommendation_date,
                'sell_date': sell_date,
                'return': return_pct
            }
            
            results['stocks'].append(stock_result)
            results['total_return'] += return_pct
            
            if return_pct > 0:
                results['win_count'] += 1
            
            if return_pct > results['max_return']:
                results['max_return'] = return_pct
            if return_pct < results['min_return']:
                results['min_return'] = return_pct
            
            pb.update(i + 1)
        
        valid_count = len(results['stocks'])
        if valid_count > 0:
            results['avg_return'] = results['total_return'] / valid_count
            results['win_rate'] = results['win_count'] / valid_count
        else:
            results['avg_return'] = 0
            results['win_rate'] = 0
        
        return results
    
    def get_next_trading_day_price(self, ts_code: str, start_date: str) -> Tuple[float, str]:
        """
        获取下一个交易日的收盘价（严格T+1）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
        
        Returns:
            (收盘价, 日期)
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = start_dt + timedelta(days=30)
            end_date = end_dt.strftime('%Y%m%d')
            
            df = self.get_historical_data(ts_code, start_date, end_date)
            
            if df.empty:
                return None, None
            
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            found_start = False
            for idx, row in df.iterrows():
                trade_date = row['trade_date']
                
                if not found_start and trade_date == start_date:
                    found_start = True
                    continue
                
                if found_start:
                    return float(row['close']), trade_date
            
            return None, None
        except Exception as e:
            log.warning(f"获取 {ts_code} 下一个交易日价格失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_exit_price_with_stop_loss(self, ts_code: str, start_date: str, max_days: int, 
                                      buy_price: float, target_price: float, stop_loss: float) -> Tuple[float, str, str]:
        """
        获取带有止盈止损的出场价格
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            max_days: 最大持有天数
            buy_price: 买入价格
            target_price: 止盈价格
            stop_loss: 止损价格
        
        Returns:
            (卖出价格, 卖出日期, 出场类型: 'take_profit', 'stop_loss', 'hold_to_end')
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = start_dt + timedelta(days=max_days + 60)
            end_date = end_dt.strftime('%Y%m%d')
            
            df = self.get_historical_data(ts_code, start_date, end_date)
            
            if df.empty:
                return None, None, None
            
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            for idx, row in df.iterrows():
                trade_date = row['trade_date']
                
                if trade_date <= start_date:
                    continue
                
                high = float(row['high'])
                low = float(row['low'])
                close = float(row['close'])
                
                if high >= target_price:
                    return target_price, trade_date, 'take_profit'
                
                if low <= stop_loss:
                    return stop_loss, trade_date, 'stop_loss'
                
                days_since_start = (datetime.strptime(trade_date, '%Y%m%d') - start_dt).days
                if days_since_start >= max_days:
                    return close, trade_date, 'hold_to_end'
            
            if len(df) > 0:
                last_row = df.iloc[-1]
                return float(last_row['close']), last_row['trade_date'], 'hold_to_end'
            
            return None, None, None
        except Exception as e:
            log.warning(f"获取 {ts_code} 止盈止损出场价格失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def run_backtest(self,
                     horizon: str = 'short',
                     top_n: int = 5,
                     stock_count: int = 80,
                     pool_type: str = 'core',
                     risk_level: str = 'low') -> Dict[str, Any]:
        """
        运行推荐回测
        
        Args:
            horizon: 推荐周期 ('short', 'medium', 'long')
            top_n: 推荐股票数量
            stock_count: 分析股票数量
            pool_type: 股票池类型
            risk_level: 风险等级
        """
        horizon_configs = {
            'short': {
                'name': '短线',
                'days_ago': 10,
                'holding_days': 1
            },
            'medium': {
                'name': '中线',
                'days_ago': 45,
                'holding_days': 30
            },
            'long': {
                'name': '长线',
                'days_ago': 210,
                'holding_days': 180
            }
        }
        
        config = horizon_configs.get(horizon, horizon_configs['short'])
        
        today = datetime.now()
        raw_recommendation_date = (today - timedelta(days=config['days_ago'])).strftime('%Y%m%d')
        recommendation_date = self.get_nearest_trading_day(raw_recommendation_date)
        
        print(f"\n{'='*80}")
        print(f"【{config['name']}推荐历史回测】")
        print(f"{'='*80}")
        if raw_recommendation_date != recommendation_date:
            print(f"\n原始日期: {raw_recommendation_date} (非交易日)")
        print(f"\n推荐日期: {recommendation_date}")
        if horizon == 'short':
            print(f"持有周期: 第二天卖出")
        else:
            print(f"持有周期: {config['holding_days']} 个交易日")
        print(f"推荐数量: {top_n} 支")
        
        print(f"\n检查是否有缓存的推荐...")
        recommendations = recommendation_cache.load_recommendations(
            horizon=horizon,
            recommendation_date=recommendation_date,
            top_n=top_n
        )
        
        if recommendations:
            print(f"✓ 从数据库缓存加载了 {len(recommendations)} 条推荐")
        else:
            print(f"未找到缓存，正在生成历史推荐...")
            recommendations = self.generate_historical_recommendations(
                recommendation_date=recommendation_date,
                horizon=horizon,
                top_n=top_n,
                stock_count=stock_count,
                pool_type=pool_type,
                risk_level=risk_level
            )
            
            if recommendations:
                print(f"保存推荐到数据库缓存...")
                recommendation_cache.save_recommendations(
                    recommendations=recommendations,
                    horizon=horizon,
                    recommendation_date=recommendation_date
                )
        
        if not recommendations:
            print("\n无法生成历史推荐，可能是历史数据不足")
            return {}
        
        print(f"\n生成的推荐:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['name']} ({rec['ts_code']}) - 评分: {rec['score']:.1f}")
        
        print(f"\n正在评估推荐表现...")
        performance = self.evaluate_recommendation_performance(
            recommendations=recommendations,
            recommendation_date=recommendation_date,
            holding_days=config['holding_days'],
            horizon=horizon
        )
        
        print(f"\n{'='*80}")
        print(f"【{config['name']}推荐回测结果】")
        print(f"{'='*80}")
        
        print(f"\n总体表现:")
        print(f"  有效推荐数: {len(performance['stocks'])} / {len(recommendations)}")
        print(f"  平均收益率: {performance['avg_return']*100:.2f}%")
        print(f"  总收益率: {performance['total_return']*100:.2f}%")
        print(f"  胜率: {performance['win_rate']*100:.2f}%")
        
        if performance['stocks']:
            print(f"\n单只股票表现:")
            for i, stock in enumerate(performance['stocks'], 1):
                status = "✓ 盈利" if stock['return'] > 0 else "✗ 亏损"
                print(f"  {i}. {stock['name']} ({stock['ts_code']})")
                print(f"     买入: {stock['buy_date']} @ {stock['buy_price']:.2f}")
                print(f"     卖出: {stock['sell_date']} @ {stock['sell_price']:.2f}")
                print(f"     收益: {stock['return']*100:+.2f}% {status}")
        
        print(f"\n{'='*80}")
        
        return performance


recommendation_backtester = RecommendationBacktester()
