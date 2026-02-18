import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from itertools import product
from logger import log
from strategy.factors import factor_library
from data.stock_universe import stock_universe
from data.data_fetcher import data_fetcher
from utils.progress_bar import create_progress_bar


class StrategyOptimizer:
    
    def __init__(self):
        pass
    
    def evaluate_strategy_on_stock(self, 
                                   df: pd.DataFrame, 
                                   strategy_config: Dict,
                                   horizon: str = 'short') -> Dict[str, Any]:
        """
        在单只股票上评估策略表现
        
        Args:
            df: 股票历史数据
            strategy_config: 策略配置
            horizon: 周期 ('short', 'medium', 'long')
        
        Returns:
            策略表现指标
        """
        if df.empty or len(df) < 60:
            return None
        
        # 因子已经在预处理阶段计算过了，不需要重复计算
        df = df.copy()
        
        signals = self._generate_signals_from_config(df, strategy_config, horizon)
        
        if len(signals) == 0:
            return None
        
        performance = self._calculate_performance_metrics(df, signals, horizon)
        return performance
    
    def _generate_signals_from_config(self, 
                                     df: pd.DataFrame, 
                                     strategy_config: Dict,
                                     horizon: str) -> List[Dict]:
        """根据策略配置生成信号"""
        signals = []
        
        if horizon == 'short':
            entry_days = 5
            exit_days = 10
        elif horizon == 'medium':
            entry_days = 20
            exit_days = 60
        else:
            entry_days = 60
            exit_days = 180
        
        for i in range(len(df) - exit_days):
            if i < max(entry_days, 60):
                continue
            
            current_df = df.iloc[:i+1]
            score = self._calculate_score_from_config(current_df, strategy_config, horizon)
            
            if score >= strategy_config.get('entry_threshold', 70):
                entry_price = df['close'].iloc[i]
                entry_date = df['trade_date'].iloc[i] if 'trade_date' in df.columns else i
                
                exit_idx = min(i + exit_days, len(df) - 1)
                exit_price = df['close'].iloc[exit_idx]
                exit_date = df['trade_date'].iloc[exit_idx] if 'trade_date' in df.columns else exit_idx
                
                max_price = df['high'].iloc[i:exit_idx+1].max()
                min_price = df['low'].iloc[i:exit_idx+1].min()
                
                signals.append({
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'entry_date': entry_date,
                    'exit_idx': exit_idx,
                    'exit_price': exit_price,
                    'exit_date': exit_date,
                    'max_price': max_price,
                    'min_price': min_price
                })
        
        return signals
    
    def _calculate_score_from_config(self, 
                                    df: pd.DataFrame, 
                                    strategy_config: Dict,
                                    horizon: str) -> float:
        """根据策略配置计算评分"""
        score = 0.0
        current_price = df['close'].iloc[-1]
        
        if horizon == 'short':
            ma5 = df['MA5'].iloc[-1] if 'MA5' in df.columns else current_price
            ma10 = df['MA10'].iloc[-1] if 'MA10' in df.columns else current_price
            ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
            
            if current_price > ma5:
                score += strategy_config.get('ma5_weight', 15)
            if ma5 > ma10:
                score += strategy_config.get('ma5_ma10_weight', 10)
            if ma10 > ma20:
                score += strategy_config.get('ma10_ma20_weight', 10)
            
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                if macd > macd_signal and macd > 0:
                    score += strategy_config.get('macd_positive_weight', 20)
                elif macd > macd_signal:
                    score += strategy_config.get('macd_cross_weight', 10)
            
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if 30 <= rsi <= 50:
                    score += strategy_config.get('rsi_low_weight', 15)
                elif 50 < rsi <= 70:
                    score += strategy_config.get('rsi_mid_weight', 10)
            
            if 'K' in df.columns and 'D' in df.columns:
                k = df['K'].iloc[-1]
                d = df['D'].iloc[-1]
                if k > d and k < 50:
                    score += strategy_config.get('kdj_weight', 15)
        
        elif horizon == 'medium':
            ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
            ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
            
            if current_price > ma20:
                score += strategy_config.get('ma20_weight', 10)
            if ma20 > ma60:
                score += strategy_config.get('ma20_ma60_weight', 15)
            
            if 'MACD' in df.columns:
                macd = df['MACD'].iloc[-1]
                if macd > 0:
                    score += strategy_config.get('macd_weight', 15)
            
            prices_60 = df['close'].iloc[-60:]
            if prices_60.iloc[-1] > prices_60.iloc[0]:
                score += strategy_config.get('trend_weight', 20)
            
            if 'OBV' in df.columns:
                obv_trend = df['OBV'].iloc[-20:].mean() > df['OBV'].iloc[-40:-20].mean()
                if obv_trend:
                    score += strategy_config.get('obv_weight', 15)
            
            volatility = df['close'].pct_change().iloc[-60:].std()
            if volatility < 0.03:
                score += strategy_config.get('volatility_weight', 15)
        
        else:
            ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
            ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
            ma120 = df['MA120'].iloc[-1] if ('MA120' in df.columns and len(df) >= 120) else ma60
            
            if current_price > ma20:
                score += strategy_config.get('ma20_weight', 10)
            if ma20 > ma60:
                score += strategy_config.get('ma20_ma60_weight', 15)
            if len(df) >= 120 and ma60 > ma120:
                score += strategy_config.get('ma60_ma120_weight', 20)
            
            lookback_days = min(len(df), 120)
            prices_period = df['close'].iloc[-lookback_days:]
            if prices_period.iloc[-1] > prices_period.iloc[0]:
                score += strategy_config.get('trend_weight', 25)
            
            max_period = prices_period.max()
            if current_price > max_period * 0.8:
                score += strategy_config.get('near_high_weight', 15)
            
            if 'OBV' in df.columns:
                obv_short = min(len(df), 30)
                obv_long = min(len(df), 60)
                if obv_long > obv_short:
                    obv_trend = df['OBV'].iloc[-obv_short:].mean() > df['OBV'].iloc[-obv_long:-obv_short].mean()
                    if obv_trend:
                        score += strategy_config.get('obv_weight', 15)
        
        return min(score, 100)
    
    def _calculate_performance_metrics(self, 
                                       df: pd.DataFrame, 
                                       signals: List[Dict],
                                       horizon: str) -> Dict[str, Any]:
        """计算策略表现指标"""
        if not signals:
            return None
        
        returns = []
        win_count = 0
        max_drawdowns = []
        max_gains = []
        
        for signal in signals:
            entry_price = signal['entry_price']
            exit_price = signal['exit_price']
            max_price = signal['max_price']
            min_price = signal['min_price']
            
            ret = (exit_price - entry_price) / entry_price
            returns.append(ret)
            
            if ret > 0:
                win_count += 1
            
            max_drawdown = (min_price - entry_price) / entry_price
            max_gain = (max_price - entry_price) / entry_price
            max_drawdowns.append(max_drawdown)
            max_gains.append(max_gain)
        
        total_trades = len(signals)
        avg_return = np.mean(returns) if returns else 0
        total_return = np.sum(returns) if returns else 0
        win_rate = win_count / total_trades if total_trades > 0 else 0
        max_drawdown = np.min(max_drawdowns) if max_drawdowns else 0
        max_gain = np.max(max_gains) if max_gains else 0
        
        profit_factor = self._calculate_profit_factor(returns)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'max_gain': max_gain,
            'profit_factor': profit_factor,
            'returns': returns
        }
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """计算盈亏比"""
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def hierarchical_optimization(self,
                                stock_data: Dict[str, pd.DataFrame],
                                param_ranges: Dict,
                                horizon: str = 'short',
                                objective: str = 'total_return') -> Tuple[Dict, Dict]:
        """
        分层优化策略参数（按重要性顺序）
        
        Args:
            stock_data: 多只股票数据
            param_ranges: 参数范围
            horizon: 周期
            objective: 优化目标 ('total_return', 'win_rate', 'profit_factor', 'sharpe')
        
        Returns:
            (最优参数, 最优表现)
        """
        # 获取按重要性排序的指标分组
        importance_groups = self._get_importance_groups(horizon)
        
        # 预计算所有股票的因子，避免重复计算
        print(f"\n预计算所有股票的技术指标因子...")
        prepared_stock_data = {}
        total = len(stock_data)
        pb_prepare = create_progress_bar(total, '预处理股票数据')
        
        for i, (ts_code, df) in enumerate(stock_data.items()):
            if not df.empty and len(df) >= 60:
                df_copy = df.copy()
                df_copy = factor_library.calculate_all_factors(df_copy)
                prepared_stock_data[ts_code] = df_copy
            pb_prepare.update(i + 1)
        
        # 初始化当前最佳参数为默认参数
        current_params = self._get_default_params(horizon)
        best_performance = None
        
        print(f"\n开始分层优化，共 {len(importance_groups)} 个层级...")
        
        for level_idx, (level_name, important_params) in enumerate(importance_groups.items()):
            print(f"\n{'='*60}")
            print(f"层级 {level_idx + 1}: {level_name}")
            print(f"{'='*60}")
            print(f"  优化指标: {important_params}")
            
            # 为当前层级创建参数范围
            level_param_ranges = {}
            for param in important_params:
                if param in param_ranges:
                    level_param_ranges[param] = param_ranges[param]
            
            # 生成当前层级的参数组合
            param_names = list(level_param_ranges.keys())
            param_values = [level_param_ranges[name] for name in param_names]
            total_combinations = np.prod([len(v) for v in param_values])
            
            print(f"  参数组合数量: {total_combinations}")
            
            # 网格搜索当前层级
            # 使用当前全局最佳得分作为基准，只有找到更好的参数才标记为更优
            if best_performance:
                current_best_score = self._calculate_objective_score(best_performance, objective)
            else:
                current_best_score = -float('inf')
            
            level_best_score = current_best_score
            level_best_params = None
            level_best_perf = None
            
            pb = create_progress_bar(total_combinations, f'优化 {level_name}')
            
            for idx, combination in enumerate(product(*param_values)):
                # 创建当前参数组合
                test_params = current_params.copy()
                for param_name, param_value in zip(param_names, combination):
                    test_params[param_name] = param_value
                
                # 评估策略
                performance = self.evaluate_strategy_on_multiple_stocks(
                    prepared_stock_data, test_params, horizon
                )
                
                if performance:
                    score = self._calculate_objective_score(performance, objective)
                    
                    if score > level_best_score:
                        level_best_score = score
                        level_best_params = test_params.copy()
                        level_best_perf = performance
                        improvement = score - current_best_score
                        print(f"  发现更优参数组合 (得分: {score:.4f}, 提升: {improvement:.4f})")
                
                pb.update(idx + 1)
            
            # 更新当前最佳参数
            if level_best_params:
                current_params = level_best_params
                best_performance = level_best_perf
                print(f"  层级 {level_idx + 1} 优化完成")
                print(f"  当前最佳参数: {current_params}")
            else:
                print(f"  层级 {level_idx + 1} 未找到有效参数组合")
        
        return current_params, best_performance
    
    def grid_search_optimization(self,
                                stock_data: Dict[str, pd.DataFrame],
                                param_ranges: Dict,
                                horizon: str = 'short',
                                objective: str = 'total_return') -> Tuple[Dict, Dict]:
        """
        网格搜索优化策略参数
        
        Args:
            stock_data: 多只股票数据
            param_ranges: 参数范围
            horizon: 周期
            objective: 优化目标 ('total_return', 'win_rate', 'profit_factor', 'sharpe')
        
        Returns:
            (最优参数, 最优表现)
        """
        # 使用分层优化代替传统网格搜索
        return self.hierarchical_optimization(stock_data, param_ranges, horizon, objective)
    
    def evaluate_strategy_on_multiple_stocks(self,
                                            stock_data: Dict[str, pd.DataFrame],
                                            strategy_config: Dict,
                                            horizon: str = 'short') -> Optional[Dict]:
        """在多只股票上评估策略"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_returns = []
        all_win_rates = []
        all_profit_factors = []
        total_trades = 0
        
        total = len(stock_data)
        pb = create_progress_bar(total, f'评估{horizon}策略')
        
        # 使用线程池实现并行处理，兼容Windows系统
        stock_items = list(stock_data.items())
        batch_size = min(8, total)  # 每批处理的股票数量
        processed_count = 0
        
        def evaluate_stock(item):
            """评估单只股票的策略表现"""
            ts_code, df = item
            return self.evaluate_strategy_on_stock(df, strategy_config, horizon)
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_stock = {
                executor.submit(evaluate_stock, item): item
                for item in stock_items
            }
            
            for future in as_completed(future_to_stock):
                item = future_to_stock[future]
                try:
                    perf = future.result()
                    if perf:
                        all_returns.append(perf['avg_return'])
                        all_win_rates.append(perf['win_rate'])
                        all_profit_factors.append(perf['profit_factor'])
                        total_trades += perf['total_trades']
                except Exception as e:
                    print(f"评估股票 {item[0]} 时出错: {e}")
                finally:
                    processed_count += 1
                    pb.update(processed_count)
        
        if not all_returns:
            return None
        
        return {
            'avg_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'avg_win_rate': np.mean(all_win_rates),
            'avg_profit_factor': np.mean(all_profit_factors),
            'total_trades': total_trades,
            'sharpe_ratio': np.mean(all_returns) / (np.std(all_returns) + 1e-6) * np.sqrt(252)
        }
    
    def _calculate_objective_score(self, performance: Dict, objective: str) -> float:
        """根据优化目标计算分数"""
        if objective == 'total_return':
            return performance['avg_return']
        elif objective == 'win_rate':
            return performance['avg_win_rate']
        elif objective == 'profit_factor':
            return performance['avg_profit_factor']
        elif objective == 'sharpe':
            return performance['sharpe_ratio']
        else:
            return performance['avg_return']
    
    def _get_default_params(self, horizon: str) -> Dict:
        """获取每个周期的默认参数（最可能的组合）"""
        if horizon == 'short':
            return {
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
        elif horizon == 'medium':
            return {
                'ma20_weight': 10,
                'ma20_ma60_weight': 15,
                'macd_weight': 15,
                'trend_weight': 20,
                'obv_weight': 15,
                'volatility_weight': 15,
                'entry_threshold': 70
            }
        else:  # long
            return {
                'ma20_weight': 10,
                'ma20_ma60_weight': 15,
                'ma60_ma120_weight': 20,
                'trend_weight': 25,
                'near_high_weight': 15,
                'obv_weight': 15,
                'entry_threshold': 70
            }
    
    def _calculate_param_distance(self, params: Dict, default_params: Dict) -> float:
        """计算参数组合与默认参数的距离"""
        distance = 0.0
        
        for key, value in params.items():
            if key in default_params:
                default_value = default_params[key]
                # 计算绝对距离
                distance += abs(value - default_value)
        
        return distance
    
    def _get_importance_groups(self, horizon: str) -> Dict[str, List[str]]:
        """获取按重要性排序的指标分组"""
        if horizon == 'short':
            return {
                '最重要指标': ['entry_threshold', 'macd_positive_weight', 'kdj_weight'],
                '重要指标': ['ma5_weight', 'ma5_ma10_weight', 'ma10_ma20_weight'],
                '次要指标': ['rsi_low_weight', 'rsi_mid_weight', 'macd_cross_weight']
            }
        elif horizon == 'medium':
            return {
                '最重要指标': ['entry_threshold', 'trend_weight', 'ma20_ma60_weight'],
                '重要指标': ['macd_weight', 'ma20_weight'],
                '次要指标': ['obv_weight', 'volatility_weight']
            }
        else:  # long
            return {
                '最重要指标': ['entry_threshold', 'trend_weight', 'ma60_ma120_weight'],
                '重要指标': ['ma20_ma60_weight', 'ma20_weight'],
                '次要指标': ['near_high_weight', 'obv_weight']
            }
    
    def optimize_all_horizons(self,
                             stock_data: Dict[str, pd.DataFrame],
                             base_configs: Dict = None) -> Dict[str, Any]:
        """
        优化所有周期（短、中、长）的策略参数
        
        Args:
            stock_data: 股票数据
            base_configs: 基础配置（可选）
        
        Returns:
            各周期最优策略配置
        """
        if base_configs is None:
            base_configs = {
                'short': self._get_default_param_ranges('short'),
                'medium': self._get_default_param_ranges('medium'),
                'long': self._get_default_param_ranges('long')
            }
        
        results = {}
        horizons = ['short', 'medium', 'long']
        total_horizons = len(horizons)
        
        pb = create_progress_bar(total_horizons, '优化所有周期策略')
        
        for i, horizon in enumerate(horizons):
            print(f"\n{'='*60}")
            print(f"优化 {horizon} 周期策略...")
            print(f"{'='*60}")
            
            best_params, best_perf = self.grid_search_optimization(
                stock_data,
                base_configs[horizon],
                horizon=horizon,
                objective='sharpe'
            )
            
            results[horizon] = {
                'best_params': best_params,
                'best_performance': best_perf
            }
            
            print(f"\n{horizon} 周期最优参数:")
            print(best_params)
            print(f"\n表现:")
            print(f"  平均收益率: {best_perf['avg_return']*100:.2f}%")
            print(f"  平均胜率: {best_perf['avg_win_rate']*100:.2f}%")
            print(f"  平均盈亏比: {best_perf['avg_profit_factor']:.2f}")
            print(f"  夏普比率: {best_perf['sharpe_ratio']:.2f}")
            
            pb.update(i + 1)
        
        return results
    
    def _get_default_param_ranges(self, horizon: str) -> Dict:
        """获取默认参数范围"""
        if horizon == 'short':
            return {
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
        elif horizon == 'medium':
            return {
                'ma20_weight': [5, 10, 15],
                'ma20_ma60_weight': [10, 15, 20],
                'macd_weight': [10, 15, 20],
                'trend_weight': [15, 20, 25],
                'obv_weight': [10, 15, 20],
                'volatility_weight': [10, 15, 20],
                'entry_threshold': [60, 70, 80]
            }
        else:
            return {
                'ma20_weight': [5, 10, 15],
                'ma20_ma60_weight': [10, 15, 20],
                'ma60_ma120_weight': [15, 20, 25],
                'trend_weight': [20, 25, 30],
                'near_high_weight': [10, 15, 20],
                'obv_weight': [10, 15, 20],
                'entry_threshold': [60, 70, 80]
            }


strategy_optimizer = StrategyOptimizer()
