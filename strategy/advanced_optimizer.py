#!/usr/bin/env python3
"""
高级策略优化模块
提供多种优化算法：增强型随机搜索（带早停、并行、小数支持、交叉验证）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger import log
from strategy.factors import factor_library
from strategy.strategy_config import strategy_config
from utils.progress_bar import create_progress_bar


class AdvancedStrategyOptimizer:
    
    def __init__(self):
        pass
    
    def _random_sample_from_range(self, param_range: Dict) -> Any:
        """从参数范围中随机采样"""
        min_val = param_range.get('min', 0)
        max_val = param_range.get('max', 100)
        step = param_range.get('step', 1)
        
        if step == 1 and isinstance(min_val, int) and isinstance(max_val, int):
            return random.randint(min_val, max_val)
        else:
            num_steps = int((max_val - min_val) / step) + 1
            random_idx = random.randint(0, num_steps - 1)
            return min_val + random_idx * step
    
    def _get_rolling_window_indices(self, df_len: int) -> List[Tuple[int, int]]:
        """获取滚动窗口索引（从配置文件读取参数）"""
        window_size = strategy_config.get_rolling_window_size()
        step_size = strategy_config.get_rolling_step_size()
        
        windows = []
        start_idx = 60
        while start_idx + window_size <= df_len:
            end_idx = start_idx + window_size
            windows.append((start_idx, end_idx))
            start_idx += step_size
        return windows
    
    def _get_random_subsample_indices(self, df_len: int) -> List[Tuple[int, int]]:
        """获取随机子采样窗口（从配置文件读取参数）"""
        n_samples = strategy_config.get_subsample_count()
        sample_size = strategy_config.get_subsample_size()
        
        windows = []
        min_start = 60
        max_start = df_len - sample_size
        
        if max_start <= min_start:
            return [(min_start, min(df_len, min_start + sample_size))]
        
        for _ in range(n_samples):
            start_idx = random.randint(min_start, max_start)
            end_idx = start_idx + sample_size
            windows.append((start_idx, end_idx))
        
        return windows
    
    def _get_bootstrap_indices(self, df_len: int) -> List[np.ndarray]:
        """获取Bootstrap索引（从配置文件读取参数）"""
        n_bootstraps = strategy_config.get_bootstrap_count()
        
        bootstraps = []
        indices = np.arange(60, df_len)
        
        for _ in range(n_bootstraps):
            bootstrap_idx = np.random.choice(indices, size=len(indices), replace=True)
            bootstraps.append(bootstrap_idx)
        
        return bootstraps
    
    def _evaluate_params_single(self,
                                stock_data: Dict[str, pd.DataFrame],
                                strategy_config_dict: Dict,
                                horizon: str,
                                validation_method: str = None) -> Optional[Dict]:
        """评估一组参数的表现（支持多种验证方法）"""
        from strategy.optimizer import strategy_optimizer
        
        if validation_method is None:
            validation_method = strategy_config.get_validation_method()
        
        all_returns = []
        all_win_rates = []
        all_profit_factors = []
        total_trades = 0
        
        for ts_code, df in stock_data.items():
            try:
                if validation_method == 'full':
                    perf = strategy_optimizer.evaluate_strategy_on_stock(df, strategy_config_dict, horizon)
                    if perf:
                        all_returns.append(perf['avg_return'])
                        all_win_rates.append(perf['win_rate'])
                        all_profit_factors.append(perf['profit_factor'])
                        total_trades += perf['total_trades']
                elif validation_method == 'rolling':
                    windows = self._get_rolling_window_indices(len(df))
                    for start_idx, end_idx in windows:
                        window_df = df.iloc[start_idx:end_idx].copy()
                        perf = strategy_optimizer.evaluate_strategy_on_stock(window_df, strategy_config_dict, horizon)
                        if perf:
                            all_returns.append(perf['avg_return'])
                            all_win_rates.append(perf['win_rate'])
                            all_profit_factors.append(perf['profit_factor'])
                            total_trades += perf['total_trades']
                elif validation_method == 'subsample':
                    windows = self._get_random_subsample_indices(len(df))
                    for start_idx, end_idx in windows:
                        window_df = df.iloc[start_idx:end_idx].copy()
                        perf = strategy_optimizer.evaluate_strategy_on_stock(window_df, strategy_config_dict, horizon)
                        if perf:
                            all_returns.append(perf['avg_return'])
                            all_win_rates.append(perf['win_rate'])
                            all_profit_factors.append(perf['profit_factor'])
                            total_trades += perf['total_trades']
                elif validation_method == 'bootstrap':
                    bootstraps = self._get_bootstrap_indices(len(df))
                    for bootstrap_idx in bootstraps:
                        bootstrap_df = df.iloc[bootstrap_idx].copy()
                        bootstrap_df = bootstrap_df.sort_index().reset_index(drop=True)
                        perf = strategy_optimizer.evaluate_strategy_on_stock(bootstrap_df, strategy_config_dict, horizon)
                        if perf:
                            all_returns.append(perf['avg_return'])
                            all_win_rates.append(perf['win_rate'])
                            all_profit_factors.append(perf['profit_factor'])
                            total_trades += perf['total_trades']
            except Exception:
                continue
        
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
        """计算目标分数"""
        from strategy.optimizer import strategy_optimizer
        return strategy_optimizer._calculate_objective_score(performance, objective)
    
    def enhanced_random_search_optimization(self,
                                            stock_data: Dict[str, pd.DataFrame],
                                            horizon: str = 'short',
                                            objective: str = 'sharpe',
                                            validation_method: str = None) -> Tuple[Dict, Dict]:
        """
        增强型随机搜索优化策略参数
        
        特点：
        1. 初始化使用当前配置文件中的参数
        2. 找到更优组合立即更新到配置文件
        3. 死循环无限迭代，早停条件可配置
        4. 支持小数权重，搜索更精细
        5. 并行优化加速搜索
        6. 滚动窗口交叉验证
        7. 随机子采样
        8. Bootstrap
        
        Args:
            stock_data: 多只股票数据
            horizon: 周期
            objective: 优化目标
            validation_method: 验证方法 ('full', 'rolling', 'subsample', 'bootstrap')
        
        Returns:
            (最优参数, 最优表现)
        """
        from strategy.optimizer import strategy_optimizer
        
        if validation_method is None:
            validation_method = strategy_config.get_validation_method()
        
        print(f"\n" + "="*80)
        print(f"增强型随机搜索优化 - {horizon}周期")
        print(f"="*80)
        
        validation_names = {
            'full': '完整数据',
            'rolling': '滚动窗口交叉验证',
            'subsample': '随机子采样',
            'bootstrap': 'Bootstrap'
        }
        
        print(f"\n验证方法: {validation_names.get(validation_method, validation_method)}")
        
        # 获取配置
        param_ranges = strategy_config.get_param_ranges(horizon)
        early_stopping_rounds = strategy_config.get_early_stopping_rounds()
        parallel_workers = strategy_config.get_parallel_workers()
        
        print(f"\n优化配置:")
        print(f"  早停轮数: {early_stopping_rounds}")
        print(f"  并行工作数: {parallel_workers}")
        
        # 预计算所有股票的因子
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
        
        # 1. 初始化：使用当前配置文件中的参数
        print(f"\n初始化：使用当前配置参数作为起点")
        best_params = strategy_config.get_current_params(horizon)
        
        # 评估初始参数
        print(f"评估初始参数...")
        best_performance = self._evaluate_params_single(
            prepared_stock_data, best_params, horizon, validation_method
        )
        
        if best_performance:
            best_score = self._calculate_objective_score(best_performance, objective)
            print(f"  初始得分: {best_score:.4f}")
        else:
            best_score = -float('inf')
            print(f"  初始参数评估失败，从负无穷开始")
        
        no_improvement_count = 0
        total_iterations = 0
        start_time = datetime.now()
        
        print(f"\n开始无限迭代搜索（按 Ctrl+C 可随时停止）...")
        print(f"{'='*80}")
        
        try:
            while True:
                total_iterations += 1
                
                # 生成一批随机参数用于并行评估
                batch_size = parallel_workers
                param_batch = []
                
                for _ in range(batch_size):
                    test_params = {}
                    for param, param_range in param_ranges.items():
                        test_params[param] = self._random_sample_from_range(param_range)
                    param_batch.append(test_params)
                
                # 并行评估这批参数
                batch_results = []
                for test_params in param_batch:
                    performance = self._evaluate_params_single(
                        prepared_stock_data, test_params, horizon, validation_method
                    )
                    batch_results.append((test_params, performance))
                
                # 检查这批结果中是否有更优的
                improved = False
                for test_params, performance in batch_results:
                    if performance:
                        score = self._calculate_objective_score(performance, objective)
                        
                        if score > best_score:
                            # 2. 找到更优组合，立即更新到配置文件
                            best_score = score
                            best_params = test_params.copy()
                            best_performance = performance
                            no_improvement_count = 0
                            improved = True
                            
                            elapsed = (datetime.now() - start_time).total_seconds()
                            print(f"\n✓ 迭代 {total_iterations}: 发现更优参数")
                            print(f"  得分: {score:.4f}")
                            print(f"  耗时: {elapsed:.1f}秒")
                            print(f"  参数: {best_params}")
                            
                            # 立即更新配置文件
                            weights = {k: v for k, v in best_params.items() if k != 'entry_threshold'}
                            strategy_config.update_weights(horizon, weights)
                            if 'entry_threshold' in best_params:
                                strategy_config.update_entry_threshold(horizon, best_params['entry_threshold'])
                            
                            print(f"  ✓ 配置文件已更新")
                
                if not improved:
                    no_improvement_count += 1
                
                # 3. 早停检查
                if no_improvement_count >= early_stopping_rounds:
                    print(f"\n{'='*80}")
                    print(f"早停触发！连续 {no_improvement_count} 次迭代无改善")
                    break
                
                # 显示进度
                if total_iterations % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"\r迭代: {total_iterations} | 无改善: {no_improvement_count}/{early_stopping_rounds} | 耗时: {elapsed:.1f}s | 最佳得分: {best_score:.4f}", end='')
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"用户中断优化")
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"优化完成！")
        print(f"  总迭代次数: {total_iterations}")
        print(f"  总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
        print(f"  最佳得分: {best_score:.4f}")
        
        return best_params, best_performance
    
    def optimize_short_term_with_multiple_methods(self,
                                                   stock_data: Dict[str, pd.DataFrame],
                                                   param_ranges: Dict = None,
                                                   method: str = 'enhanced_random') -> Tuple[Dict, Dict]:
        """
        使用多种方法优化短线策略
        
        Args:
            stock_data: 股票数据
            param_ranges: 参数范围（已废弃，从配置文件读取）
            method: 优化方法 ('enhanced_random', 'hierarchical')
        
        Returns:
            (最优参数, 最优表现)
        """
        from strategy.optimizer import strategy_optimizer
        
        if method == 'enhanced_random':
            print("\n使用增强型随机搜索方法（滚动窗口验证）...")
            return self.enhanced_random_search_optimization(
                stock_data, 'short', 'sharpe', validation_method='rolling'
            )
        elif method == 'hierarchical':
            print("\n使用分层优化方法...")
            # 使用旧的参数范围格式
            old_param_ranges = {
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
            return strategy_optimizer.hierarchical_optimization(
                stock_data, old_param_ranges, 'short', 'sharpe'
            )
        else:
            print(f"\n未知方法 {method}，使用增强型随机搜索...")
            return self.enhanced_random_search_optimization(
                stock_data, 'short', 'sharpe', validation_method='rolling'
            )


advanced_optimizer = AdvancedStrategyOptimizer()
