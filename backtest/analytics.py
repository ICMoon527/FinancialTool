import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
import matplotlib.pyplot as plt
from itertools import product
from logger import log

class PerformanceAnalyzer:
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series, 
                                     benchmark_returns: Optional[pd.Series] = None,
                                     risk_free_rate: float = 0.03) -> Dict[str, float]:
        if returns.empty:
            return {}
        
        returns = returns.dropna()
        
        total_return = (1 + returns).prod() - 1
        
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        annual_volatility = returns.std() * np.sqrt(252)
        
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility != 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_std if downside_std != 0 else 0
        
        rolling_max = (1 + returns).cumprod()
        drawdown = rolling_max / rolling_max.cummax() - 1
        max_drawdown = drawdown.min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        win_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = win_days / total_days if total_days > 0 else 0
        
        avg_win = returns[returns > 0].mean()
        avg_loss = returns[returns < 0].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.reindex(returns.index).dropna()
            if not benchmark_returns.empty:
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                alpha, beta = PerformanceAnalyzer.calculate_alpha_beta(returns, benchmark_returns, risk_free_rate)
                metrics['benchmark_return'] = benchmark_total_return
                metrics['alpha'] = alpha
                metrics['beta'] = beta
                metrics['excess_return'] = total_return - benchmark_total_return
        
        return metrics
    
    @staticmethod
    def calculate_alpha_beta(returns: pd.Series, 
                           benchmark_returns: pd.Series, 
                           risk_free_rate: float = 0.03) -> Tuple[float, float]:
        daily_rf = risk_free_rate / 252
        
        excess_returns = returns - daily_rf
        excess_benchmark = benchmark_returns - daily_rf
        
        covariance = np.cov(excess_returns, excess_benchmark)[0, 1]
        benchmark_variance = excess_benchmark.var()
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        alpha = excess_returns.mean() - beta * excess_benchmark.mean()
        alpha_annual = alpha * 252
        
        return alpha_annual, beta
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.DataFrame, 
                         benchmark: Optional[pd.Series] = None,
                         title: str = '净值曲线'):
        plt.figure(figsize=(12, 6))
        
        plt.plot(equity_curve.index, equity_curve['total_value'], 
                label='策略净值', linewidth=2)
        
        if benchmark is not None:
            benchmark = benchmark.reindex(equity_curve.index)
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve['total_value'].iloc[0]
            plt.plot(benchmark.index, benchmark_normalized, 
                    label='基准净值', linewidth=2, linestyle='--')
        
        plt.title(title, fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt
    
    @staticmethod
    def plot_drawdown(equity_curve: pd.DataFrame, title: str = '回撤曲线'):
        rolling_max = equity_curve['total_value'].cummax()
        drawdown = (equity_curve['total_value'] - rolling_max) / rolling_max
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, drawdown, 0, 
                        color='red', alpha=0.3, label='回撤')
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        
        plt.title(title, fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('回撤', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt
    
    @staticmethod
    def plot_monthly_returns(returns: pd.Series, title: str = '月度收益热力图'):
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly
        })
        
        heatmap_data = monthly_df.pivot(index='year', columns='month', values='return')
        
        plt.figure(figsize=(14, 8))
        plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                   vmin=-0.2, vmax=0.2)
        
        plt.xticks(range(12), ['1月', '2月', '3月', '4月', '5月', '6月',
                               '7月', '8月', '9月', '10月', '11月', '12月'])
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        
        for i in range(len(heatmap_data.index)):
            for j in range(12):
                val = heatmap_data.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.05 else 'black'
                    plt.text(j, i, f'{val:.1%}', ha='center', va='center', color=color)
        
        plt.title(title, fontsize=14)
        plt.colorbar(label='月度收益')
        plt.tight_layout()
        
        return plt
    
    @staticmethod
    def generate_report(results: Dict[str, Any]) -> str:
        report = []
        report.append("=" * 60)
        report.append("回测绩效报告")
        report.append("=" * 60)
        
        report.append(f"初始资金: {results.get('initial_cash', 0):,.2f}")
        report.append(f"最终资金: {results.get('final_value', 0):,.2f}")
        report.append(f"总收益率: {results.get('total_return', 0):.2%}")
        report.append(f"年化收益率: {results.get('annual_return', 0):.2%}")
        report.append(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        report.append(f"最大回撤: {results.get('max_drawdown', 0):.2%}")
        report.append(f"总交易次数: {results.get('total_trades', 0)}")
        report.append(f"胜率: {results.get('win_rate', 0):.2%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)

class ParameterOptimizer:
    @staticmethod
    def grid_search(strategy_func: Callable, 
                   param_grid: Dict[str, List],
                   df: pd.DataFrame,
                   objective: str = 'sharpe_ratio',
                   **kwargs) -> Dict[str, Any]:
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_score = -np.inf
        best_params = {}
        results = []
        
        total_combinations = np.prod([len(v) for v in param_values])
        log.info(f"开始网格搜索，共 {total_combinations} 组参数组合")
        
        from utils.progress_bar import create_progress_bar
        pb = create_progress_bar(total_combinations, '网格搜索参数')
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            try:
                result = strategy_func(df, **params, **kwargs)
                score = result.get(objective, -np.inf)
                
                results.append({
                    'params': params,
                    'score': score,
                    'result': result
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    log.info(f"找到新最优参数: {params}, {objective} = {score:.4f}")
            
            except Exception as e:
                log.error(f"参数组合 {params} 执行失败: {e}")
                continue
            finally:
                pb.update(i + 1)
        
        pb.finish()
        log.info(f"网格搜索完成，最优参数: {best_params}, 最优{objective}: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    @staticmethod
    def walk_forward_analysis(strategy_func: Callable,
                             df: pd.DataFrame,
                             train_period: int = 252,
                             test_period: int = 63,
                             param_grid: Optional[Dict[str, List]] = None,
                             **kwargs) -> Dict[str, Any]:
        df = df.sort_values('trade_date').reset_index(drop=True)
        total_days = len(df)
        
        all_results = []
        oos_returns = []
        
        start_idx = 0
        while start_idx + train_period + test_period <= total_days:
            train_end = start_idx + train_period
            test_end = train_end + test_period
            
            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            if param_grid is not None:
                opt_result = ParameterOptimizer.grid_search(
                    strategy_func, param_grid, train_df, **kwargs
                )
                best_params = opt_result['best_params']
            else:
                best_params = {}
            
            test_result = strategy_func(test_df, **best_params, **kwargs)
            
            all_results.append({
                'period': start_idx,
                'train_start': train_df['trade_date'].iloc[0],
                'train_end': train_df['trade_date'].iloc[-1],
                'test_start': test_df['trade_date'].iloc[0],
                'test_end': test_df['trade_date'].iloc[-1],
                'params': best_params,
                'result': test_result
            })
            
            if 'returns' in test_result:
                oos_returns.extend(test_result['returns'])
            
            start_idx += test_period
        
        log.info(f"滚动分析完成，共 {len(all_results)} 个周期")
        
        return {
            'all_periods': all_results,
            'oos_returns': oos_returns
        }

performance_analyzer = PerformanceAnalyzer()
parameter_optimizer = ParameterOptimizer()
