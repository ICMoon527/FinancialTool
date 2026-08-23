# -*- coding: utf-8 -*-
"""
使用QuantStats库生成专业的回测分析报告
"""

import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import quantstats as qs
    from quantstats import stats, plots, reports
except ImportError:
    print("Warning: QuantStats not installed. Please install with 'pip install quantstats'")
    qs = None

from .engine import Portfolio

logger = logging.getLogger(__name__)


class QuantStatsVisualizer:
    """
    使用QuantStats库的回测可视化工具
    """

    def __init__(
        self,
        portfolio: Portfolio,
        output_dir: str = "strategy_backtest_results",
        benchmark_code: str = "sh000001"  # 默认使用上证指数
    ):
        """
        初始化QuantStats可视化工具
        
        Args:
            portfolio: 回测投资组合对象
            output_dir: 输出目录
            benchmark_code: 基准指数代码（sh000001=上证指数，sz399001=深证成指）
        """
        self.portfolio = portfolio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_code = benchmark_code
        
        self.returns_series = self._build_returns_series()
        self.benchmark_returns = self._load_benchmark_returns()
        
    def _load_benchmark_returns(self) -> Optional[pd.Series]:
        """
        加载基准指数收益率
        
        Returns:
            pandas Series（日期索引，日收益率）
        """
        cache_file = Path(f"data/cache/market_{self.benchmark_code}.pkl")
        
        if not cache_file.exists():
            logger.warning(f"基准指数缓存文件不存在: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, pd.DataFrame) and 'close' in data.columns and 'date' in data.columns:
                df = data.copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                returns = df['close'].pct_change()
                returns = returns.dropna()
                
                logger.info(f"加载基准指数 {self.benchmark_code}，{len(returns)}个数据点")
                return returns
            
            logger.warning("基准指数数据格式不正确")
            return None
            
        except Exception as e:
            logger.error(f"加载基准指数失败: {e}")
            return None
    
    def _align_returns_with_benchmark(self):
        """
        对齐策略收益率和基准收益率的日期
        """
        if self.returns_series is None or self.benchmark_returns is None:
            return self.returns_series, None
        
        # 获取共同日期
        common_dates = self.returns_series.index.intersection(self.benchmark_returns.index)
        
        aligned_strategy = self.returns_series.loc[common_dates].copy()
        aligned_benchmark = self.benchmark_returns.loc[common_dates].copy()
        
        logger.info(f"对齐后日期数量: {len(common_dates)}")
        
        return aligned_strategy, aligned_benchmark
    
    def _build_returns_series(self) -> Optional[pd.Series]:
        """
        从equity_history构建收益率Series，供QuantStats使用
        
        Returns:
            pandas Series（日期索引，日收益率）
        """
        if not self.portfolio.equity_history:
            logger.warning("没有equity_history数据")
            return None
        
        dates = []
        equities = []
        
        # 兼容两种数据格式
        for entry in self.portfolio.equity_history:
            # 格式1: 字典格式 {"date": ..., "equity": ...}
            if isinstance(entry, dict):
                dates.append(entry['date'])
                equities.append(entry['equity'])
            # 格式2: 元组格式 (date, equity)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                dates.append(entry[0])
                equities.append(entry[1])
            else:
                logger.warning(f"无法识别的equity_history条目格式: {type(entry)}")
        
        if not dates:
            logger.warning("无法从equity_history中提取有效数据")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'equity': equities
        })
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 计算日收益率
        returns = df['equity'].pct_change()
        returns = returns.dropna()
        
        if len(returns) < 2:
            logger.warning("收益率数据不足")
            return None
        
        logger.info(f"构建收益率Series，{len(returns)}个数据点")
        return returns
    
    def generate_html_report(
        self,
        filename: str = "quantstats_report.html",
        title: str = "策略回测分析报告"
    ) -> str:
        """
        生成完整的QuantStats HTML报告（包含基准对比）
        
        Args:
            filename: 保存的文件名
            title: 报告标题
            
        Returns:
            保存的文件路径
        """
        if qs is None:
            logger.error("QuantStats未安装，无法生成HTML报告")
            return ""
        
        if self.returns_series is None:
            logger.error("没有足够的数据生成报告")
            return ""
        
        filepath = self.output_dir / filename
        
        try:
            # 对齐数据
            strategy_returns, benchmark_returns = self._align_returns_with_benchmark()
            
            # 生成报告
            if benchmark_returns is not None:
                logger.info("生成包含基准对比的QuantStats HTML报告")
                reports.html(
                    strategy_returns,
                    benchmark=benchmark_returns,
                    title=title,
                    output=str(filepath),
                    download_filename=filename
                )
            else:
                logger.info("生成QuantStats HTML报告（无基准对比）")
                reports.html(
                    strategy_returns,
                    title=title,
                    output=str(filepath),
                    download_filename=filename
                )
            
            logger.info(f"QuantStats HTML报告已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"生成QuantStats HTML报告失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def plot_equity_curve(
        self,
        filename: str = "qs_equity_curve.png"
    ) -> str:
        """
        使用标准的 matplotlib 绘制净值曲线
        
        Args:
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        if qs is None:
            return ""
        
        if self.returns_series is None:
            return ""
        
        filepath = self.output_dir / filename
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # 首先计算累积收益
            equity = (1 + self.returns_series).cumprod()
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity.index, equity.values, linewidth=2, color="#1f77b4")
            ax.set_title("策略净值曲线")
            ax.set_ylabel("净值")
            ax.grid(True, alpha=0.3)
            
            # 自动调整日期标签
            fig.autofmt_xdate()
            
            # 保存图形
            fig.savefig(str(filepath), dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"净值曲线已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"绘制净值曲线失败: {e}")
            return ""
    
    def plot_drawdown_curve(
        self,
        filename: str = "qs_drawdown_curve.png"
    ) -> str:
        """
        使用QuantStats绘制回撤曲线
        
        Args:
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        if qs is None:
            return ""
        
        if self.returns_series is None:
            return ""
        
        filepath = self.output_dir / filename
        
        try:
            plots.drawdown(
                self.returns_series,
                figsize=(12, 6),
                savefig=str(filepath),
                show=False
            )
            
            logger.info(f"回撤曲线已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"绘制回撤曲线失败: {e}")
            return ""
    
    def plot_monthly_heatmap(
        self,
        filename: str = "qs_monthly_heatmap.png"
    ) -> str:
        """
        使用QuantStats绘制月度收益率热力图
        
        Args:
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        if qs is None:
            return ""
        
        if self.returns_series is None:
            return ""
        
        filepath = self.output_dir / filename
        
        try:
            plots.monthly_heatmap(
                self.returns_series,
                figsize=(12, 8),
                savefig=str(filepath),
                show=False
            )
            
            logger.info(f"月度热力图已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"绘制月度热力图失败: {e}")
            return ""
    
    def calculate_key_metrics(self) -> Dict[str, Any]:
        """
        计算完整的 30+ 风险收益指标
        
        Returns:
            指标字典
        """
        if qs is None or self.returns_series is None:
            return {}
        
        try:
            # 帮助函数: 获取 float 值
            def get_float(val):
                import pandas as pd
                if isinstance(val, pd.Series) and len(val) == 1:
                    return float(val.iloc[0])
                return float(val)
            
            # 对齐数据
            strategy_returns, benchmark_returns = self._align_returns_with_benchmark()
            
            metrics = {}
            
            # ========== 基础收益指标 ==========
            cagr_val = stats.cagr(strategy_returns)
            metrics['cagr'] = get_float(cagr_val)
            metrics['total_return'] = get_float(cagr_val) * len(strategy_returns) / 252
            
            # 年化收益（多种时间周期）
            metrics['annual_return'] = metrics['cagr']
            try:
                metrics['monthly_return'] = get_float(stats.monthly_returns(strategy_returns).mean() * 12)
            except:
                pass
            
            # ========== 风险指标 ==========
            metrics['volatility'] = get_float(stats.volatility(strategy_returns))
            metrics['volatility_monthly'] = get_float(stats.volatility(strategy_returns, periods=12))
            metrics['max_drawdown'] = get_float(stats.max_drawdown(strategy_returns))
            
            # ========== 风险调整收益指标 ==========
            metrics['sharpe'] = get_float(stats.sharpe(strategy_returns))
            metrics['sortino'] = get_float(stats.sortino(strategy_returns))
            metrics['calmar'] = get_float(stats.calmar(strategy_returns))
            metrics['omega'] = get_float(stats.omega(strategy_returns))
            metrics['risk_of_ruin'] = get_float(stats.risk_of_ruin(strategy_returns))
            metrics['value_at_risk'] = get_float(stats.value_at_risk(strategy_returns))
            
            # ========== 交易统计指标 ==========
            metrics['win_rate'] = get_float(stats.win_rate(strategy_returns))
            metrics['best_day'] = get_float(strategy_returns.max())
            metrics['worst_day'] = get_float(strategy_returns.min())
            metrics['avg_win'] = get_float(stats.avg_win(strategy_returns))
            metrics['avg_loss'] = get_float(stats.avg_loss(strategy_returns))
            metrics['profit_factor'] = get_float(stats.profit_factor(strategy_returns))
            metrics['gain_to_pain_ratio'] = get_float(stats.gain_to_pain_ratio(strategy_returns))
            
            # ========== 分布特征指标 ==========
            metrics['skew'] = get_float(stats.skew(strategy_returns))
            metrics['kurtosis'] = get_float(stats.kurtosis(strategy_returns))
            metrics['tail_ratio'] = get_float(stats.tail_ratio(strategy_returns))
            metrics['common_sense_ratio'] = get_float(stats.common_sense_ratio(strategy_returns))
            
            # ========== 基准对比指标 ==========
            if benchmark_returns is not None:
                try:
                    metrics['benchmark_cagr'] = get_float(stats.cagr(benchmark_returns))
                    metrics['benchmark_volatility'] = get_float(stats.volatility(benchmark_returns))
                    metrics['benchmark_sharpe'] = get_float(stats.sharpe(benchmark_returns))
                    metrics['benchmark_max_drawdown'] = get_float(stats.max_drawdown(benchmark_returns))
                    
                    # 相对指标
                    metrics['relative_return'] = metrics['cagr'] - metrics['benchmark_cagr']
                    metrics['information_ratio'] = get_float(stats.information_ratio(strategy_returns, benchmark_returns))
                    metrics['treynor_ratio'] = get_float(stats.treynor_ratio(strategy_returns, benchmark_returns))
                    
                    logger.info(f"成功计算包含基准对比的 {len(metrics)} 个指标")
                except Exception as e:
                    logger.warning(f"计算基准对比指标时出错: {e}")
            else:
                logger.info(f"成功计算 {len(metrics)} 个指标（无基准对比）")
            
            # ========== 转换为百分比显示 ==========
            for key in [
                'total_return', 'annual_return', 'monthly_return',
                'volatility', 'volatility_monthly',
                'max_drawdown',
                'win_rate', 'best_day', 'worst_day', 'avg_win', 'avg_loss',
                'benchmark_cagr', 'benchmark_volatility', 'benchmark_max_drawdown',
                'relative_return'
            ]:
                if key in metrics:
                    metrics[f"{key}_pct"] = metrics[key] * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算关键指标失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def plot_all(self, report_title: str = "策略回测分析报告") -> Dict[str, str]:
        """
        绘制所有图形并生成报告
        
        Args:
            report_title: 报告标题
            
        Returns:
            文件路径字典
        """
        results = {}
        
        logger.info("开始使用QuantStats生成分析报告和图形...")
        
        # 生成HTML报告
        results['html_report'] = self.generate_html_report(title=report_title)
        
        # 生成关键图形
        # 注：不再额外产出独立 PNG 图（qs_*.png），其内容已由 quantstats_report.html 内嵌 SVG 呈现
        # results['equity_curve'] = self.plot_equity_curve()
        # results['drawdown_curve'] = self.plot_drawdown_curve()
        # results['monthly_heatmap'] = self.plot_monthly_heatmap()
        
        # 计算并保存指标
        metrics = self.calculate_key_metrics()
        if metrics:
            metrics_file = self.output_dir / "quantstats_metrics.json"
            try:
                import json
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                results['metrics_json'] = str(metrics_file)
                logger.info(f"指标已保存到: {metrics_file}")
            except Exception as e:
                logger.error(f"保存指标JSON失败: {e}")
        
        # 统计成功的文件数量
        success_count = sum(1 for f in results.values() if f)
        
        logger.info(f"QuantStats可视化完成，生成 {success_count} 个文件")
        
        return results

