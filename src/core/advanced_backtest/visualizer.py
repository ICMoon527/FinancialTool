# -*- coding: utf-8 -*-
"""
===================================
回测结果可视化模块
===================================

提供回测结果的图表可视化功能。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from .engine import BacktestResult

logger = logging.getLogger(__name__)


class Visualizer:
    """
    回测结果可视化器
    """
    
    def __init__(self, result: BacktestResult):
        self.result = result
        self.figures: Dict[str, plt.Figure] = {}
    
    def plot_equity_curve(
        self,
        title: str = "权益曲线",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        绘制权益曲线
        
        Args:
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            matplotlib Figure
        """
        if self.result.equity_curve.empty:
            logger.warning("没有权益曲线数据")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制策略权益曲线
        ax.plot(
            pd.to_datetime(self.result.equity_curve["date"]),
            self.result.equity_curve["equity"],
            label="策略",
            linewidth=2,
            color="#1f77b4",
        )
        
        # 绘制基准（如果有）
        if self.result.benchmark_curve is not None and not self.result.benchmark_curve.empty:
            # 标准化基准（从 1 开始）
            benchmark = self.result.benchmark_curve.copy()
            if "equity" in benchmark.columns:
                benchmark_norm = benchmark["equity"] / benchmark["equity"].iloc[0]
                # 应用到初始资金
                benchmark_equity = benchmark_norm * self.result.config.initial_capital
                
                ax.plot(
                    pd.to_datetime(benchmark["date"]),
                    benchmark_equity,
                    label="基准",
                    linewidth=2,
                    color="#ff7f0e",
                    linestyle="--",
                )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("权益", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        # 格式化日期
        fig.autofmt_xdate()
        
        self.figures["equity_curve"] = fig
        return fig
    
    def plot_drawdown(
        self,
        title: str = "回撤曲线",
        figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """
        绘制回撤曲线
        
        Args:
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            matplotlib Figure
        """
        if self.result.equity_curve.empty:
            logger.warning("没有权益曲线数据")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算回撤
        df = self.result.equity_curve.copy()
        df["cummax"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["cummax"]) / df["cummax"] * 100
        
        ax.fill_between(
            pd.to_datetime(df["date"]),
            df["drawdown"],
            0,
            alpha=0.3,
            color="#d62728",
        )
        ax.plot(
            pd.to_datetime(df["date"]),
            df["drawdown"],
            linewidth=1.5,
            color="#d62728",
        )
        
        # 标注最大回撤
        max_dd_idx = df["drawdown"].idxmin()
        max_dd_date = df["date"].iloc[max_dd_idx]
        max_dd_value = df["drawdown"].iloc[max_dd_idx]
        
        ax.scatter(
            pd.to_datetime(max_dd_date),
            max_dd_value,
            color="#d62728",
            s=100,
            zorder=5,
            label=f"最大回撤: {max_dd_value:.2f}%",
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("回撤 (%)", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        
        fig.autofmt_xdate()
        
        self.figures["drawdown"] = fig
        return fig
    
    def plot_performance_metrics(
        self,
        title: str = "绩效指标",
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        绘制绩效指标图表
        
        Args:
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            matplotlib Figure
        """
        metrics = self.result.performance_metrics
        
        if not metrics:
            logger.warning("没有绩效指标数据")
            return None
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 收益率指标
        ax1 = fig.add_subplot(gs[0, 0])
        return_metrics = {
            "总收益率": metrics.get("total_return", 0) * 100,
            "年化收益率": metrics.get("annualized_return", 0) * 100,
        }
        
        colors = ["#1f77b4", "#2ca02c"]
        bars = ax1.bar(
            list(return_metrics.keys()),
            list(return_metrics.values()),
            color=colors,
            alpha=0.8,
        )
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        
        ax1.set_title("收益率指标", fontsize=12, fontweight="bold")
        ax1.set_ylabel("收益率 (%)")
        ax1.grid(True, alpha=0.3, axis="y")
        
        # 2. 风险指标
        ax2 = fig.add_subplot(gs[0, 1])
        risk_metrics = {
            "年化波动率": metrics.get("annualized_volatility", 0) * 100,
            "最大回撤": abs(metrics.get("max_drawdown", 0)) * 100,
        }
        
        colors = ["#ff7f0e", "#d62728"]
        bars = ax2.bar(
            list(risk_metrics.keys()),
            list(risk_metrics.values()),
            color=colors,
            alpha=0.8,
        )
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        
        ax2.set_title("风险指标", fontsize=12, fontweight="bold")
        ax2.set_ylabel("波动率/回撤 (%)")
        ax2.grid(True, alpha=0.3, axis="y")
        
        # 3. 风险调整后收益
        ax3 = fig.add_subplot(gs[1, 0])
        sharpe = metrics.get("sharpe_ratio", 0)
        
        colors = ["#9467bd" if sharpe >= 1 else "#8c564b"]
        bars = ax3.bar(
            ["夏普比率"],
            [sharpe],
            color=colors,
            alpha=0.8,
        )
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        
        ax3.set_title("风险调整后收益", fontsize=12, fontweight="bold")
        ax3.axhline(y=1, color="#9467bd", linestyle="--", alpha=0.5, label="优秀线")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3, axis="y")
        
        # 4. 交易统计
        ax4 = fig.add_subplot(gs[1, 1])
        trade_metrics = {
            "总交易次数": metrics.get("total_trades", 0),
            "胜率": metrics.get("win_rate", 0) * 100,
        }
        
        colors = ["#17becf", "#e377c2"]
        bars = ax4.bar(
            list(trade_metrics.keys()),
            list(trade_metrics.values()),
            color=colors,
            alpha=0.8,
        )
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label = f"{height:.0f}" if i == 0 else f"{height:.2f}%"
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        
        ax4.set_title("交易统计", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")
        
        fig.suptitle(title, fontsize=16, fontweight="bold")
        
        self.figures["performance_metrics"] = fig
        return fig
    
    def save_figure(
        self,
        figure_name: str,
        filepath: str,
        dpi: int = 150,
        bbox_inches: str = "tight",
    ) -> bool:
        """
        保存图表
        
        Args:
            figure_name: 图表名称（在 self.figures 中的 key）
            filepath: 保存路径
            dpi: DPI
            bbox_inches: 边框设置
            
        Returns:
            是否成功
        """
        if figure_name not in self.figures:
            logger.warning(f"图表不存在: {figure_name}")
            return False
        
        try:
            self.figures[figure_name].savefig(
                filepath,
                dpi=dpi,
                bbox_inches=bbox_inches,
            )
            logger.info(f"图表已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存图表失败: {e}")
            return False
    
    def save_all_figures(
        self,
        output_dir: str,
        prefix: str = "backtest",
        dpi: int = 150,
    ) -> Dict[str, str]:
        """
        保存所有图表
        
        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
            dpi: DPI
            
        Returns:
            图表名称 -> 文件路径 的字典
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for name, fig in self.figures.items():
            filepath = os.path.join(output_dir, f"{prefix}_{name}.png")
            if self.save_figure(name, filepath, dpi=dpi):
                saved_files[name] = filepath
        
        return saved_files
    
    def close_all(self) -> None:
        """关闭所有图表，释放内存"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()
