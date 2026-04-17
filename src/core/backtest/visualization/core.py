# -*- coding: utf-8 -*-
"""
回测结果可视化模块。

实现净值曲线、回撤曲线和指标热力图的绘制。
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.engine import Portfolio
from ..metrics.performance import PerformanceMetrics

logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


class BacktestVisualizer:
    """
    回测结果可视化器。
    """

    def __init__(
        self,
        portfolio: Portfolio,
        metrics: Optional[PerformanceMetrics] = None,
        output_dir: str = "backtest_results",
    ):
        """
        初始化可视化器。

        Args:
            portfolio: 投资组合
            metrics: 绩效指标计算器
            output_dir: 输出目录
        """
        self.portfolio = portfolio
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._equity_df = self._build_equity_dataframe()

    def _build_equity_dataframe(self) -> pd.DataFrame:
        """
        构建权益数据DataFrame。

        Returns:
            权益数据DataFrame
        """
        if not self.portfolio.equity_history:
            return pd.DataFrame(columns=["date", "equity"])

        dates, equities = zip(*self.portfolio.equity_history)
        df = pd.DataFrame({"date": dates, "equity": equities})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def plot_equity_curve(
        self,
        title: str = "净值曲线",
        benchmark_equity: Optional[List[float]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        绘制净值曲线。

        Args:
            title: 图表标题
            benchmark_equity: 基准净值序列
            save_path: 保存路径

        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            self._equity_df["date"],
            self._equity_df["equity"],
            label="策略净值",
            linewidth=2,
            color="#1f77b4",
        )

        if benchmark_equity is not None and len(benchmark_equity) == len(self._equity_df):
            ax.plot(
                self._equity_df["date"],
                benchmark_equity,
                label="基准净值",
                linewidth=2,
                linestyle="--",
                color="#ff7f0e",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("净值", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = str(self.output_dir / "equity_curve.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"净值曲线已保存至: {save_path}")
        return save_path

    def plot_drawdown_curve(
        self,
        title: str = "回撤曲线",
        save_path: Optional[str] = None,
    ) -> str:
        """
        绘制回撤曲线。

        Args:
            title: 图表标题
            save_path: 保存路径

        Returns:
            保存的文件路径
        """
        if self._equity_df.empty or len(self._equity_df) < 2:
            logger.warning("数据不足，无法绘制回撤曲线")
            return ""

        df = self._equity_df.copy()
        df["cummax"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["cummax"]) / df["cummax"] * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(
            df["date"],
            df["drawdown"],
            0,
            color="#d62728",
            alpha=0.5,
            label="回撤",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("回撤 (%)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = str(self.output_dir / "drawdown_curve.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"回撤曲线已保存至: {save_path}")
        return save_path

    def plot_metrics_heatmap(
        self,
        title: str = "绩效指标热力图",
        save_path: Optional[str] = None,
    ) -> str:
        """
        绘制绩效指标热力图。

        Args:
            title: 图表标题
            save_path: 保存路径

        Returns:
            保存的文件路径
        """
        if self.metrics is None:
            logger.warning("绩效指标未提供，无法绘制热力图")
            return ""

        all_metrics = self.metrics.get_all_metrics()

        key_metrics = {
            "总收益率": all_metrics.get("total_return", 0) * 100,
            "年化收益率": all_metrics.get("annualized_return", 0) * 100,
            "波动率": all_metrics.get("volatility", 0) * 100,
            "夏普比率": all_metrics.get("sharpe_ratio", 0),
            "最大回撤": all_metrics.get("max_drawdown", 0) * 100,
            "卡尔马比率": all_metrics.get("calmar_ratio", 0),
            "胜率": all_metrics.get("win_rate", 0) * 100,
            "盈亏比": all_metrics.get("profit_loss_ratio", 0),
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_df = pd.DataFrame(
            list(key_metrics.items()),
            columns=["指标", "数值"],
        )
        metrics_df = metrics_df.set_index("指标")

        sns.heatmap(
            metrics_df.T,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": "数值"},
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

        if save_path is None:
            save_path = str(self.output_dir / "metrics_heatmap.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"指标热力图已保存至: {save_path}")
        return save_path

    def plot_metrics_radar(
        self,
        title: str = "绩效指标雷达图",
        save_path: Optional[str] = None,
    ) -> str:
        """
        绘制绩效指标雷达图。

        Args:
            title: 图表标题
            save_path: 保存路径

        Returns:
            保存的文件路径
        """
        if self.metrics is None:
            logger.warning("绩效指标未提供，无法绘制雷达图")
            return ""

        all_metrics = self.metrics.get_all_metrics()

        metrics_list = [
            ("年化收益率", all_metrics.get("annualized_return", 0) * 100, 50),
            ("夏普比率", all_metrics.get("sharpe_ratio", 0), 3),
            ("卡尔马比率", all_metrics.get("calmar_ratio", 0), 5),
            ("胜率", all_metrics.get("win_rate", 0) * 100, 100),
            ("盈亏比", all_metrics.get("profit_loss_ratio", 0), 5),
        ]

        labels = [m[0] for m in metrics_list]
        values = [min(m[1] / m[2], 1.0) for m in metrics_list]

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        ax.plot(angles, values, "o-", linewidth=2, label="策略表现")
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight="bold", y=1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        if save_path is None:
            save_path = str(self.output_dir / "metrics_radar.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"指标雷达图已保存至: {save_path}")
        return save_path

    def plot_all(
        self,
        benchmark_equity: Optional[List[float]] = None,
    ) -> Dict[str, str]:
        """
        绘制所有图表。

        Args:
            benchmark_equity: 基准净值序列

        Returns:
            图表文件路径字典
        """
        results = {}

        results["equity_curve"] = self.plot_equity_curve(benchmark_equity=benchmark_equity)
        results["drawdown_curve"] = self.plot_drawdown_curve()
        results["metrics_heatmap"] = self.plot_metrics_heatmap()
        results["metrics_radar"] = self.plot_metrics_radar()

        return results
