# -*- coding: utf-8 -*-
"""
参数敏感性测试模块。

测试策略在不同参数设置下的稳定性。
"""

from __future__ import annotations

import logging
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .engine import StrategyBacktestEngine, Portfolio
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class ParameterSensitivityTest:
    """
    参数敏感性测试器。
    """

    def __init__(
        self,
        engine_factory: Callable[..., StrategyBacktestEngine],
        strategy_factory: Callable[..., Any],
        output_dir: str = "backtest_results",
    ):
        """
        初始化参数敏感性测试器。

        Args:
            engine_factory: 回测引擎工厂函数
            strategy_factory: 策略工厂函数
            output_dir: 输出目录
        """
        self.engine_factory = engine_factory
        self.strategy_factory = strategy_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        trading_dates: List[date],
        stock_pool: List[str],
    ) -> pd.DataFrame:
        """
        运行网格搜索。

        Args:
            param_grid: 参数网格
            trading_dates: 交易日列表
            stock_pool: 股票池

        Returns:
            结果DataFrame
        """
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)

        logger.info(f"开始参数敏感性测试，共 {total_combinations} 种参数组合")

        for idx, param_combination in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, param_combination))
            logger.info(f"测试参数组合 {idx}/{total_combinations}: {params}")

            try:
                result = self._run_single_backtest(
                    params=params,
                    trading_dates=trading_dates,
                    stock_pool=stock_pool,
                )
                result.update(params)
                self.results.append(result)
            except Exception as e:
                logger.error(f"参数组合测试失败 {params}: {e}")

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_dir / "sensitivity_results.csv", index=False, encoding="utf-8-sig")

        logger.info(f"参数敏感性测试完成，结果已保存至: {self.output_dir / 'sensitivity_results.csv'}")

        return results_df

    def _run_single_backtest(
        self,
        params: Dict[str, Any],
        trading_dates: List[date],
        stock_pool: List[str],
    ) -> Dict[str, Any]:
        """
        运行单次回测。

        Args:
            params: 参数
            trading_dates: 交易日列表
            stock_pool: 股票池

        Returns:
            回测结果
        """
        engine = self.engine_factory()
        strategy = self.strategy_factory(**params)

        engine.set_strategy(strategy)
        engine.set_stock_pool(stock_pool)
        engine.set_trading_dates(trading_dates)

        portfolio = engine.run()

        metrics = PerformanceMetrics(portfolio)
        all_metrics = metrics.get_all_metrics()

        return {
            "total_return": all_metrics.get("total_return", 0),
            "annualized_return": all_metrics.get("annualized_return", 0),
            "sharpe_ratio": all_metrics.get("sharpe_ratio", 0),
            "max_drawdown": all_metrics.get("max_drawdown", 0),
            "calmar_ratio": all_metrics.get("calmar_ratio", 0),
            "win_rate": all_metrics.get("win_rate", 0),
            "profit_loss_ratio": all_metrics.get("profit_loss_ratio", 0),
        }

    def plot_parameter_impact(
        self,
        param_name: str,
        metric_name: str = "sharpe_ratio",
        save_path: Optional[str] = None,
    ) -> str:
        """
        绘制单个参数对指标的影响。

        Args:
            param_name: 参数名称
            metric_name: 指标名称
            save_path: 保存路径

        Returns:
            保存的文件路径
        """
        if not self.results:
            logger.warning("没有测试结果，无法绘制")
            return ""

        df = pd.DataFrame(self.results)

        if param_name not in df.columns or metric_name not in df.columns:
            logger.warning(f"参数或指标不存在: {param_name}, {metric_name}")
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        df_sorted = df.sort_values(param_name)
        ax.plot(
            df_sorted[param_name].astype(str),
            df_sorted[metric_name],
            "o-",
            linewidth=2,
            markersize=8,
        )

        ax.set_title(f"{param_name} 对 {metric_name} 的影响", fontsize=14, fontweight="bold")
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if save_path is None:
            save_path = str(self.output_dir / f"sensitivity_{param_name}_{metric_name}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"参数影响图已保存至: {save_path}")
        return save_path

    def plot_heatmap(
        self,
        param1: str,
        param2: str,
        metric_name: str = "sharpe_ratio",
        save_path: Optional[str] = None,
    ) -> str:
        """
        绘制两个参数的热力图。

        Args:
            param1: 参数1名称
            param2: 参数2名称
            metric_name: 指标名称
            save_path: 保存路径

        Returns:
            保存的文件路径
        """
        if not self.results:
            logger.warning("没有测试结果，无法绘制")
            return ""

        df = pd.DataFrame(self.results)

        if param1 not in df.columns or param2 not in df.columns or metric_name not in df.columns:
            logger.warning(f"参数或指标不存在")
            return ""

        pivot_df = df.pivot_table(values=metric_name, index=param1, columns=param2)

        fig, ax = plt.subplots(figsize=(10, 8))

        import seaborn as sns
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn",
            center=pivot_df.mean().mean(),
            ax=ax,
        )

        ax.set_title(f"{param1} vs {param2} - {metric_name}", fontsize=14, fontweight="bold")

        if save_path is None:
            save_path = str(self.output_dir / f"sensitivity_heatmap_{param1}_{param2}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"热力图已保存至: {save_path}")
        return save_path
