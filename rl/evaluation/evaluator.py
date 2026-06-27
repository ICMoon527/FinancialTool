# -*- coding: utf-8 -*-
"""模型评估器：在验证集上运行模型，计算评估指标"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from rl.environment import T0Environment

if TYPE_CHECKING:
    from rl.config import RLConfig
    from rl.data.dataset import IntradayDataset, DaySample
    from rl.algorithms.base import AbstractRLModel

logger = logging.getLogger(__name__)


class RLEvaluator:
    """模型评估器"""

    def __init__(
        self,
        config: "RLConfig",
        model: "AbstractRLModel",
        dataset: "IntradayDataset",
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.env = T0Environment(config)

    def evaluate(self) -> Dict:
        """在全部验证集上评估模型

        Returns:
            {
                "cumulative_returns": List[float],
                "benchmark_returns": List[float],
                "daily_summaries": List[Dict],
                "summary_metrics": {
                    "sharpe_ratio": float,
                    "total_return": float,
                    "win_rate": float,
                    "max_drawdown": float,
                    "total_trades": int,
                },
            }
        """
        daily_returns = []
        daily_summaries = []
        benchmark_returns = []

        val_samples = self.dataset.val_samples
        if not val_samples:
            logger.warning("验证集为空，无法评估")
            return {
                "cumulative_returns": [],
                "benchmark_returns": [],
                "daily_summaries": [],
                "summary_metrics": {
                    "sharpe_ratio": 0.0,
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0,
                },
            }

        for sample in val_samples:
            day_return, day_summary, bench_return = self._evaluate_one_day(sample)
            daily_returns.append(day_return)
            daily_summaries.append(day_summary)
            benchmark_returns.append(bench_return)

        # 计算累积收益
        cumulative = np.cumprod(1 + np.array(daily_returns)) - 1
        benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns)) - 1

        # 计算总体指标
        summary = self._compute_summary_metrics(daily_returns, daily_summaries)

        return {
            "cumulative_returns": cumulative.tolist(),
            "benchmark_returns": benchmark_cumulative.tolist(),
            "daily_summaries": daily_summaries,
            "summary_metrics": summary,
        }

    def evaluate_daily(self, stock_code: str, target_date: date) -> Dict:
        """获取指定股票/日期的单日逐笔决策明细

        Returns:
            {
                "stock_code": str,
                "date": str,
                "klines": List[Dict],
                "decisions": List[Dict],
                "reward_heatmap": List[float],
                "trades": List[Dict],
            }
        """
        sample = self.dataset.find_sample(stock_code, target_date)
        if sample is None:
            raise ValueError(f"Sample not found: {stock_code} {target_date}")

        state = self.env.reset(
            self._sample_to_dict(sample),
            self._get_prev_klines(sample),
        )
        done = False
        decisions = []
        reward_heatmap = []
        klines = sample.klines if hasattr(sample, 'klines') else sample.get('klines', [])

        while not done:
            action = self.model.predict(state, deterministic=True)
            next_state, reward, done, info = self.env.step(action)
            decisions.append({
                "step": info.get("step", len(decisions)),
                "action": T0Environment.ACTION_NAMES[action],
                "reward": float(reward),
                "position": info.get("position", 0),
            })
            reward_heatmap.append(float(reward))
            state = next_state

        return {
            "stock_code": stock_code,
            "date": target_date.isoformat(),
            "klines": klines,
            "decisions": decisions,
            "reward_heatmap": reward_heatmap,
            "trades": self.env._trades,
        }

    def _evaluate_one_day(self, sample: "DaySample") -> Tuple[float, Dict, float]:
        """评估单个交易日

        Returns:
            (日收益率, 日摘要, 基准收益率)
        """
        state = self.env.reset(
            self._sample_to_dict(sample),
            self._get_prev_klines(sample),
        )
        done = False
        total_reward = 0.0
        trade_count = 0

        while not done:
            action = self.model.predict(state, deterministic=True)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            if info.get("action_applied", 0) != 0:
                trade_count += 1
            state = next_state

        # 日收益率（近似）
        day_return = total_reward / 100.0

        # 基准收益率（买入持有）
        klines = sample.klines if hasattr(sample, 'klines') else sample.get('klines', [])
        if len(klines) >= 2:
            open_price = klines[0].get("Close", 0)
            close_price = klines[-1].get("Close", 0)
            if open_price > 0:
                bench_return = (close_price - open_price) / open_price
            else:
                bench_return = 0.0
        else:
            bench_return = 0.0

        day_summary = {
            "date": sample.date.isoformat() if hasattr(sample, 'date') else sample.get("date", ""),
            "stock_code": sample.stock_code if hasattr(sample, 'stock_code') else sample.get("stock_code", ""),
            "daily_return": day_return,
            "trade_count": trade_count,
            "avg_reward": total_reward / max(len(klines), 1),
            "trades": self.env._trades,
        }

        return day_return, day_summary, bench_return

    def _compute_summary_metrics(
        self, daily_returns: List[float], daily_summaries: List[Dict]
    ) -> Dict:
        """计算总体评估指标"""
        returns = np.array(daily_returns)

        # Sharpe Ratio（年化，假设252个交易日）
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

        # 总收益
        total_return = float(np.prod(1 + returns) - 1)

        # 胜率
        win_count = sum(1 for r in returns if r > 0)
        win_rate = float(win_count / max(len(returns), 1))

        # 最大回撤
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # 总交易次数
        total_trades = sum(s["trade_count"] for s in daily_summaries)

        return {
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
        }

    @staticmethod
    def _sample_to_dict(sample) -> dict:
        """将 sample（dataclass 或 dict）统一转为 dict"""
        if isinstance(sample, dict):
            return {
                "klines": sample["klines"],
                "stock_code": sample["stock_code"],
                "date": sample["date"],
            }
        return {
            "klines": sample.klines,
            "stock_code": sample.stock_code,
            "date": sample.date,
        }

    @staticmethod
    def _get_prev_klines(sample):
        """获取前一日 K 线"""
        if isinstance(sample, dict):
            return sample.get("prev_day_klines")
        return sample.prev_day_klines