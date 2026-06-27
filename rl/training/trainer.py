# -*- coding: utf-8 -*-
"""RL 训练编排器

职责：
1. 管理训练循环（episode 迭代）
2. 协调环境、算法、数据集之间的交互
3. 记录训练指标
4. 定期验证 + Early Stopping
5. 保存模型 checkpoint
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from rl.environment import T0Environment
from rl.algorithms.dqn import DQNModel

if TYPE_CHECKING:
    from rl.config import RLConfig
    from rl.data.dataset import IntradayDataset, DaySample
    from rl.algorithms.base import AbstractRLModel
    from rl.training.callbacks import TrainingCallback

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """训练指标记录器"""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.td_errors: List[float] = []
        self.epsilons: List[float] = []
        self.entropies: List[float] = []
        self.val_sharpe_ratios: List[float] = []
        self.val_returns: List[float] = []
        self.val_win_rates: List[float] = []

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "td_errors": self.td_errors,
            "epsilons": self.epsilons,
            "entropies": self.entropies,
            "val_sharpe_ratios": self.val_sharpe_ratios,
            "val_returns": self.val_returns,
            "val_win_rates": self.val_win_rates,
        }


class RLTrainer:
    """RL 训练编排器"""

    def __init__(
        self,
        config: "RLConfig",
        model: "AbstractRLModel",
        dataset: "IntradayDataset",
        callbacks: Optional[List["TrainingCallback"]] = None,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.env = T0Environment(config)
        self.metrics = TrainingMetrics()
        self.callbacks = callbacks or []

        self._best_val_sharpe = -float("inf")
        self._patience_counter = 0
        self._model_dir = Path(config.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> TrainingMetrics:
        """执行完整训练流程

        Returns:
            metrics: 训练指标记录器
        """
        logger.info(f"开始训练: {self.config.training_episodes} episodes, "
                     f"算法: {self.config.default_algorithm}, "
                     f"设备: {self.model.device}")

        for episode in range(self.config.training_episodes):
            # 1. 采样一个训练样本
            sample = self.dataset.sample_train()

            # 2. 运行一个 episode
            episode_reward, episode_length, episode_metrics = self._run_episode(
                sample, training=True
            )

            # 3. 记录指标
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_lengths.append(episode_length)
            if episode_metrics:
                self.metrics.losses.append(episode_metrics.get("loss", 0))
                self.metrics.td_errors.append(episode_metrics.get("td_error", 0))
                if isinstance(self.model, DQNModel):
                    self.metrics.epsilons.append(self.model.epsilon)

            # 4. 定期验证
            if self.dataset.val_samples and (episode + 1) % self.config.validation_freq == 0:
                val_metrics = self._validate()
                self.metrics.val_sharpe_ratios.append(val_metrics["sharpe"])
                self.metrics.val_returns.append(val_metrics["total_return"])
                self.metrics.val_win_rates.append(val_metrics["win_rate"])

                logger.info(
                    f"验证 E{episode + 1}: sharpe={val_metrics['sharpe']:.4f}, "
                    f"return={val_metrics['total_return']:.4f}, "
                    f"win_rate={val_metrics['win_rate']:.2%}"
                )

                # Early Stopping
                if val_metrics["sharpe"] > self._best_val_sharpe:
                    self._best_val_sharpe = val_metrics["sharpe"]
                    self._patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at episode {episode + 1}")
                        break

            # 5. 触发回调
            for cb in self.callbacks:
                cb.on_episode_end(episode, episode_reward, self.metrics)

        # 最终保存
        self._save_checkpoint("final")
        logger.info("训练完成")
        return self.metrics

    def _run_episode(
        self, sample: "DaySample", training: bool
    ) -> Tuple[float, int, Optional[Dict]]:
        """运行一个 episode

        Args:
            sample: 交易日样本
            training: 是否为训练模式

        Returns:
            (总奖励, 步数, 训练指标)
        """
        state = self.env.reset(
            self._sample_to_dict(sample),
            self._get_prev_klines(sample),
        )
        done = False
        total_reward = 0.0
        episode_length = 0
        train_metrics = None

        while not done:
            action = self.model.predict(state, deterministic=not training)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            episode_length += 1

            if training and not info.get("is_warmup", False):
                # 存储经验到 Replay Buffer（DQN）
                if isinstance(self.model, DQNModel):
                    self.model.replay_buffer.push(
                        state, action, reward, next_state, float(done)
                    )
                    # 每步执行一次训练
                    train_metrics = self.model.train_step()

            state = next_state

        return total_reward, episode_length, train_metrics

    def _validate(self) -> Dict[str, float]:
        """在验证集上评估模型，返回 Sharpe、总收益、胜率等"""
        daily_returns = []
        daily_summaries = []

        for sample in self.dataset.val_samples:
            state = self.env.reset(
                self._sample_to_dict(sample),
                self._get_prev_klines(sample),
            )
            done = False
            day_reward = 0.0
            trade_count = 0

            while not done:
                action = self.model.predict(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                day_reward += reward
                if info.get("action_applied", 0) != 0:
                    trade_count += 1
                state = next_state

            # 将 episode reward 近似为日收益率
            daily_returns.append(day_reward / 100.0)  # 归一化
            daily_summaries.append({
                "trade_count": trade_count,
                "reward": day_reward,
            })

        returns = np.array(daily_returns)
        # Sharpe Ratio（年化，假设252个交易日）
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))
        # 总收益
        total_return = float(np.sum(returns))
        # 胜率
        win_count = sum(1 for r in returns if r > 0)
        win_rate = float(win_count / max(len(returns), 1))

        return {
            "sharpe": sharpe,
            "total_return": total_return,
            "win_rate": win_rate,
        }

    def _save_checkpoint(self, tag: str) -> str:
        """保存模型 checkpoint

        Args:
            tag: 标签（如 "best", "final", "ep100"）

        Returns:
            model_path: 模型存储路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.default_algorithm}_{tag}_{timestamp}"
        model_path = self._model_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save(str(model_path / "model.pt"))

        # 保存配置
        config_dict = {}
        for field_name in self.config.__dataclass_fields__:
            config_dict[field_name] = str(getattr(self.config, field_name))
        with open(model_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # 保存指标
        with open(model_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        logger.info(f"Checkpoint 已保存: {model_path}")
        return str(model_path)

    @staticmethod
    def _sample_to_dict(sample) -> dict:
        """将 sample（dataclass 或 dict）统一转为 dict，兼容 MockDataset"""
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