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

import csv
import json
import logging
import time
import uuid
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

    def load_dict(self, data: Dict[str, List[float]]) -> None:
        """从字典恢复指标历史（断点续训用）"""
        self.episode_rewards = list(data.get("episode_rewards", []))
        self.episode_lengths = list(data.get("episode_lengths", []))
        self.losses = list(data.get("losses", []))
        self.td_errors = list(data.get("td_errors", []))
        self.epsilons = list(data.get("epsilons", []))
        self.entropies = list(data.get("entropies", []))
        self.val_sharpe_ratios = list(data.get("val_sharpe_ratios", []))
        self.val_returns = list(data.get("val_returns", []))
        self.val_win_rates = list(data.get("val_win_rates", []))


class RLTrainer:
    """RL 训练编排器"""

    def __init__(
        self,
        config: "RLConfig",
        model: "AbstractRLModel",
        dataset: "IntradayDataset",
        callbacks: Optional[List["TrainingCallback"]] = None,
        save_freq: int = 50,
        log_dir: Optional[str] = None,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.env = T0Environment(config)
        self.metrics = TrainingMetrics()
        self.callbacks = callbacks or []
        self.save_freq = save_freq  # latest checkpoint 保存频率（episode）

        self._best_val_sharpe = -float("inf")
        self._patience_counter = 0
        self._model_dir = Path(config.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # CSV 逐集日志（崩溃安全：每集立即落盘）
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        log_root = Path(log_dir) if log_dir else self._model_dir / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_root / f"train_log_{self._run_id}.csv"
        self._csv_initialized = False

    # ── 断点续训 ──

    def resume(self, checkpoint_dir: str) -> int:
        """从 checkpoint 目录恢复训练状态

        恢复内容：模型权重、优化器、epsilon、经验回放缓冲区、
        指标历史、最佳验证 Sharpe、早停计数器、episode 进度、run_id（延续同一 CSV 日志）。

        Args:
            checkpoint_dir: checkpoint 目录（需含 model.pt / trainer_state.json / metrics.json）

        Returns:
            start_episode: 下一个待训练的 episode 序号
        """
        ckpt_dir = Path(checkpoint_dir)
        model_path = ckpt_dir / "model.pt"
        state_path = ckpt_dir / "trainer_state.json"
        metrics_path = ckpt_dir / "metrics.json"

        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 1. 恢复模型（权重/优化器/epsilon/replay buffer）
        self.model.load(str(model_path))

        # 2. 恢复指标历史
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics.load_dict(json.load(f))

        # 3. 恢复训练状态（episode 进度/最佳 sharpe/早停计数/run_id）
        start_episode = len(self.metrics.episode_rewards)
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            start_episode = state.get("next_episode", start_episode)
            self._best_val_sharpe = state.get("best_val_sharpe", -float("inf"))
            self._patience_counter = state.get("patience_counter", 0)
            resumed_run_id = state.get("run_id")
            if resumed_run_id:
                self._run_id = resumed_run_id
                self._csv_path = self._csv_path.parent / f"train_log_{self._run_id}.csv"
                self._csv_initialized = True  # 续写已有 CSV

        logger.info(
            f"断点续训: 从 {ckpt_dir.name} 恢复, start_episode={start_episode}, "
            f"best_sharpe={self._best_val_sharpe:.4f}, "
            f"replay_buffer={len(getattr(self.model, 'replay_buffer', []))} 条"
        )
        return start_episode

    def train(self, start_episode: int = 0) -> TrainingMetrics:
        """执行完整训练流程

        Args:
            start_episode: 起始 episode（断点续训时 > 0）

        Returns:
            metrics: 训练指标记录器
        """
        logger.info(f"开始训练: episode {start_episode} -> {self.config.training_episodes}, "
                     f"算法: {self.config.default_algorithm}, "
                     f"设备: {self.model.device}")

        t_start = time.time()
        for episode in range(start_episode, self.config.training_episodes):
            ep_start = time.time()

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
            val_metrics = None
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
                    self._save_checkpoint("best", next_episode=episode + 1)
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at episode {episode + 1}")

            # 5. 逐集 CSV 日志（立即落盘，崩溃安全）
            self._log_episode_csv(episode, episode_reward, episode_length, val_metrics, ep_start)

            # 6. 触发回调
            for cb in self.callbacks:
                cb.on_episode_end(episode, episode_reward, self.metrics)

            # 7. 定期保存 latest checkpoint（断点续训用，固定目录覆盖写入）
            if self.save_freq > 0 and (episode + 1) % self.save_freq == 0:
                self._save_checkpoint("latest", next_episode=episode + 1)
                elapsed = time.time() - t_start
                logger.info(
                    f"进度: {episode + 1}/{self.config.training_episodes}, "
                    f"已耗时 {elapsed / 60:.1f} 分钟"
                )

            # Early stopping 跳出（放在日志/保存之后）
            if (
                val_metrics is not None
                and self._patience_counter >= self.config.early_stopping_patience
            ):
                break

        # 训练结束：仅更新 latest（固定目录覆盖写入）。
        # 不保存 final 时间戳快照，避免频繁训练/续训产生越来越多的时间戳文件夹；
        # 每个模型只保留 best（最优）与 latest（最近，断点续训用）两个固定目录
        self._save_checkpoint("latest", next_episode=self.config.training_episodes)
        logger.info(f"训练完成, 总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
        return self.metrics

    def _log_episode_csv(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        val_metrics: Optional[Dict[str, float]],
        ep_start: float,
    ) -> None:
        """追加一行 episode 级日志到 CSV（首次写入表头）"""
        write_header = not self._csv_initialized and not self._csv_path.exists()
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "run_id", "episode", "reward", "length", "loss", "td_error",
                    "epsilon", "val_sharpe", "val_return", "val_win_rate", "seconds",
                ])
            self._csv_initialized = True
            writer.writerow([
                self._run_id,
                episode + 1,
                f"{episode_reward:.4f}",
                episode_length,
                f"{self.metrics.losses[-1]:.6f}" if self.metrics.losses else "",
                f"{self.metrics.td_errors[-1]:.6f}" if self.metrics.td_errors else "",
                f"{self.metrics.epsilons[-1]:.6f}" if self.metrics.epsilons else "",
                f"{val_metrics['sharpe']:.4f}" if val_metrics else "",
                f"{val_metrics['total_return']:.4f}" if val_metrics else "",
                f"{val_metrics['win_rate']:.4f}" if val_metrics else "",
                f"{time.time() - ep_start:.2f}",
            ])

    def _load_sample_data(self, sample):
        """加载样本的当日K线和前一日K线（惰性加载，兼容 MockDataset）"""
        if hasattr(self.dataset, "get_klines"):
            # IntradayDataset：按需查询数据库
            klines = self.dataset.get_klines(sample)
            prev_klines = self.dataset.get_prev_klines(sample)
        else:
            # MockDataset 等自带数据的数据集
            if isinstance(sample, dict):
                klines = sample["klines"]
                prev_klines = sample.get("prev_day_klines")
            else:
                klines = sample.klines
                prev_klines = sample.prev_day_klines
        return klines, prev_klines

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
        klines, prev_klines = self._load_sample_data(sample)
        state = self.env.reset(
            self._sample_to_dict(sample, klines),
            prev_klines,
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

    def _validate(self, max_val_days: int = 50) -> Dict[str, float]:
        """在验证集上评估模型，返回 Sharpe、总收益、胜率等

        Args:
            max_val_days: 验证集抽样上限（验证集可能非常大，全量验证过慢）
        """
        daily_returns = []
        daily_summaries = []

        # 验证集抽样（数据量大时避免全量验证拖慢训练）
        val_samples = self.dataset.val_samples
        if len(val_samples) > max_val_days:
            import random as _random
            val_samples = _random.sample(val_samples, max_val_days)
            logger.info(f"验证集抽样: {max_val_days}/{len(self.dataset.val_samples)}")

        for sample in val_samples:
            klines, prev_klines = self._load_sample_data(sample)
            state = self.env.reset(
                self._sample_to_dict(sample, klines),
                prev_klines,
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

    def _save_checkpoint(self, tag: str, next_episode: Optional[int] = None) -> str:
        """保存模型 checkpoint（固定目录覆盖写入，不产生时间戳版本）

        Args:
            tag: 标签（"best" 记录最优，"latest" 记录最近状态用于断点续训）
            next_episode: 下一个待训练的 episode 序号（断点续训用）

        Returns:
            model_path: 模型存储路径
        """
        # 固定目录，覆盖写入：latest 始终保留最近可恢复状态，
        # best 始终只保留最近一次验证 Sharpe 创新高的模型
        model_path = self._model_dir / f"{self.config.model_tag}_{tag}"
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

        # 保存训练状态（断点续训用）
        if next_episode is not None:
            trainer_state = {
                "next_episode": next_episode,
                "best_val_sharpe": self._best_val_sharpe,
                "patience_counter": self._patience_counter,
                "run_id": self._run_id,
                "saved_at": datetime.now().isoformat(),
            }
            with open(model_path / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2)

        logger.info(f"Checkpoint 已保存: {model_path}")
        return str(model_path)

    @staticmethod
    def _sample_to_dict(sample, klines=None) -> dict:
        """将 sample（dataclass 或 dict）统一转为 dict，兼容 MockDataset

        Args:
            sample: DaySample 或 dict
            klines: 已加载的当日K线（惰性加载时由调用方传入）
        """
        if isinstance(sample, dict):
            return {
                "klines": sample["klines"],
                "stock_code": sample["stock_code"],
                "date": sample["date"],
            }
        return {
            "klines": klines if klines is not None else sample.klines,
            "stock_code": sample.stock_code,
            "date": sample.date,
        }