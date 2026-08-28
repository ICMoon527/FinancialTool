# -*- coding: utf-8 -*-
"""训练回调模块"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rl.training.trainer import TrainingMetrics, RLTrainer


class TrainingCallback(ABC):
    """训练回调基类"""

    @abstractmethod
    def on_episode_end(
        self, episode: int, episode_reward: float, metrics: "TrainingMetrics"
    ) -> None:
        """每个 episode 结束时调用

        Args:
            episode: 当前 episode 索引（从 0 开始）
            episode_reward: 当前 episode 的总奖励
            metrics: 训练指标记录器
        """
        ...


class ProgressCallback(TrainingCallback):
    """进度回调：记录到日志 + 更新进度存储（供前端轮询）"""

    def __init__(self, progress_store: Dict[str, object]):
        self._store = progress_store
        self._log_interval = 10

    def on_episode_end(self, episode: int, episode_reward: float, metrics) -> None:
        self._store["current_episode"] = episode
        self._store["latest_reward"] = episode_reward
        self._store["metrics"] = metrics.to_dict()

        if (episode + 1) % self._log_interval == 0:
            avg_reward = 0.0
            if metrics.episode_rewards:
                recent = metrics.episode_rewards[-self._log_interval:]
                avg_reward = sum(recent) / len(recent) if recent else 0.0
            epsilon_str = (
                f"{metrics.epsilons[-1]:.4f}" if metrics.epsilons else "N/A"
            )
            logger.info(
                f"Episode {episode + 1}: avg_reward={avg_reward:.4f}, "
                f"epsilon={epsilon_str}"
            )


class CheckpointCallback(TrainingCallback):
    """定期保存 checkpoint"""

    def __init__(self, save_interval: int = 100, trainer: "RLTrainer" = None):
        self.save_interval = save_interval
        self._trainer = trainer

    def on_episode_end(self, episode: int, episode_reward: float, metrics) -> None:
        if (episode + 1) % self.save_interval == 0 and self._trainer is not None:
            self._trainer._save_checkpoint(f"ep{episode + 1}")
            logger.info(f"已保存 checkpoint: episode {episode + 1}")


class TrainingInterrupted(Exception):
    """训练被外部中断（暂停后停止），携带进度信息供断点续训"""

    def __init__(self, next_episode: int, reason: str = "stopped"):
        self.next_episode = next_episode
        self.reason = reason
        super().__init__(f"训练中断于 episode {next_episode}: {reason}")


class TrainingControlCallback(TrainingCallback):
    """训练控制回调：响应暂停/停止事件（由服务层注入 threading.Event）

    - stop_event 置位：抛出 TrainingInterrupted，训练线程可保存断点后退出
    - pause_event 置位：阻塞等待（每 0.5s 轮询），期间响应停止
    """

    def __init__(self, pause_event, stop_event):
        import threading
        self._pause_event: threading.Event = pause_event
        self._stop_event: threading.Event = stop_event

    def on_episode_end(self, episode: int, episode_reward: float, metrics) -> None:
        # 停止：立即中断（next_episode 指向下一集，续训从该集继续）
        if self._stop_event.is_set():
            raise TrainingInterrupted(episode + 1, "stopped")
        # 暂停：阻塞等待恢复
        while self._pause_event.is_set():
            if self._stop_event.is_set():
                raise TrainingInterrupted(episode + 1, "stopped")
            import time
            time.sleep(0.5)