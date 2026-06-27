# -*- coding: utf-8 -*-
"""强化学习算法抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from rl.config import RLConfig


class AbstractRLModel(ABC):
    """RL 算法抽象基类"""

    def __init__(self, config: "RLConfig"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def predict(self, state: np.ndarray, deterministic: bool = False) -> int:
        """给定状态，返回动作

        Args:
            state: 状态向量，形状 (state_dim,)
            deterministic: 是否使用确定性策略（关闭探索）

        Returns:
            action: 动作索引 0=HOLD, 1=BUY, 2=SELL
        """
        ...

    @abstractmethod
    def train_step(self, batch: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """执行一步训练

        Args:
            batch: 训练数据批次（可选，DQN 从 ReplayBuffer 内部采样）

        Returns:
            metrics: 包含 loss, td_error 等指标的字典
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型到指定路径"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从指定路径加载模型"""
        ...

    @abstractmethod
    def get_networks(self) -> Dict[str, torch.nn.Module]:
        """返回所有网络模块，用于序列化"""
        ...