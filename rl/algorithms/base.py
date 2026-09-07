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
        # 设备检测必须同时检查 is_available 与 device_count：
        # 本环境 CUDA_VISIBLE_DEVICES="" 时 torch.cuda.is_available() 仍返回 True 但
        # device_count()=0（GPU 实际仍可被 .to('cuda') 使用），仅凭 is_available 会把
        # 设备误判为 cuda，导致 GPU 进程保存的 checkpoint 在 CPU 进程加载时崩溃
        # （RuntimeError: ... CUDA device 0 but torch.cuda.device_count() is 0）。
        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.device = torch.device("cuda" if cuda_ok else "cpu")

    @abstractmethod
    def predict(self, state: np.ndarray, deterministic: bool = False) -> int:
        """给定状态，返回动作

        Args:
            state: 状态向量，形状 (state_dim,)
            deterministic: 是否使用确定性策略（关闭探索）

        Returns:
            action: 动作索引 0=HOLD, 1=BUY1, 2=BUY2, 3=BUY3, 4=SELL1, 5=SELL2, 6=SELL3
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