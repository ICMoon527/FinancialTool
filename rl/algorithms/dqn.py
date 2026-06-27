# -*- coding: utf-8 -*-
"""DQN 算法实现（含 Double DQN 和 Dueling DQN 变体）

核心组件：
- q_network: 在线 Q 网络
- target_network: 目标 Q 网络（延迟更新）
- replay_buffer: 经验回放缓冲区
- epsilon: 探索率（指数衰减）
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.algorithms.base import AbstractRLModel
from rl.networks import create_dqn_network

if TYPE_CHECKING:
    from rl.config import RLConfig


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: Deque = deque(maxlen=capacity)
        self.state_dim = state_dim

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: float) -> None:
        """存入一条经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止 (1.0 或 0.0)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """随机采样 batch_size 条经验

        Returns:
            states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(dones)).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNModel(AbstractRLModel):
    """DQN 算法实现（含 Double DQN 和 Dueling DQN 变体）"""

    def __init__(self, config: "RLConfig"):
        super().__init__(config)
        self.q_network = create_dqn_network(config).to(self.device)
        self.target_network = create_dqn_network(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(
            config.replay_buffer_size, config.state_dim, self.device
        )
        self.epsilon = config.epsilon_start
        self._train_step_count = 0
        self._loss_fn = nn.MSELoss()

    def predict(self, state: np.ndarray, deterministic: bool = False) -> int:
        """ε-greedy 策略

        Args:
            state: 状态向量，形状 (state_dim,)
            deterministic: 是否使用确定性策略

        Returns:
            action: 动作索引
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """从 ReplayBuffer 采样并训练一步

        如果 ReplayBuffer 不足 batch_size，返回空指标。

        Returns:
            metrics: {"loss": float, "td_error": float}
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {"loss": 0.0, "td_error": 0.0}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        # 当前 Q 值
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标 Q 值
        with torch.no_grad():
            if self.config.dqn_double:
                # Double DQN: 用 q_network 选动作，target_network 评估
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = self._loss_fn(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Epsilon 衰减
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        # 目标网络更新
        self._train_step_count += 1
        if self._train_step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "td_error": float((target_q - q_value).abs().mean().item()),
        }

    def save(self, path: str) -> None:
        """保存模型到指定路径"""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self._train_step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """从指定路径加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_start)
        self._train_step_count = checkpoint.get("train_step", 0)

    def get_networks(self) -> Dict[str, nn.Module]:
        """返回所有网络模块"""
        return {"q_network": self.q_network, "target_network": self.target_network}