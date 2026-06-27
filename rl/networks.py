# -*- coding: utf-8 -*-
"""强化学习神经网络模块"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from rl.config import RLConfig


class RLNetworkBase(nn.Module):
    """网络基类：统一 Xavier 权重初始化"""

    def __init__(self):
        super().__init__()

    def _init_weights(self):
        """Xavier 初始化所有 Linear 层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class DQNNetwork(RLNetworkBase):
    """
    DQN 网络：Input(state_dim) → Dense(256) → ReLU → Dense(128) → ReLU
             → Dense(64) → ReLU → Dense(action_dim)

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_sizes: 隐藏层节点数列表，默认 (256, 128, 64)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 Q(s, a) 各动作的 Q 值"""
        return self.net(x)


class DuelingDQNNetwork(RLNetworkBase):
    """
    Dueling DQN：共享层 → 分叉为 V(s) 和 A(s,a)，最后合并 Q = V + (A - mean(A))

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_sizes: 共享层节点数，默认 (256, 128)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 128)):
        super().__init__()
        # 共享特征提取层
        shared = []
        in_dim = state_dim
        for h in hidden_sizes:
            shared.append(nn.Linear(in_dim, h))
            shared.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*shared)

        # 价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 优势流 A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class SharedBackbone(RLNetworkBase):
    """
    PPO 共享特征提取层：Input(state_dim) → Dense(256) → ReLU → Dense(128) → ReLU

    Args:
        state_dim: 状态维度
        hidden_sizes: 隐藏层节点数，默认 (256, 128)
    """

    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, ...] = (256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNetwork(RLNetworkBase):
    """PPO Actor：backbone输出 → Dense(64) → ReLU → Dense(action_dim) → Softmax"""

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticNetwork(RLNetworkBase):
    """PPO Critic：backbone输出 → Dense(64) → ReLU → Dense(1)"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_dqn_network(config: "RLConfig") -> nn.Module:
    """根据配置创建 DQN 网络"""
    if config.dqn_dueling:
        return DuelingDQNNetwork(config.state_dim, config.action_dim)
    return DQNNetwork(config.state_dim, config.action_dim, config.dqn_hidden_sizes)


def create_ppo_networks(config: "RLConfig") -> Tuple[SharedBackbone, ActorNetwork, CriticNetwork]:
    """创建 PPO 的 Actor-Critic 网络组合"""
    backbone = SharedBackbone(config.state_dim)
    actor = ActorNetwork(backbone.output_dim, config.action_dim)
    critic = CriticNetwork(backbone.output_dim)
    return backbone, actor, critic