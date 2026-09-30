# -*- coding: utf-8 -*-
"""强化学习神经网络模块"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(
        self, x: torch.Tensor, window: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """返回 Q(s, a) 各动作的 Q 值

        Args:
            x: 状态张量 (batch, state_dim)
            window: 形态窗口张量 (batch, W, 5)；MLP 网络不使用，仅保持统一签名
        """
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

    def forward(
        self, x: torch.Tensor, window: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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


class CNNEncoderDQN(RLNetworkBase):
    """1D-CNN 形态编码器 + MLP 输出 Q 值（时间序列形态归纳）

    结构：
      - cnn：输入 (batch, C, window) → 三层空洞 Conv1d（保留序列长度）
      - 多尺度池化：全局平均 + 全局最大 + 末段时点 → 拼接
      - shape_fc：压缩为 cnn_out_dim 维形态向量
      - head：形态向量与 state 拼接后经 Dueling 式 MLP 输出 Q 值

    设计动机：时间序列拼接让 buffer 含「前日全天 + 当日」K线序列，但纯 MLP
    无法从长序列归纳「昨日尾盘走势/今日开盘方向/V型W型」等形态；CNN 卷积天然
    提取局部时间模式，弥补 MLP 的时序盲区。注意 state 维度不变，前日信息只经
    由这条时间序列进入模型。

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        window: 时间序列窗口长度（前日全天 + 当日全天，约 480 根）
        out_dim: 形态向量维度
        in_channels: 输入通道数（OHLCV + day_flag + intraday_pos + valid_mask = 8）
        hidden_channels: 各层 Conv1d 通道数
        kernel_size: Conv1d 卷积核大小
        dilation: 各层膨胀率（逐层扩大感受野）
        hidden_sizes: 拼接后 MLP 隐藏层节点数，默认 (256, 128)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        window: int = 480,
        out_dim: int = 16,
        in_channels: int = 8,
        hidden_channels: Tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 5,
        dilation: Tuple[int, ...] = (1, 2, 4),
        hidden_sizes: Tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        self.window = window
        self.in_channels = in_channels

        # 1D-CNN 形态编码器：三层空洞卷积逐层扩大感受野（约 60 根K线），
        # padding 按膨胀率补足以保持序列长度（不在此处做池化，保留时序给下游多尺度池化）
        cnn_layers = []
        in_ch = in_channels
        for i, ch in enumerate(hidden_channels):
            d = dilation[i] if i < len(dilation) else 1
            pad = d * (kernel_size - 1) // 2
            cnn_layers.append(
                nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=pad, dilation=d)
            )
            cnn_layers.append(nn.ReLU())
            in_ch = ch
        self.cnn = nn.Sequential(*cnn_layers)      # → (batch, last_ch, W)
        # 多尺度池化（平均 + 最大 + 末段）拼接后压缩为形态向量
        self.shape_fc = nn.Linear(hidden_channels[-1] * 3, out_dim)

        # 形态向量与状态拼接后走 Dueling 式 MLP
        shared = []
        in_dim = state_dim + out_dim
        for h in hidden_sizes:
            shared.append(nn.Linear(in_dim, h))
            shared.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*shared)
        self.value_stream = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))

        self._init_weights()

    def forward(
        self, x: torch.Tensor, window: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """返回 Q(s, a) 各动作的 Q 值

        Args:
            x: 状态张量 (batch, state_dim)
            window: 形态窗口张量 (batch, W, C)，时间序列拼接（前日 + 当日）归一化后；
                    为 None 时以全 0 窗口兜底（预测早期无历史数据场景）
        """
        if window is None:
            batch = x.shape[0]
            window = torch.zeros(
                batch, self.window, self.in_channels, device=x.device, dtype=x.dtype
            )
        # CNN: (batch, W, C) → permute → (batch, C, W) → (batch, last_ch, W)
        h = self.cnn(window.permute(0, 2, 1))
        # 多尺度池化：全局平均 + 全局最大 + 末段时点拼接。
        # 旧实现仅用 AdaptiveAvgPool1d(1) 做全局平均池化，输出对时间位置不变，
        # 「V 型」与「倒 V 型」得到完全相同的表达，结构上无法表征形态；
        # 加入最大池化（捕捉极值事件）与末段（最近时点状态）后保留时序信息。
        h_avg = h.mean(dim=2)
        h_max = h.max(dim=2).values
        h_last = h[:, :, -1]
        h = torch.cat([h_avg, h_max, h_last], dim=1)
        h = F.relu(self.shape_fc(h))               # (batch, out_dim)
        features = self.shared(torch.cat([x, h], dim=1))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


def create_dqn_network(config: "RLConfig") -> nn.Module:
    """根据配置创建 DQN 网络"""
    if config.use_cnn_encoder:
        return CNNEncoderDQN(
            config.state_dim,
            config.action_dim,
            window=config.cnn_window,
            out_dim=config.cnn_out_dim,
            in_channels=config.cnn_in_channels,
            hidden_channels=config.cnn_hidden_channels,
            kernel_size=config.cnn_kernel_size,
            dilation=config.cnn_dilation,
        )
    if config.dqn_dueling:
        return DuelingDQNNetwork(config.state_dim, config.action_dim)
    return DQNNetwork(config.state_dim, config.action_dim, config.dqn_hidden_sizes)


def create_ppo_networks(config: "RLConfig") -> Tuple[SharedBackbone, ActorNetwork, CriticNetwork]:
    """创建 PPO 的 Actor-Critic 网络组合"""
    backbone = SharedBackbone(config.state_dim)
    actor = ActorNetwork(backbone.output_dim, config.action_dim)
    critic = CriticNetwork(backbone.output_dim)
    return backbone, actor, critic