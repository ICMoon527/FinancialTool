# -*- coding: utf-8 -*-
"""DQN 算法实现（含 Double DQN、Dueling DQN 与 Prioritized Experience Replay）

核心组件：
- q_network: 在线 Q 网络
- target_network: 目标 Q 网络（延迟更新）
- replay_buffer: 优先级经验回放（PER，proportional prioritization + IS 权重）
- epsilon: 探索率（指数衰减）
"""

from __future__ import annotations

import dataclasses
import logging
import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.algorithms.base import AbstractRLModel
from rl.networks import create_dqn_network

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rl.config import RLConfig


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区（PER，proportional prioritization）

    采样概率 P(i) ∝ (|δ_i| + ε)^α，其中 δ_i 为 TD 误差；
    并用重要性采样（IS）权重 w_i = (N·P_i)^(-β) 修正采样偏差，
    β 随训练步数从 beta_start 线性退火到 beta_end（保证收敛时无偏）。

    实现为定长 numpy 环形数组；新经验以当前最大优先级入队（保证被采到）。
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        eps: float = 1e-6,
        beta_anneal_steps: int = 200_000,
        window_len: int = 0,
        window_channels: int = 5,
    ):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.window_len = window_len  # 形态窗口长度（0 = 不使用 CNN 编码器）
        self.window_channels = window_channels  # 形态窗口通道数（OHLCV + 会话标记）

        # PER 超参数
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.eps = eps
        self._beta_anneal_steps = beta_anneal_steps

        # 存储数组
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._priorities = np.zeros(capacity, dtype=np.float32)
        # 形态窗口数组（CNN 编码器输入；window_len=0 时不分配）
        if window_len > 0:
            self._windows = np.zeros((capacity, window_len, window_channels), dtype=np.float32)
            self._next_windows = np.zeros((capacity, window_len, window_channels), dtype=np.float32)
        else:
            self._windows = None
            self._next_windows = None

        # 环形指针与状态
        self._ptr = 0
        self._size = 0
        self._max_priority = 1.0
        self._beta = beta_start
        self._anneal_count = 0

    # ── 写入 ──

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        window: Optional[np.ndarray] = None,
        next_window: Optional[np.ndarray] = None,
    ) -> None:
        """存入一条经验（新经验赋予当前最大优先级，保证至少被采样一次）

        NaN/Inf 防护：状态向量或奖励非有限时丢弃该经验，避免脏数据进入
        回放缓冲区后经 Q 网络传播为 NaN 梯度（数值发散时的第一道防线）。
        """
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if (
            not np.isfinite(state).all()
            or not np.isfinite(next_state).all()
            or not np.isfinite(reward)
        ):
            return
        idx = self._ptr
        self._states[idx] = state
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self._dones[idx] = float(done)
        self._priorities[idx] = self._max_priority
        if self._windows is not None:
            self._windows[idx] = (
                np.asarray(window, dtype=np.float32) if window is not None else 0.0
            )
            self._next_windows[idx] = (
                np.asarray(next_window, dtype=np.float32) if next_window is not None else 0.0
            )
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ── 采样 ──

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """按优先级概率采样 batch_size 条经验

        Returns:
            states, actions, rewards, next_states, dones, indices, is_weights
        """
        # 优先级概率分布 P(i) ∝ p_i^α
        priorities = self._priorities[: self._size] ** self.alpha
        probs = priorities / (priorities.sum() + 1e-8)
        indices = np.random.choice(self._size, batch_size, p=probs)

        # 重要性采样权重（修正优先级采样偏差），归一化到最大值 1
        total = self._size
        weights = (total * probs[indices]) ** (-self._beta)
        weights = weights / (weights.max() + 1e-8)

        # β 线性退火（每采样一次推进一步）
        self._anneal_count = min(self._anneal_count + 1, self._beta_anneal_steps)
        self._beta = self.beta_start + (self.beta_end - self.beta_start) * (
            self._anneal_count / self._beta_anneal_steps
        )

        return (
            torch.FloatTensor(self._states[indices]).to(self.device),
            torch.LongTensor(self._actions[indices]).to(self.device),
            torch.FloatTensor(self._rewards[indices]).to(self.device),
            torch.FloatTensor(self._next_states[indices]).to(self.device),
            torch.FloatTensor(self._dones[indices]).to(self.device),
            indices,
            torch.FloatTensor(weights).to(self.device),
            # 形态窗口（CNN 编码器输入；未启用时返回 None）
            (torch.FloatTensor(self._windows[indices]).to(self.device) if self._windows is not None else None),
            (torch.FloatTensor(self._next_windows[indices]).to(self.device) if self._next_windows is not None else None),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """用 TD 误差更新被采样经验的优先级

        NaN/Inf 防护：TD 误差发散为 NaN/±Inf 时先归零（nan_to_num），
        避免 NaN 优先级污染采样分布 p（numpy 对 NaN 概率的采样是未定义行为，
        曾导致 np.random.choice 触发原生段错误 0xC0000005）。
        """
        safe_td = np.nan_to_num(
            np.asarray(td_errors, dtype=np.float32),
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        )
        for idx, td in zip(indices, safe_td):
            # 优先级上限：TD 误差发散时单个经验权重过大，会垄断 PER 采样分布、
            # 反复放大同一批高误差样本进一步加剧发散，故封顶为 20
            # （正常 TD 误差 <5，cap=20 即 4 倍上限，兼顾正常采样与发散抑制）
            priority = min(float(abs(td)) + self.eps, 20.0)
            self._priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    # ── 序列化（断点续训）──

    def serialize(self) -> Dict:
        """导出缓冲区内容（numpy 数组）"""
        data = {
            "states": self._states[: self._size],
            "actions": self._actions[: self._size],
            "rewards": self._rewards[: self._size],
            "next_states": self._next_states[: self._size],
            "dones": self._dones[: self._size],
            "priorities": self._priorities[: self._size],
            "size": self._size,
            "max_priority": self._max_priority,
            "beta": self._beta,
            "anneal_count": self._anneal_count,
        }
        if self._windows is not None:
            data["windows"] = self._windows[: self._size]
            data["next_windows"] = self._next_windows[: self._size]
        return data

    def restore(self, data: Dict) -> None:
        """从序列化字典恢复缓冲区"""
        n = int(data["size"])
        self._states[:n] = data["states"]
        self._actions[:n] = data["actions"]
        self._rewards[:n] = data["rewards"]
        self._next_states[:n] = data["next_states"]
        self._dones[:n] = data["dones"]
        self._priorities[:n] = data["priorities"]
        if self._windows is not None:
            if "windows" in data:
                self._windows[:n] = data["windows"]
                self._next_windows[:n] = data["next_windows"]
            else:
                # 兼容旧版无窗口 checkpoint：历史经验窗口置 0（旧策略行为已变化，
                # 但保留可继续训练，新经验将携带正确窗口）
                self._windows[:n] = 0.0
                self._next_windows[:n] = 0.0
        self._ptr = n % self.capacity
        self._size = n
        self._max_priority = float(data.get("max_priority", 1.0))
        self._beta = float(data.get("beta", self.beta_start))
        self._anneal_count = int(data.get("anneal_count", 0))

    def __len__(self) -> int:
        return self._size


class DQNModel(AbstractRLModel):
    """DQN 算法实现（Double DQN + Dueling DQN + Prioritized Experience Replay）"""

    # Q 值裁剪边界（%刻度）：reward_clip=5、γ=0.9 时收敛上限 ≈ 5/(1-γ) = 50。
    # 用于 target_q 与预测 Q 值的有界化，阻断 bootstrap 正反馈发散
    Q_VALUE_CLIP: float = 50.0

    def __init__(self, config: "RLConfig"):
        super().__init__(config)
        self._build()

    def _build(self) -> None:
        """按 self.config 构建网络/优化器/回放缓冲区

        独立成方法以便 load 时按 checkpoint 反推的 state_dim 重建（见 load）。
        """
        self.q_network = create_dqn_network(self.config).to(self.device)
        self.target_network = create_dqn_network(self.config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.replay_buffer_size,
            self.config.state_dim,
            self.device,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_end=self.config.per_beta_end,
            eps=self.config.per_eps,
            window_len=self.config.cnn_window if self.config.use_cnn_encoder else 0,
            window_channels=self.config.cnn_in_channels,
        )
        self.epsilon = self.config.epsilon_start
        self._train_step_count = 0
        self._loss_fn = nn.MSELoss(reduction="none")

    def predict(
        self,
        state: np.ndarray,
        window: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """ε-greedy 策略

        Args:
            state: 状态向量，形状 (state_dim,)
            window: 形态窗口（时间序列拼接，形状 (W, C)）；
                    仅 use_cnn_encoder 时使用
            deterministic: 是否使用确定性策略

        Returns:
            action: 动作索引
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            window_tensor = None
            if window is not None:
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor, window_tensor)
            if not torch.isfinite(q_values).all():
                # 兜底：Q 值含 NaN/Inf（权重被污染）时返回 HOLD，
                # 避免 argmax 在 NaN 上返回任意动作进一步污染经验
                return self.HOLD
            return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """从 PER 缓冲区采样并训练一步

        如果缓冲区不足 batch_size，返回空指标。

        Returns:
            metrics: {"loss": float, "td_error": float}
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {"loss": 0.0, "td_error": 0.0}

        states, actions, rewards, next_states, dones, indices, is_weights, windows, next_windows = (
            self.replay_buffer.sample(self.config.batch_size)
        )

        # 当前 Q 值
        q_values = self.q_network(states, windows)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标 Q 值（Double DQN + 裁剪）
        with torch.no_grad():
            if self.config.dqn_double:
                # Double DQN: 用 q_network 选动作，target_network 评估
                next_actions = self.q_network(next_states, next_windows).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states, next_windows).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states, next_windows).max(dim=1)[0]
            # Q 值裁剪（防发散核心）：限制 target 输出值域 [-Q_VALUE_CLIP, Q_VALUE_CLIP]。
            # 此前未裁剪：Q 值经 bootstrap 正反馈（target → TD误差 → 权重 → Q）指数膨胀，
            # 实测涨到 +4546（%刻度）而真实回报为负，agent 误以为交易能赚几百点。
            # 理论推导：reward_clip=5、γ=0.9 时 Q 值收敛上限 ≈ 5/(1-0.9) = 50，取 50 为界。
            next_q_values = torch.clamp(next_q_values, -self.Q_VALUE_CLIP, self.Q_VALUE_CLIP)
            target_q = torch.clamp(
                rewards + self.config.gamma * next_q_values * (1 - dones),
                -self.Q_VALUE_CLIP,
                self.Q_VALUE_CLIP,
            )

        # TD 误差（用于更新 PER 优先级）
        td_errors = (target_q - q_value).detach()

        # PER 加权损失：L = (1/N) Σ w_i · (y_i - Q(s,a))²
        loss = (is_weights * self._loss_fn(q_value, target_q)).mean()

        # NaN/Inf 防护（最后防线）：loss 发散为 NaN/±Inf 时跳过本次更新。
        # 若不拦截，NaN 梯度会经 optimizer.step() 把权重不可逆污染为 NaN，
        # 之后 Q 值全 NaN，np.random.choice 采样 NaN 概率分布直接触发
        # 原生段错误 0xC0000005（与 update_priorities 的 nan_to_num 双保险）。
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return {"loss": 0.0, "td_error": 0.0}

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新被采样经验的优先级
        self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        # 目标网络更新：优先 Polyak 软更新（τ 平滑追踪在线网络）。
        # 旧实现每 target_update_freq 步硬拷贝一次，目标网络阶跃跳变，配合 Q 值
        # 高估易造成「验证冲高后暴跌」的不稳定；软更新让目标平滑跟随。
        self._train_step_count += 1
        tau = self.config.target_update_tau
        if tau and tau > 0:
            with torch.no_grad():
                for target_p, online_p in zip(
                    self.target_network.parameters(), self.q_network.parameters()
                ):
                    target_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)
        elif self._train_step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "td_error": float(td_errors.abs().mean().item()),
        }

    def decay_epsilon(self, episode: int = 0, total_episodes: int = 0) -> None:
        """按训练进度线性退火探索率（ε 每个 episode 更新一次，episode 内保持不变）

        在「前 epsilon_anneal_ratio × total_episodes」轮内从 epsilon_start 线性降到
        epsilon_end，之后固定为 epsilon_end，为训练后段留出纯利用阶段。

        旧实现是每轮 ε × epsilon_decay 的指数衰减：decay=0.99 时约第 458 轮就触底
        0.01，3000 轮训练里后 85% 轮次纯利用。本任务的成本结构决定「不交易 = reward 0」
        是一个强局部最优，agent 早期偶然锁进去后再无探索机会翻盘，故改为与总轮数绑定
        的线性退火。未传 total_episodes（如单元测试直接调用）时退回原指数衰减。
        """
        if total_episodes > 0:
            ratio = max(1e-6, self.config.epsilon_anneal_ratio)
            progress = (episode + 1) / total_episodes
            if progress >= ratio:
                self.epsilon = self.config.epsilon_end
            else:
                frac = progress / ratio
                self.epsilon = (
                    self.config.epsilon_start
                    + frac * (self.config.epsilon_end - self.config.epsilon_start)
                )
        else:
            self.epsilon = max(
                self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
            )

    def save(self, path: str) -> None:
        """保存模型到指定路径（含 PER 缓冲区，支持断点续训）"""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self._train_step_count,
                # 状态维度元数据：加载时据此对齐网络（旧 checkpoint 无此字段，改用权重形状反推）
                "state_dim": self.config.state_dim,
                "signal_state_dims": self.config.signal_state_dims,
                # 优先级经验回放缓冲区（断点续训用）
                "replay_buffer": (
                    self.replay_buffer.serialize() if len(self.replay_buffer) > 0 else None
                ),
            },
            path,
        )

    def _checkpoint_state_dim(self, checkpoint: Dict) -> Optional[int]:
        """反推 checkpoint 训练时的 state_dim（用于兼容先验维度不同的历史模型）

        优先读 save 时写入的元数据；旧 checkpoint 无该字段，则按网络首层输入维反推：
        - CNN 编码器：state 与形态向量拼接后进 shared，首层输入 = state_dim + cnn_out_dim
        - Dueling MLP：shared 首层输入 = state_dim
        - 朴素 MLP：net 首层输入 = state_dim
        无法识别时返回 None，调用方回退到当前 config（保持旧行为）。
        """
        dim = checkpoint.get("state_dim")
        if isinstance(dim, int) and dim > 0:
            return dim
        sd = checkpoint.get("q_network") or {}
        if self.config.use_cnn_encoder:
            w = sd.get("shared.0.weight")
            return (int(w.shape[1]) - self.config.cnn_out_dim) if w is not None else None
        if "shared.0.weight" in sd:
            return int(sd["shared.0.weight"].shape[1])
        if "net.0.weight" in sd:
            return int(sd["net.0.weight"].shape[1])
        return None

    def load(self, path: str) -> None:
        """从指定路径加载模型（自动恢复 PER 缓冲区，若存在）

        状态维度对齐：先验买卖点编码存在 1 维（净信号，现用）与 2 维（买/卖分列，旧版）
        两代，目录名无法区分，故按 checkpoint 反推真实 state_dim，与当前 config 不一致时
        按反推值就地重建网络再加载。这样旧 10 维模型仍可评估/回放，且加载后 self.config
        已被更新为与权重匹配的口径（调用方应改用 model.config 构建环境，保证 state_dim 一致）。

        设备无关加载：checkpoint 可能由 GPU/CPU 进程保存（旧版 --no-gpu 因
        CUDA_VISIBLE_DEVICES="" 失效而实际仍在 GPU 保存），统一先 map_location='cpu'
        再迁移到当前设备，避免加载时崩溃：
        RuntimeError: Attempting to deserialize object on CUDA device 0 but
        torch.cuda.device_count() is 0。
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_dim = self._checkpoint_state_dim(checkpoint)
        if ckpt_dim is not None and ckpt_dim != self.config.state_dim:
            # 仅先验维度可变（基础 8 维固定）：8=无先验，9/10=先验 1/2 维
            self.config = dataclasses.replace(
                self.config,
                use_signal_scores=ckpt_dim > 8,
                signal_state_dims=max(1, ckpt_dim - 8),
            )
            logger.info(
                f"checkpoint state_dim={ckpt_dim} 与配置 {self.config.state_dim} 不一致，"
                f"已按 checkpoint 重建网络（先验 {self.config.signal_state_dims} 维）"
            )
            self._build()
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        # 迁移到当前设备（Module.to 就地修改参数并返回 self）
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        # 优化器状态显式迁移到参数所在设备（torch<2.0 的 load_state_dict 不会自动迁移）
        opt_state = checkpoint["optimizer"]
        for state in opt_state.get("state", {}).values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        self.optimizer.load_state_dict(opt_state)
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_start)
        self._train_step_count = checkpoint.get("train_step", 0)
        # 恢复经验回放缓冲区（兼容旧版元组列表格式）
        buffer_data = checkpoint.get("replay_buffer")
        if isinstance(buffer_data, dict):
            self.replay_buffer.restore(buffer_data)
        elif isinstance(buffer_data, list):
            for state, action, reward, next_state, done in buffer_data:
                self.replay_buffer.push(state, action, reward, next_state, done)

    def get_networks(self) -> Dict[str, nn.Module]:
        """返回所有网络模块"""
        return {"q_network": self.q_network, "target_network": self.target_network}
