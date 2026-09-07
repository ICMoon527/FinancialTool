# -*- coding: utf-8 -*-
"""RL 模块全局配置，所有字段从 .env 或 config_registry.py 注入"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class RLConfig:
    """RL 模块全局配置"""

    # ── 通用配置 ──
    enabled: bool = False
    default_algorithm: Literal["dqn", "ppo"] = "dqn"
    training_episodes: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    train_data_days: int = 60
    validation_split: float = 0.2
    dense_reward_scale: float = 20.0  # 密集奖励缩放（R_dense = 持仓变动 × 价格变动% × scale，已弃用保留兼容）
    trade_act_bonus: float = 0.05     # 有效 BUY 的行为激励（鼓励做T操作，reward 主体为已实现做T收益增量）
    warmup_steps: int = 20

    # ── 交易成本配置（按 A 股真实成本，2023-08 起印花税 0.05% 卖出单边）──
    # commission_rate = 0.05% 每边：含双边佣金万2.5（0.025%）与卖出印花税 0.05% 的平均分摊
    # slippage_rate   = 0.05% 每边：滑点万5 保守估计
    # 一买一卖合计 = (0.05% + 0.05%) * 2 = 0.2%（此前 0.4% 过高，1分钟做T毛收益难以覆盖，
    # 导致模型学到的理性策略是 HOLD；0.2% 下做T在经济上可行）
    commission_rate: float = 0.0005       # 0.05%/边（佣金万2.5 双边 + 印花税 0.05% 卖出）
    slippage_rate: float = 0.0005         # 0.05%/边（滑点万5）

    # ── DQN 特有 ──
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99   # 按 episode 衰减系数（每轮衰减一次，约460轮到终值）
    replay_buffer_size: int = 10000
    target_update_freq: int = 100
    dqn_double: bool = True
    dqn_dueling: bool = True
    dqn_hidden_sizes: Tuple[int, ...] = (256, 128, 64)

    # ── PER（Prioritized Experience Replay）──
    # 优先级 P(i) ∝ (|δ_i| + ε)^α，按概率采样；用 IS 权重 (N·P)^(-β) 修正采样偏差
    per_alpha: float = 0.6          # 优先级幂指数 α（0=均匀采样，越大越偏向高TD误差样本）
    per_beta_start: float = 0.4     # IS 权重指数 β 初始值
    per_beta_end: float = 1.0       # IS 权重指数 β 终值（随训练步数线性退火）
    per_eps: float = 1e-6           # 优先级下界常数 ε（保证所有经验都可被采样）

    # ── PPO 特有 ──
    ppo_clip_epsilon: float = 0.2
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5

    # ── 奖励函数（r = ΔPnL - cost - λ·inventory² + terminal_constraint）──
    # ΔPnL：总持仓的逐分钟 mark-to-market 盈亏（% 刻度，密集浮盈信号）
    # cost ：单边交易成本（佣金+滑点，每份 0.1% = (commission+slippage)*100）
    # inventory：总持仓份数（底仓 3 + 当日买入 0~3）
    reward_lambda: float = 0.01          # 库存惩罚系数 λ（λ·inventory²，抑制过度建仓）
    reward_terminal_coef: float = 0.5    # 终态约束系数 κ（收盘仍有未平当日买入时 -κ·leftover²）

    # ── 训练控制 ──
    validation_freq: int = 50            # 每 N 个 episode 验证一次
    early_stopping_patience: int = 15    # 连续 N 次验证未提升则停止（验证指标为真实做T收益，耐心加大减少噪声误触发）
    reward_clip: float = 5.0             # reward 裁剪范围 [-5, 5]

    # ── 模型存储 ──
    model_dir: str = "rl/models"
    save_best_only: bool = True

    # ── 状态特征扩展 ──
    # 是否把分时做T规则引擎（SignalEvaluator）的买点/卖点得分加入状态特征（+2维）
    # 注意：开启后 state_dim=52，与已有 50 维模型权重不兼容；仅用于新训练的对照实验
    use_signal_scores: bool = False

    @property
    def model_tag(self) -> str:
        """模型目录/ID 使用的算法前缀；开启先验买卖点时加 _prior 标识，
        用于在文件夹名上区分 52 维（带先验）与 50 维（不带）模型"""
        if self.use_signal_scores:
            return f"{self.default_algorithm}_prior"
        return self.default_algorithm

    @classmethod
    def from_env(cls) -> "RLConfig":
        """从 .env 和 config_registry 加载配置"""
        config = cls()

        # 从 .env 加载所有 RL_ 前缀的环境变量
        env_map = {
            "RL_ENABLED": ("enabled", "bool"),
            "RL_DEFAULT_ALGORITHM": ("default_algorithm", "str"),
            "RL_TRAINING_EPISODES": ("training_episodes", "int"),
            "RL_BATCH_SIZE": ("batch_size", "int"),
            "RL_LEARNING_RATE": ("learning_rate", "float"),
            "RL_GAMMA": ("gamma", "float"),
            "RL_TRAIN_DATA_DAYS": ("train_data_days", "int"),
            "RL_VALIDATION_SPLIT": ("validation_split", "float"),
            "RL_DENSE_REWARD_SCALE": ("dense_reward_scale", "float"),
            "RL_TRADE_ACT_BONUS": ("trade_act_bonus", "float"),
            "RL_WARMUP_STEPS": ("warmup_steps", "int"),
            "RL_COMMISSION_RATE": ("commission_rate", "float"),
            "RL_SLIPPAGE_RATE": ("slippage_rate", "float"),
            "RL_EPSILON_START": ("epsilon_start", "float"),
            "RL_EPSILON_END": ("epsilon_end", "float"),
            "RL_EPSILON_DECAY": ("epsilon_decay", "float"),
            "RL_REPLAY_BUFFER_SIZE": ("replay_buffer_size", "int"),
            "RL_TARGET_UPDATE_FREQ": ("target_update_freq", "int"),
            "RL_DQN_DOUBLE": ("dqn_double", "bool"),
            "RL_DQN_DUELING": ("dqn_dueling", "bool"),
            "RL_DQN_HIDDEN_SIZES": ("dqn_hidden_sizes", "tuple_int"),
            "RL_PER_ALPHA": ("per_alpha", "float"),
            "RL_PER_BETA_START": ("per_beta_start", "float"),
            "RL_PER_BETA_END": ("per_beta_end", "float"),
            "RL_PER_EPS": ("per_eps", "float"),
            "RL_REWARD_LAMBDA": ("reward_lambda", "float"),
            "RL_REWARD_TERMINAL_COEF": ("reward_terminal_coef", "float"),
            "RL_PPO_CLIP_EPSILON": ("ppo_clip_epsilon", "float"),
            "RL_PPO_GAE_LAMBDA": ("ppo_gae_lambda", "float"),
            "RL_PPO_ENTROPY_COEF": ("ppo_entropy_coef", "float"),
            "RL_PPO_VALUE_COEF": ("ppo_value_coef", "float"),
            "RL_VALIDATION_FREQ": ("validation_freq", "int"),
            "RL_EARLY_STOPPING_PATIENCE": ("early_stopping_patience", "int"),
            "RL_REWARD_CLIP": ("reward_clip", "float"),
            "RL_MODEL_DIR": ("model_dir", "str"),
            "RL_SAVE_BEST_ONLY": ("save_best_only", "bool"),
            "RL_USE_SIGNAL_SCORES": ("use_signal_scores", "bool"),
        }

        for env_name, (attr_name, type_name) in env_map.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                try:
                    if type_name == "bool":
                        setattr(config, attr_name, env_value.lower() in ("true", "1", "yes"))
                    elif type_name == "int":
                        setattr(config, attr_name, int(env_value))
                    elif type_name == "float":
                        setattr(config, attr_name, float(env_value))
                    elif type_name == "tuple_int":
                        setattr(
                            config,
                            attr_name,
                            tuple(int(x.strip()) for x in env_value.split(",")),
                        )
                    else:
                        setattr(config, attr_name, env_value)
                except (ValueError, TypeError):
                    pass  # 使用默认值

        return config

    @property
    def transaction_cost(self) -> float:
        """单次交易成本（一买一卖合计），**百分比刻度**（与 reward 收益单位一致：1.0 = 1%）

        修复：原实现返回小数刻度 0.004（0.4%），而 reward 中 gross_return 为百分比刻度
        （价差/成本*100，1.0 表示 1%），直接相减会把 0.4% 成本低估 100 倍为 0.004%。
        现统一为百分比刻度 0.4，保证成本惩罚与收益同一量纲。
        """
        return (self.commission_rate * 2 + self.slippage_rate * 2) * 100

    @property
    def per_side_cost(self) -> float:
        """单边单份交易成本（%刻度）：佣金 + 滑点 = 0.1%（0.05% + 0.05%）"""
        return (self.commission_rate + self.slippage_rate) * 100

    @property
    def state_dim(self) -> int:
        """状态空间维度：基础 18 维 + 可选规则买卖点得分 2 维

        基础 18 维 = OHLCV(5) + 多尺度return(4: 1/5/15/60根) + 波动率(1)
                     + 时间编码(3) + 仓位状态(5)
        """
        return 20 if self.use_signal_scores else 18

    @property
    def action_dim(self) -> int:
        """动作空间维度"""
        return 7  # HOLD / BUY1 / BUY2 / BUY3 / SELL1 / SELL2 / SELL3