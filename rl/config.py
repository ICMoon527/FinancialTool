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
    dense_reward_scale: float = 20.0
    warmup_steps: int = 20

    # ── 交易成本配置 ──
    commission_rate: float = 0.001       # 0.1%
    slippage_rate: float = 0.001         # 0.1%

    # ── DQN 特有 ──
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    replay_buffer_size: int = 10000
    target_update_freq: int = 100
    dqn_double: bool = True
    dqn_dueling: bool = True
    dqn_hidden_sizes: Tuple[int, ...] = (256, 128, 64)

    # ── PPO 特有 ──
    ppo_clip_epsilon: float = 0.2
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5

    # ── 训练控制 ──
    validation_freq: int = 50            # 每 N 个 episode 验证一次
    early_stopping_patience: int = 10    # 连续 N 次验证未提升则停止
    reward_clip: float = 5.0             # reward 裁剪范围 [-5, 5]

    # ── 模型存储 ──
    model_dir: str = "rl/models"
    save_best_only: bool = True

    # ── 状态特征扩展 ──
    # 是否把分时做T规则引擎（SignalEvaluator）的买点/卖点得分加入状态特征（+2维）
    # 注意：开启后 state_dim=52，与已有 50 维模型权重不兼容；仅用于新训练的对照实验
    use_signal_scores: bool = False

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
        """单次交易成本（一买一卖合计）"""
        return self.commission_rate * 2 + self.slippage_rate * 2

    @property
    def state_dim(self) -> int:
        """状态空间维度：基础 50 维 + 可选规则买卖点得分 2 维"""
        return 52 if self.use_signal_scores else 50  # 基础特征见规格文档 2.2 节

    @property
    def action_dim(self) -> int:
        """动作空间维度"""
        return 3  # HOLD / BUY / SELL