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
    train_data_days: int = 60  # 已弃用（保留兼容）：数据切分按 validation_split 全量执行，此字段无引用，修改不生效
    validation_split: float = 0.2
    dense_reward_scale: float = 20.0  # 已弃用（保留兼容）：奖励函数已不含该项，仅设置页展示，修改不生效
    trade_act_bonus: float = 0.05     # 已弃用（保留兼容）：奖励函数已不含行为激励，修改不生效
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
    target_update_freq: int = 100  # 目标网络硬拷贝间隔（步）；仅当 target_update_tau=0（关闭软更新）时生效
    # Polyak 软更新系数 τ（>0 时启用软更新，替代每 target_update_freq 步的硬拷贝）：
    # 硬拷贝会让目标网络每 100 步阶跃跳变，配合 Q 值高估易出现
    # 「验证冲高后暴跌」的不稳定（此前 E1350 +2.33% → E1400-1500 -19%~-41%）
    target_update_tau: float = 0.005
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

    # ── 空头尾部风险抑制（卖底仓做空）──
    # 基线评估暴露：模型偶尔「先卖后买」卖底仓做空，遇到当日暴力拉升（如600519单日+8%）
    # 被收盘强平买回，单笔亏 -8%，两笔就拖垮整个账户。单点动量检查挡不住「开盘价附近就做空、
    # 之后才拉升」的灾难尾。改为只在「当日已跌破开盘价（弱势）」时允许做空，逢强势一律禁空，
    # 从动作层面彻底切断空头灾难尾，而非依赖模型自行规避。
    short_guard_enabled: bool = True        # 是否启用卖底仓做空的当日弱势约束
    short_down_margin: float = 0.0          # 允许做空需当日相对开盘价回落到该值(%)以下（默认0：仅低于开盘价时做空）
    short_stop: float = 2.0                 # 做空日内硬止损(%)：价格相对卖出价反弹超该值立即买回封死亏损

    # ── 模型存储 ──
    model_dir: str = "rl/models"
    save_best_only: bool = True

    # ── 状态特征扩展 ──
    # 是否把分时做T规则引擎（SignalEvaluator）的买点/卖点得分加入状态特征（+2维）
    # 注意：开启后 state_dim=10（基础 8 维 + 先验 2 维），与未开启时的 8 维
    # 模型权重不兼容；仅用于新训练的对照实验
    use_signal_scores: bool = False

    # ── 前日K线时间拼接 ──
    # 是否在时间维度拼接前一日全天分时K线（约240根）到当日K线序列前，
    # 使 return_1/5/15/60、波动率等跨日特征在开盘时刻即有完整前日上下文，
    # 而非仅用前日尾盘30根预热。解决「开盘时刻状态无当日信息、只能开盘秒买赌方向」：
    # 模型可对比「今日当前价 vs 前日各时段」判断当日强弱再决定买卖时点。
    # 注意：状态维度不变（8/10），但特征值基于更长的跨日序列，权重需重新训练
    use_prev_day_features: bool = False

    # ── 1D-CNN 形态编码器（序列建模）──
    # 时间拼接让 buffer 含「前日全天+当日」K线，但 MLP 无法从长序列归纳形态；
    # 本开关启用 CNN 编码器：最近 cnn_window 根K线（时间序列拼接：前日全天 + 当日，
    # 8 通道）经三层空洞 Conv1d 编码为 cnn_out_dim 维形态向量，与 state_dim 状态拼接后
    # 进 MLP 输出 Q 值。模型可直接观察「昨日尾盘走势/今日开盘方向/V型W型」等形态。
    # 注意：开启后模型输入 = state_dim + cnn_out_dim，权重与纯 MLP 模型不兼容
    use_cnn_encoder: bool = False
    # 窗口 480 = 前日全天(240) + 当日全天(240)。旧值 60 会在 10:30 后把前日K线
    # 全部挤出窗口，前日信息实际只存活 1 小时，「时间序列拼接」名存实亡；
    # 扩到 480 后前日数据在整个交易日都可见（buffer max_window=500 已够容纳）。
    cnn_window: int = 480
    cnn_out_dim: int = 16           # 形态编码器输出维度（拼接进状态）
    cnn_hidden_channels: Tuple[int, ...] = (32, 64, 64)  # 三层 Conv1d 通道数
    cnn_kernel_size: int = 5        # Conv1d 卷积核大小
    # 空洞卷积膨胀率：逐层扩大感受野（1,2,4 时约 60 根K线），
    # 旧实现两层无膨胀感受野仅 9 根，连半小时形态都覆盖不到
    cnn_dilation: Tuple[int, ...] = (1, 2, 4)
    # 形态窗口输入通道数：OHLCV(5) + day_flag + intraday_pos + valid_mask = 8。
    # day_flag 标记该根属于前日(0)/当日(1)，intraday_pos 为日内归一化位置，
    # valid_mask 标记补零位，避免 CNN 把 padding 当作真实行情。
    cnn_in_channels: int = 8

    @property
    def model_tag(self) -> str:
        """模型目录/ID 使用的算法前缀；开启先验买卖点加 _prior，开启前日K线拼接加 _prevf，
        开启 CNN 形态编码器加 _cnn，用于在文件夹名上区分不同训练配置（权重结构不同，不能混用）"""
        tag = self.default_algorithm
        if self.use_signal_scores:
            tag += "_prior"
        if self.use_prev_day_features:
            tag += "_prevf"
        if self.use_cnn_encoder:
            tag += "_cnn"
        return tag

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
            "RL_USE_PREV_DAY_FEATURES": ("use_prev_day_features", "bool"),
            "RL_USE_CNN_ENCODER": ("use_cnn_encoder", "bool"),
            "RL_CNN_WINDOW": ("cnn_window", "int"),
            "RL_CNN_OUT_DIM": ("cnn_out_dim", "int"),
            "RL_CNN_HIDDEN_CHANNELS": ("cnn_hidden_channels", "tuple_int"),
            "RL_CNN_KERNEL_SIZE": ("cnn_kernel_size", "int"),
            "RL_CNN_DILATION": ("cnn_dilation", "tuple_int"),
            "RL_CNN_IN_CHANNELS": ("cnn_in_channels", "int"),
            "RL_TARGET_UPDATE_TAU": ("target_update_tau", "float"),
            "RL_SHORT_GUARD_ENABLED": ("short_guard_enabled", "bool"),
            "RL_SHORT_DOWN_MARGIN": ("short_down_margin", "float"),
            "RL_SHORT_STOP": ("short_stop", "float"),
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
        """状态空间维度：基础 8 维 + 可选规则买卖点得分 2 维

        基础 8 维 = OHLCV(5, 相对前收的百分比 / 相对均量倍数) + 时间编码(1: 距收盘剩余比例)
                    + 仓位状态(2: 底仓比例 / 有符号净敞口)
        市场时序信息（动量、波动率、形态）由 CNN 编码器承担，不占 state 维度；
        前日K线时间拼接同样不改变维度，只扩展 CNN 窗口的历史上下文。
        """
        return 10 if self.use_signal_scores else 8

    @property
    def action_dim(self) -> int:
        """动作空间维度"""
        return 7  # HOLD / BUY1 / BUY2 / BUY3 / SELL1 / SELL2 / SELL3