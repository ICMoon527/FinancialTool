# -*- coding: utf-8 -*-
"""分时做T RL 环境

核心约束：
- 底仓管理：初始 3 份底仓，尾盘强制恢复为 3 份（卖出的底仓按收盘价买回补齐）
- 动作空间：0=HOLD, 1~3=BUY1/2/3, 4~6=SELL1/2/3（一次可买卖多份；当日累计买入 ≤ 3 份，卖出 ≤ 底仓）
- T+1 规则：当天买入的份额不可当天卖出，SELL1/2/3 只卖底仓（先卖后买），尾盘按收盘价买回补齐 3 份
- 未配对的当日买入尾盘强制平仓结算，仅保留 3 份底仓（维持现金流，防止一直买入/卖出）
- 预热期：前 warmup_steps 步强制 HOLD + reward=0
- 每个交易日为一个独立 episode

奖励机制（密集浮盈信号，r = ΔPnL - cost - λ·inventory² + terminal_constraint）：
- ΔPnL：总持仓（底仓 + 当日买入）的逐分钟 mark-to-market 盈亏（% 刻度），
  让模型在每根K线都能感知持仓浮盈变化，而非仅事件驱动
- cost ：每笔成交的单边交易成本（佣金+滑点，每份 0.1%），含尾盘强平/买回成本
- λ·inventory²：库存惩罚（λ=reward_lambda，库存=agent可控的日内双向敞口=当日买入+先卖后买，
  不含恒定底仓；不做T时惩罚为0，避免恒定偏置淹没学习信号）
- terminal_constraint：收盘仍有未平当日买入时惩罚 -κ·leftover²（κ=reward_terminal_coef），
  鼓励日内 SELL 配对平仓而非拖到收盘强平

性能簿记（不参与 reward）：_realized_pnl 仍按真实 T+1 账目精确结算当日做T已实现收益，
供训练验证与评估输出真实绩效数字。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from watchdog.strategies.intraday_t0_strategy import (
    IndicatorSnapshot,
    IntradayDataBuffer,
    IntradayIndicatorEngine,
    SignalEvaluator,
)

if TYPE_CHECKING:
    from rl.config import RLConfig

logger = logging.getLogger(__name__)


class T0Environment:
    """分时做T RL 环境"""

    # ── 动作常量 ──
    HOLD: int = 0
    BUY1: int = 1   # 买入 1 份
    BUY2: int = 2   # 买入 2 份
    BUY3: int = 3   # 买入 3 份
    SELL1: int = 4  # 卖出 1 份
    SELL2: int = 5  # 卖出 2 份
    SELL3: int = 6  # 卖出 3 份
    ACTION_NAMES: Dict[int, str] = {
        0: "HOLD", 1: "BUY1", 2: "BUY2", 3: "BUY3",
        4: "SELL1", 5: "SELL2", 6: "SELL3",
    }

    # 买卖动作 → 份数（一次可买卖多份；买入总量 ≤ MAX_TODAY_BUY，卖出总量 ≤ 可卖持仓）
    BUY_AMOUNTS: Dict[int, int] = {BUY1: 1, BUY2: 2, BUY3: 3}
    SELL_AMOUNTS: Dict[int, int] = {SELL1: 1, SELL2: 2, SELL3: 3}

    # ── 最大持仓份数 ──
    MAX_BASE_POSITION: int = 3  # 底仓（尾盘强制恢复到此份数）
    MAX_TODAY_BUY: int = 3      # 当日最多累计买入份数
    MAX_STEPS: int = 240         # 9:30-15:00，240根1分钟K线

    def __init__(
        self,
        config: "RLConfig",
        indicator_engine: Optional[IntradayIndicatorEngine] = None,
    ):
        """
        Args:
            config: RL 配置
            indicator_engine: 指标计算引擎（复用现有 IntradayIndicatorEngine）
        """
        self.config = config
        self.warmup_steps = config.warmup_steps

        # 指标计算
        self._indicator_engine = indicator_engine or IntradayIndicatorEngine()
        self._data_buffer = IntradayDataBuffer(max_window=200)

        # 规则买卖点评分器（use_signal_scores=True 时为状态特征提供人工先验）
        self._signal_evaluator = SignalEvaluator()

        # 环境状态
        self._step: int = 0
        self._is_warmup: bool = False
        self._done: bool = False

        # 持仓状态
        self._base_position: int = 0          # 底仓（可卖出）
        self._today_bought: List[float] = []  # 当日买入的每笔成本
        self._avg_cost: float = 0.0           # 加权平均成本
        self._unrealized_pnl: float = 0.0     # 未实现盈亏
        self._realized_pnl: float = 0.0       # 已实现盈亏
        self._total_reward: float = 0.0       # episode 累计 reward

        # 当日价格统计
        self._day_open: float = 0.0
        self._day_high: float = 0.0
        self._day_low: float = float("inf")
        self._prev_close: float = 0.0

        # 单步交易成本累计（reward 的 cost 项：每笔成交单边佣金+滑点）
        self._step_cost: float = 0.0

        # 当前K线数据（由 step 更新）
        self._current_kline: Optional[Dict] = None
        self._current_indicator: Optional[IndicatorSnapshot] = None

        # K线历史（用于密集奖励计算和引用）
        self._klines: List[Dict] = []

        # 交易记录
        self._trades: List[Dict] = []          # 完整交易记录
        self._pending_buys: List[Dict] = []    # 未配对的买入记录（FIFO）
        self._sold_short: List[float] = []     # 先卖后买：已卖出底仓的卖出价（尾盘买回补齐3份底仓）

        # 当前 episode 样本信息
        self._current_stock_code: str = ""
        self._current_date: str = ""

        # 预热数据计数（用于 MACD_Bar_Sum 从当日第一根K线开始累加）
        self._warmup_bar_count: int = 0

    # ═══════════════════════════════════════════════
    #  公开接口
    # ═══════════════════════════════════════════════

    def reset(
        self,
        sample: Dict[str, object],
        prev_day_klines: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """重置环境到新 episode

        每个交易日为一个独立 episode，episode 开始时重置底仓为 3 份、
        今日买入标志为 False、未实现盈亏为 0。

        Args:
            sample: 当日样本数据，包含：
                - klines: List[Dict]  当日K线
                - stock_code: str     股票代码
                - date: date          交易日
            prev_day_klines: 前一日最后 N 根K线（用于预热），None 时启用 episode 内预热

        Returns:
            state: np.ndarray 形状 (state_dim,)，初始状态向量
        """
        # 重置持仓状态
        self._base_position = self.MAX_BASE_POSITION
        self._today_bought = []
        self._avg_cost = 0.0
        self._unrealized_pnl = 0.0
        self._realized_pnl = 0.0
        self._total_reward = 0.0
        self._step_cost = 0.0

        # 重置价格统计
        self._day_high = 0.0
        self._day_low = float("inf")

        # 重置环境状态
        self._step = 0
        self._is_warmup = False
        self._done = False
        self._current_kline = None
        self._current_indicator = None
        self._klines = []
        self._trades = []
        self._pending_buys = []
        self._sold_short = []

        # 加载样本
        self._klines = sample.get("klines", [])
        self._current_stock_code = str(sample.get("stock_code", ""))
        if hasattr(sample.get("date"), "isoformat"):
            self._current_date = sample["date"].isoformat()

        # 预热指标
        self._data_buffer = IntradayDataBuffer(max_window=200)
        self._warmup_bar_count = 0
        if prev_day_klines:
            self._data_buffer.warmup(prev_day_klines)
            self._warmup_bar_count = len(prev_day_klines)
            self._is_warmup = False  # 有前日数据，无需 episode 内预热
            # 用前日数据预计算指标初始状态
            self._precompute_indicators_from_buffer()
        else:
            self._is_warmup = True  # 无前日数据，需要 episode 内预热

        # 推进到第一根K线
        if self._klines:
            self._advance_kline()
            self._update_indicators()
            self._update_price_stats()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步环境交互

        Args:
            action: 0=HOLD, 1=BUY1, 2=BUY2, 3=BUY3, 4=SELL1, 5=SELL2, 6=SELL3

        Returns:
            next_state: np.ndarray  下一状态向量
            reward: float           即时奖励（基于当日已实现做T收益增量，与单根K线无关）
            done: bool              episode 是否结束
            info: dict              额外信息
        """
        info = {
            "is_warmup": self._is_warmup,
            "action_valid": True,
            "action_applied": action,
            "position": 0,
            "price": 0.0,
            "step": self._step,
        }

        # ── 预热期统一处理入口 ──
        if self._is_warmup and self._step < self.warmup_steps:
            info["is_warmup"] = True
            info["action_applied"] = self.HOLD

            self._step += 1
            if self._step < len(self._klines):
                self._advance_kline()
                self._update_indicators()
                self._update_price_stats()

            # 预热期结束时检查
            if self._step >= self.warmup_steps:
                self._is_warmup = False

            next_state = self._get_state()
            return next_state, 0.0, False, info

        # ── 正常交易期 ──
        self._is_warmup = False

        # 0. 记录动作前状态（奖励计算用：本步成交价 = 当前K线收盘价）
        prev_close = self._current_kline["Close"] if self._current_kline else 0.0
        self._step_cost = 0.0

        # 1. 动作合法性校验
        is_valid, applied_action = self._parse_action(action)
        info["action_valid"] = is_valid
        info["action_applied"] = applied_action

        # 2. 执行动作（交易成本在 _execute_action 内累计到 _step_cost）
        if is_valid and applied_action != self.HOLD:
            self._execute_action(applied_action)

        # 3. 推进到下一根K线
        self._step += 1
        if self._step < len(self._klines):
            self._advance_kline()
            self._update_indicators()
            self._update_price_stats()

        # 4. 终态约束（奖励 r 的 terminal_constraint 项）与收盘强平簿记
        terminal_constraint = 0.0
        if self._step >= len(self._klines) or self._step >= self.MAX_STEPS:
            self._done = True
            # 先捕获未平当日买入数（_force_close 会清空），再强平结算
            leftover = len(self._today_bought)
            terminal_constraint = -self.config.reward_terminal_coef * leftover**2
            self._force_close()

        # 5. 奖励计算：r = ΔPnL - cost - λ·inventory² + terminal_constraint
        #    ΔPnL：总持仓逐分钟 mark-to-market 盈亏（%刻度，密集浮盈信号）
        cur_close = self._current_kline["Close"] if self._current_kline else prev_close
        if prev_close > 0:
            price_return = (cur_close - prev_close) / prev_close * 100.0
        else:
            price_return = 0.0
        mtm_delta = self.total_position * price_return
        # 库存惩罚只针对 agent 可控的日内双向敞口（当日买入 + 先卖后买的卖出底仓），
        # 恒定底仓 3 份不参与惩罚（否则每步恒定 -λ·9 偏置会淹没学习信号）
        intraday_exposure = len(self._today_bought) + len(self._sold_short)
        inv_penalty = self.config.reward_lambda * intraday_exposure**2

        reward = mtm_delta - self._step_cost - inv_penalty + terminal_constraint
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        self._total_reward += reward

        next_state = self._get_state()
        info["position"] = self.total_position
        info["price"] = self._current_kline["Close"] if self._current_kline else 0.0
        info["step"] = self._step

        return next_state, reward, self._done, info

    # ═══════════════════════════════════════════════
    #  状态构建
    # ═══════════════════════════════════════════════

    def _get_state(self) -> np.ndarray:
        """构建 18 维状态向量（use_signal_scores=True 时 20 维），所有特征归一化到合理范围

        组成：OHLCV(5) + 多尺度return(4: 1/5/15/60根) + 波动率(1)
              + 时间编码(3) + 仓位状态(5) [+ 规则买卖点得分(2)]
        """
        k = self._current_kline
        features = []

        # ── OHLCV (5维)：原始K线数据 ──
        if k:
            features.extend([
                k["Open"] / 100.0,
                k["High"] / 100.0,
                k["Low"] / 100.0,
                k["Close"] / 100.0,
                np.log1p(k["Volume"]) / 20.0,
            ])
        else:
            features.extend([0.0] * 5)

        # ── 多尺度 return (4维) 与 波动率 (1维) ──
        # 基于缓冲区已见收盘价（含前日K线，跨日上下文）；未足窗口时补 0
        data = self._data_buffer.data
        closes = data["Close"].tolist() if len(data) else []
        for n in (1, 5, 15, 60):
            if len(closes) > n and closes[-n - 1] > 0:
                features.append((closes[-1] / closes[-n - 1] - 1.0) * 100.0)
            else:
                features.append(0.0)
        if len(closes) >= 2:
            log_ret = np.diff(np.log(np.maximum(closes[-21:], 1e-8)))
            features.append(float(np.std(log_ret)) * 100.0)
        else:
            features.append(0.0)

        # ── 时间编码 (3维) ──
        bar_index = self._step
        max_steps = max(len(self._klines), self.MAX_STEPS)
        features.extend([
            np.sin(2 * np.pi * bar_index / max_steps),
            np.cos(2 * np.pi * bar_index / max_steps),
            (max_steps - bar_index) / max_steps,  # bars_remaining
        ])

        # ── 仓位状态 (5维) ──
        unrealized_pct = 0.0
        if k and self._avg_cost > 0:
            unrealized_pct = (k["Close"] / self._avg_cost - 1.0) * 100.0
        features.extend([
            self._base_position / self.MAX_BASE_POSITION,
            len(self._today_bought) / self.MAX_TODAY_BUY,
            self.total_position / (self.MAX_BASE_POSITION + self.MAX_TODAY_BUY),
            self._avg_cost / 100.0 if self._avg_cost > 0 else 0.0,
            unrealized_pct,
        ])

        # ── 规则买卖点得分 (2维，use_signal_scores=True 时启用) ──
        # 直接复用分时做T页面同款 SignalEvaluator 规则引擎的原始得分，
        # 作为人工先验注入状态；引力场部分不参与（环境无参考线数据）。
        # 买/卖满分约 40/35（规则权重和），除以 10 归一化到约 0~4 区间
        if self.config.use_signal_scores:
            buy_score = sell_score = 0.0
            ind = self._current_indicator
            if ind is not None:
                buy_score = float(self._signal_evaluator.evaluate_buy(ind)[0])
                sell_score = float(self._signal_evaluator.evaluate_sell(ind)[0])
            features.extend([buy_score / 10.0, sell_score / 10.0])

        return np.array(features, dtype=np.float32)

    # ═══════════════════════════════════════════════
    #  动作解析与执行
    # ═══════════════════════════════════════════════

    def _parse_action(self, action: int) -> Tuple[bool, int]:
        """校验动作合法性并返回实际执行的动作

        Returns:
            (is_valid, applied_action)
        """
        if action == self.HOLD:
            return True, self.HOLD
        elif action in self.BUY_AMOUNTS:
            # 当日累计买入 ≤ MAX_TODAY_BUY（一次可买多份，超出剩余额度则无效）
            n = self.BUY_AMOUNTS[action]
            if len(self._today_bought) + n <= self.MAX_TODAY_BUY:
                return True, action
            return False, self.HOLD
        elif action in self.SELL_AMOUNTS:
            # T+1：当天买入不可卖出，SELL 只卖底仓（先卖后买）。一次可卖 N 份（N=1/2/3），需 ≤ 底仓
            n = self.SELL_AMOUNTS[action]
            if n <= self._base_position:
                return True, action
            return False, self.HOLD
        return False, self.HOLD

    def _execute_action(self, action: int) -> float:
        """执行交易动作，返回本轮已实现盈亏（T+1 下 SELL 卖底仓的盈亏延后到尾盘买回时结算）"""
        if not self._current_kline:
            return 0.0

        price = self._current_kline["Close"]
        timestamp = self._klines[self._step]["timestamp"] if self._step < len(self._klines) else ""

        if action in self.BUY_AMOUNTS:
            # 一次买入 N 份（N=1/2/3），当日累计买入 ≤ MAX_TODAY_BUY
            n = self.BUY_AMOUNTS[action]
            # reward 的 cost 项：单边买入成本（佣金+滑点，每份）
            self._step_cost += n * self.config.per_side_cost
            for _ in range(n):
                self._today_bought.append(price)
                self._pending_buys.append({
                    "time": timestamp,
                    "price": price,
                    "action": "BUY",
                    "paired_at": None,  # 奖励配对：SELL 卖出底仓时配对当日买入，记录配对卖出价
                })
                self._trades.append({
                    "time": timestamp,
                    "action": "BUY",
                    "price": price,
                    "pnl": 0.0,
                })

            # 更新平均成本
            total_cost = sum(self._today_bought) + self._base_position * self._avg_cost
            total_position = self.total_position
            self._avg_cost = total_cost / total_position if total_position > 0 else price
            return 0.0

        elif action in self.SELL_AMOUNTS:
            # T+1：当天买入不可卖出，SELL 只卖底仓（先卖后买），尾盘按收盘价买回补齐 3 份底仓
            n = self.SELL_AMOUNTS[action]
            # reward 的 cost 项：单边卖出成本（佣金+滑点，每份）
            self._step_cost += n * self.config.per_side_cost
            total_reward = 0.0
            for _ in range(n):
                # 奖励配对：若有未配对的当日买入，按（卖出价-买入成本）立即计入做T收益，
                # 并把该买入标记为已配对（paired_at=卖出价），尾盘只结算残余（收盘价-配对卖出价）
                unpaired = None
                for buy_record in self._pending_buys:
                    if buy_record.get("paired_at") is None:
                        unpaired = buy_record
                        break
                if unpaired is not None:
                    buy_price = unpaired["price"]
                    unpaired["paired_at"] = price
                    gross_return = (price - buy_price) / buy_price * 100
                    trade_reward = gross_return - self.config.transaction_cost
                    self._realized_pnl += trade_reward
                    total_reward += trade_reward
                # 无论是否配对，卖出都作用于底仓（T+1 仓位约束：当日买入保持锁定）
                self._base_position -= 1
                self._sold_short.append(price)  # 记录卖出价，尾盘买回补齐底仓

            self._trades.append({
                "time": timestamp,
                "action": f"SELL{n}",
                "price": price,
                "pnl": total_reward,
            })
            return total_reward

        return 0.0

    # ═══════════════════════════════════════════════
    #  奖励函数
    # ═══════════════════════════════════════════════

    def _force_close(self) -> float:
        """收盘强制平仓，使用收盘价，仅保留 3 份底仓

        1. 当日买入（pending_buys）全部强平结算（先买后卖/收盘平）；
        2. 先卖后买（sold_short）按收盘价买回补齐底仓到 3 份（卖出价高于买回价则赚）。

        Returns:
            平仓奖励（仅作簿记，新奖励机制下 reward 的终态约束由 terminal_constraint 承担）
        """
        if not self._current_kline:
            return 0.0

        close_price = self._current_kline["Close"]
        close_reward = 0.0

        # 1. 强平所有当日买入（当日买入尾盘必须平掉，仅保留 3 份底仓）
        if self._pending_buys:
            total_pnl = 0.0
            for buy_record in self._pending_buys:
                buy_price = buy_record["price"]
                paired_at = buy_record.get("paired_at")
                if buy_price > 0 and close_price > 0:
                    if paired_at is not None:
                        # 已配对：SELL 时已按（卖出价-买入价）计入做T收益并计过一次成本，
                        # 此处只结算残余（收盘价-配对卖出价），统一以买入价为基准，保证全天总收益与真实 T+1 账目精确一致、不重复计成本
                        gross_return = (close_price - paired_at) / buy_price * 100
                    else:
                        # 未配对：按收盘价结算（收盘价-买入价）
                        ref_price = buy_price
                        gross_return = (close_price - ref_price) / ref_price * 100 - self.config.transaction_cost
                    total_pnl += gross_return
            self._realized_pnl += total_pnl
            n_pending = len(self._pending_buys)
            # reward 的 cost 项：强平卖出成本（每份单边）
            self._step_cost += n_pending * self.config.per_side_cost
            self._pending_buys = []
            self._today_bought = []  # 当日买入已全部强平，同步清空（保持 total_position = 底仓）
            # 小惩罚：鼓励日内 SELL 配对实现做T，而非拖到收盘
            close_reward += -0.1 * n_pending

        # 2. 先卖后买：按收盘价买回卖出的底仓，恢复底仓到 3 份
        if self._sold_short:
            total_pnl = 0.0
            for sell_price in self._sold_short:
                if sell_price > 0 and close_price > 0:
                    # 先卖后买：卖出价高于买回价（收盘价）则赚
                    gross_return = (sell_price - close_price) / close_price * 100
                    total_pnl += gross_return - self.config.transaction_cost
            self._realized_pnl += total_pnl
            # reward 的 cost 项：买回底仓成本（每份单边）
            self._step_cost += len(self._sold_short) * self.config.per_side_cost
            self._sold_short = []
            # 恢复底仓到 MAX_BASE_POSITION（3 份）
            self._base_position = self.MAX_BASE_POSITION

        return close_reward

    # ═══════════════════════════════════════════════
    #  内部辅助方法
    # ═══════════════════════════════════════════════

    def _advance_kline(self) -> None:
        """推进到当前 step 对应的K线"""
        if self._step < len(self._klines):
            self._current_kline = self._klines[self._step]

    def _update_indicators(self) -> None:
        """将当前K线 feed 到指标引擎，更新 _current_indicator"""
        if not self._current_kline:
            self._current_indicator = None
            return

        # Feed K线到数据缓冲区
        self._data_buffer.append(self._current_kline)

        # 原始基线（use_signal_scores=False）：状态仅用 OHLCV/return/vol，
        # 无需计算技术指标，跳过以节省每步开销
        if not self.config.use_signal_scores:
            self._current_indicator = None
            return

        # 计算指标
        data = self._data_buffer.data
        if len(data) < 5:
            self._current_indicator = IndicatorSnapshot()
            return

        try:
            # 计算各指标
            ind = IndicatorSnapshot()

            # 主力吸筹/出货
            abs_data = self._indicator_engine.calc_absorption(data)
            dist_data = self._indicator_engine.calc_distribution(data)

            if "absorption" in abs_data.columns:
                ind.absorption_value = float(abs_data["absorption"].iloc[-1])
                ind.absorption_active = abs(ind.absorption_value) > 0.5

            if "distribution" in dist_data.columns:
                ind.distribution_active = dist_data["distribution"].iloc[-1] != 0

            # 量能
            close = data["Close"]
            volume = data["Volume"]
            if len(volume) >= self._indicator_engine.vol_ma_period:
                vol_ma = volume.rolling(window=self._indicator_engine.vol_ma_period).mean()
                ind.volume_surge = bool(volume.iloc[-1] > vol_ma.iloc[-1] * self._indicator_engine.vol_surge_ratio)
                ind.volume_shrink = bool(volume.iloc[-1] < vol_ma.iloc[-1] * 0.5)

            # 价格均线
            if len(close) >= self._indicator_engine.price_ma20_period:
                ma5 = close.rolling(window=self._indicator_engine.price_ma5_period).mean()
                ma20 = close.rolling(window=self._indicator_engine.price_ma20_period).mean()
                ind.ma5 = float(ma5.iloc[-1])
                ind.ma20 = float(ma20.iloc[-1])
                price = close.iloc[-1]
                ind.price_above_ma5 = price > ind.ma5
                ind.price_above_ma20 = price > ind.ma20
                if len(ma5) >= 2:
                    ind.price_cross_ma5_up = (close.iloc[-2] <= ma5.iloc[-2]) and (price > ma5.iloc[-1])
                    ind.price_cross_ma5_down = (close.iloc[-2] >= ma5.iloc[-2]) and (price < ma5.iloc[-1])

            # 均价偏离度
            if "AvgPrice" in data.columns:
                avg_price = data["AvgPrice"].iloc[-1]
                if avg_price > 0:
                    ind.avg_price = float(avg_price)
                    ind.deviation_pct = float((close.iloc[-1] - avg_price) / avg_price * 100)
                    ind.deviation_oversold = ind.deviation_pct < self._indicator_engine.dev_oversold_threshold
                    ind.deviation_overbought = ind.deviation_pct > self._indicator_engine.dev_overbought_threshold

            # MACD
            if len(close) >= self._indicator_engine.macd_slow + self._indicator_engine.macd_signal:
                ema_fast = close.ewm(span=self._indicator_engine.macd_fast, adjust=False).mean()
                ema_slow = close.ewm(span=self._indicator_engine.macd_slow, adjust=False).mean()
                dif = ema_fast - ema_slow
                dea = dif.ewm(span=self._indicator_engine.macd_signal, adjust=False).mean()
                macd_bar = 2 * (dif - dea)
                ind.dif = float(dif.iloc[-1])
                ind.dea = float(dea.iloc[-1])
                ind.macd_bar = float(macd_bar.iloc[-1])

                # 统一前后端 MACD_Bar_Sum 计算逻辑：从当日第一根K线开始累加，预热数据不参与
                # 预热数据（prev_day_klines）仅用于 EMA 初始化，不参与柱高和累加
                if self._warmup_bar_count > 0 and len(macd_bar) > self._warmup_bar_count:
                    current_day_bars = macd_bar.iloc[self._warmup_bar_count:]
                    ind.macd_bar_sum = float(current_day_bars.sum())
                else:
                    # 无预热数据或 episode 内预热：从第一根K线开始累加
                    ind.macd_bar_sum = float(macd_bar.sum())
                if len(macd_bar) >= 2 and macd_bar.iloc[-2] != 0:
                    ind.macd_bar_diff = float((macd_bar.iloc[-1] - macd_bar.iloc[-2]) / macd_bar.iloc[-2])
                else:
                    ind.macd_bar_diff = 0.0

                # 金叉死叉
                if len(dif) >= 2:
                    ind.macd_golden_cross = (dif.iloc[-2] <= dea.iloc[-2]) and (dif.iloc[-1] > dea.iloc[-1])
                    ind.macd_death_cross = (dif.iloc[-2] >= dea.iloc[-2]) and (dif.iloc[-1] < dea.iloc[-1])
                # 多头动能衰减/空头动能衰竭
                if len(macd_bar) >= 2:
                    ind.macd_bullish_weakening = (macd_bar.iloc[-1] > 0) and (macd_bar.iloc[-1] < macd_bar.iloc[-2])
                    ind.macd_bearish_recovering = (macd_bar.iloc[-1] < 0) and (macd_bar.iloc[-1] > macd_bar.iloc[-2])

            # RSI
            if len(close) >= self._indicator_engine.rsi_period + 1:
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = (-delta).clip(lower=0)
                avg_gain = gain.rolling(window=self._indicator_engine.rsi_period).mean()
                avg_loss = loss.rolling(window=self._indicator_engine.rsi_period).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - 100 / (1 + rs)
                ind.rsi_value = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
                ind.rsi_oversold = ind.rsi_value < self._indicator_engine.rsi_oversold
                ind.rsi_overbought = ind.rsi_value > self._indicator_engine.rsi_overbought

            # KDJ
            if len(close) >= self._indicator_engine.kdj_n_period:
                low_n = data["Low"].rolling(window=self._indicator_engine.kdj_n_period).min()
                high_n = data["High"].rolling(window=self._indicator_engine.kdj_n_period).max()
                rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
                k = rsv.ewm(com=self._indicator_engine.kdj_m1_period - 1, adjust=False).mean()
                d = k.ewm(com=self._indicator_engine.kdj_m2_period - 1, adjust=False).mean()
                j = 3 * k - 2 * d
                ind.kdj_k = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0
                ind.kdj_d = float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0
                ind.kdj_j = float(j.iloc[-1]) if not np.isnan(j.iloc[-1]) else 50.0
                ind.kdj_oversold = ind.kdj_j < self._indicator_engine.kdj_oversold
                ind.kdj_overbought = ind.kdj_j > self._indicator_engine.kdj_overbought
                if len(k) >= 2:
                    ind.kdj_golden_cross = (k.iloc[-2] <= d.iloc[-2]) and (k.iloc[-1] > d.iloc[-1])
                    ind.kdj_death_cross = (k.iloc[-2] >= d.iloc[-2]) and (k.iloc[-1] < d.iloc[-1])

            # MFI
            if len(close) >= self._indicator_engine.mfi_period + 1:
                typical_price = (data["High"] + data["Low"] + close) / 3
                raw_money_flow = typical_price * volume
                pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
                neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
                pos_sum = pos_flow.rolling(window=self._indicator_engine.mfi_period).sum()
                neg_sum = neg_flow.rolling(window=self._indicator_engine.mfi_period).sum()
                money_ratio = pos_sum / neg_sum.replace(0, np.nan)
                mfi = 100 - 100 / (1 + money_ratio)
                ind.mfi_value = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50.0
                ind.mfi_oversold = ind.mfi_value < self._indicator_engine.mfi_oversold
                ind.mfi_overbought = ind.mfi_value > self._indicator_engine.mfi_overbought
                if len(mfi) >= 2:
                    ind.mfi_cross_50_up = (mfi.iloc[-2] <= 50) and (mfi.iloc[-1] > 50)
                    ind.mfi_cross_50_down = (mfi.iloc[-2] >= 50) and (mfi.iloc[-1] < 50)

            self._current_indicator = ind
        except Exception as e:
            logger.debug(f"指标计算异常: {e}")
            self._current_indicator = IndicatorSnapshot()

    def _update_price_stats(self) -> None:
        """更新当日价格统计"""
        if not self._current_kline:
            return
        price = self._current_kline["Close"]
        if self._day_high < price:
            self._day_high = price
        if price < self._day_low:
            self._day_low = price
        if self._day_open == 0.0:
            self._day_open = self._current_kline.get("Open", price)

    def _precompute_indicators_from_buffer(self) -> None:
        """用前日数据预计算指标初始状态（已通过 warmup 加载）"""
        data = self._data_buffer.data
        if data.empty:
            return
        # 创建一个临时 IndicatorSnapshot 作为初始状态
        self._current_indicator = IndicatorSnapshot()

    # ═══════════════════════════════════════════════
    #  属性
    # ═══════════════════════════════════════════════

    @property
    def total_position(self) -> int:
        """总持仓份数 = 底仓 + 当日买入"""
        return self._base_position + len(self._today_bought)

    @property
    def can_buy(self) -> bool:
        """是否还能买入（当日买入 < 3 份）"""
        return len(self._today_bought) < self.MAX_TODAY_BUY

    @property
    def can_sell(self) -> bool:
        """是否还能卖出（T+1：当天买入不可卖出，可卖 = 底仓 > 0）"""
        return self._base_position > 0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def current_step(self) -> int:
        return self._step