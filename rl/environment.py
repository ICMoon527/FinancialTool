# -*- coding: utf-8 -*-
"""分时做T RL 环境

核心约束：
- 底仓管理：初始 3 份底仓，尾盘强制恢复为 3 份（卖出的底仓按收盘价买回补齐）
- 动作空间：0=HOLD, 1~3=BUY1/2/3, 4~6=SELL1/2/3（一次可买卖多份；当日累计买入 ≤ 3 份，卖出 ≤ 底仓）
- T+1 规则：当天买入的份额不可当天卖出，SELL1/2/3 只卖底仓（先卖后买），尾盘按收盘价买回补齐 3 份
- 未配对的当日买入尾盘强制平仓结算，仅保留 3 份底仓（维持现金流，防止一直买入/卖出）
- 预热期：前 warmup_steps 步强制 HOLD + reward=0
- 每个交易日为一个独立 episode

奖励机制（与单根K线无关，只与当日做T已实现盈亏挂钩）：
- reward 主体 = realized_pnl 增量（收盘强平当日买入 / 尾盘买回底仓产生的已实现做T收益%）
- 有效 BUY/SELL 给小额行为激励（trade_act_bonus），鼓励做T操作、打破"HOLD=0"惰性
- 无效动作给 -1 惩罚；收盘仍有未强平当日买入给 -0.1/笔 惩罚（鼓励日内平仓）
- 当日做T盈利（realized_pnl>0）episode 结束额外 +1
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

        # 0. 记录动作前的已实现做T收益（reward 只基于该增量，与单根K线无关）
        prev_realized = self._realized_pnl

        # 1. 动作合法性校验
        is_valid, applied_action = self._parse_action(action)
        info["action_valid"] = is_valid
        info["action_applied"] = applied_action

        # 2. 执行动作（有效 BUY/SELL 给小额行为激励，鼓励做T操作；收益通过 realized_pnl 增量结算）
        act_bonus = 0.0
        if is_valid and (applied_action in self.BUY_AMOUNTS or applied_action in self.SELL_AMOUNTS):
            act_bonus = self.config.trade_act_bonus
        if is_valid and applied_action != self.HOLD:
            self._execute_action(applied_action)

        # 3. 推进到下一根K线
        self._step += 1
        if self._step < len(self._klines):
            self._advance_kline()
            self._update_indicators()
            self._update_price_stats()

        # 4. 检查是否结束（收盘强平会把未配对买入收益计入 realized_pnl）
        force_close_reward = 0.0
        episode_bonus = 0.0
        if self._step >= len(self._klines) or self._step >= self.MAX_STEPS:
            self._done = True
            force_close_reward = self._force_close()
            episode_bonus = self._compute_episode_bonus()

        # 5. 已实现做T收益增量（%）：当日做T盈利的核心信号，与单根K线无关
        realized_delta = self._realized_pnl - prev_realized

        # 6. 无效动作惩罚
        penalty = -1.0 if not is_valid else 0.0

        # 7. 汇总 reward
        reward = realized_delta + act_bonus + penalty + force_close_reward + episode_bonus
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
        """构建 ~50 维状态向量，所有特征归一化到合理范围"""
        ind = self._current_indicator
        k = self._current_kline
        features = []

        # ── 价格特征 (8维) ──
        if k:
            features.extend([
                k["Close"] / 100.0,
                k["Open"] / 100.0,
                k["High"] / 100.0,
                k["Low"] / 100.0,
            ])
            if ind:
                features.extend([
                    ind.deviation_pct / 100.0 if ind.deviation_pct else 0.0,
                    float(ind.price_above_ma5),
                    float(ind.price_above_ma20),
                    float(ind.price_cross_ma5_up) - float(ind.price_cross_ma5_down),
                ])
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 8)

        # ── 量能特征 (3维) ──
        if k:
            features.extend([
                np.log1p(k["Volume"]) / 20.0,
                float(ind.volume_surge) if ind else 0.0,
                float(ind.volume_shrink) if ind else 0.0,
            ])
        else:
            features.extend([0.0] * 3)

        # ── 主力吸筹/出货 (3维) ──
        if ind:
            features.extend([
                ind.absorption_value / 100.0 if ind.absorption_value else 0.0,
                float(ind.distribution_active),
                (ind.main_in_value - ind.main_out_value) / 100.0,  # 主力进出差值
            ])
        else:
            features.extend([0.0] * 3)

        # ── MACD (8维) ──
        if ind:
            features.extend([
                ind.dif / 10.0 if ind.dif else 0.0,
                ind.dea / 10.0 if ind.dea else 0.0,
                ind.macd_bar / 10.0 if ind.macd_bar else 0.0,
                getattr(ind, "macd_bar_sum", 0.0) / 50.0,
                getattr(ind, "macd_bar_diff", 0.0),
                float(ind.macd_golden_cross),
                float(ind.macd_death_cross),
                float(ind.macd_bullish_weakening) - float(ind.macd_bearish_recovering),
            ])
        else:
            features.extend([0.0] * 8)

        # ── RSI (3维) ──
        if ind:
            features.extend([
                ind.rsi_value / 100.0,
                float(ind.rsi_oversold),
                float(ind.rsi_overbought),
            ])
        else:
            features.extend([0.0] * 3)

        # ── KDJ (7维) ──
        if ind:
            features.extend([
                ind.kdj_k / 100.0,
                ind.kdj_d / 100.0,
                ind.kdj_j / 100.0,
                float(ind.kdj_oversold),
                float(ind.kdj_overbought),
                float(ind.kdj_golden_cross),
                float(ind.kdj_death_cross),
            ])
        else:
            features.extend([0.0] * 7)

        # ── MFI (6维) ──
        if ind:
            features.extend([
                ind.mfi_value / 100.0,
                float(ind.mfi_oversold),
                float(ind.mfi_overbought),
                float(ind.mfi_cross_50_up),
                float(ind.mfi_cross_50_down),
                float(ind.mfi_bottom_divergence) - float(ind.mfi_top_divergence),
            ])
        else:
            features.extend([0.0] * 6)

        # ── 时间特征 (3维) ──
        bar_index = self._step
        max_steps = max(len(self._klines), self.MAX_STEPS)
        features.extend([
            np.sin(2 * np.pi * bar_index / max_steps),
            np.cos(2 * np.pi * bar_index / max_steps),
            (max_steps - bar_index) / max_steps,  # bars_remaining
        ])

        # ── 相对位置 (1维) ──
        if self._day_high > self._day_low:
            price = k["Close"] if k else 0
            day_position = (price - self._day_low) / (self._day_high - self._day_low)
        else:
            day_position = 0.5
        features.append(day_position)

        # ── 仓位状态 (5维) ──
        features.extend([
            self._base_position / 3.0,
            len(self._today_bought) / 3.0,
            self.total_position / 6.0,
            self._avg_cost / 100.0 if self._avg_cost > 0 else 0.0,
            self._unrealized_pnl / 100.0,
        ])

        # ── 资金状态 (2维) ──
        max_buy = self.MAX_TODAY_BUY - len(self._today_bought)
        features.extend([
            float(max_buy > 0),              # 可用资金（有买入额度=1）
            len(self._today_bought) / 3.0,   # 已用资金比例
        ])

        # ── 预热标志 (1维) ──
        features.append(float(self._is_warmup))

        # ── 规则买卖点得分 (2维，use_signal_scores=True 时启用) ──
        # 直接复用分时做T页面同款 SignalEvaluator 规则引擎的原始得分，
        # 作为人工先验注入状态；引力场部分不参与（环境无参考线数据）。
        # 买/卖满分约 40/35（规则权重和），除以 10 归一化到约 0~4 区间，
        # 使 weak(~4)/medium(~5-7)/strong(≥6/11) 档位落在 0.4~1.1+ 的可区分范围
        if self.config.use_signal_scores:
            buy_score = sell_score = 0.0
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
            for _ in range(n):
                self._today_bought.append(price)
                self._pending_buys.append({
                    "time": timestamp,
                    "price": price,
                    "action": "BUY",
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
            # T+1：当天买入不可卖出。一次卖出 N 份底仓（先卖后买），尾盘按收盘价买回补齐 3 份底仓
            n = self.SELL_AMOUNTS[action]
            self._base_position -= n
            for _ in range(n):
                self._sold_short.append(price)  # 记录卖出价，尾盘买回结算

            self._trades.append({
                "time": timestamp,
                "action": f"SELL{n}",
                "price": price,
                "pnl": 0.0,
            })
            return 0.0

        return 0.0

    # ═══════════════════════════════════════════════
    #  奖励函数
    # ═══════════════════════════════════════════════

    def _compute_episode_bonus(self) -> float:
        """episode 结束时的奖励项"""
        bonus = 0.0
        if self._realized_pnl > 0:
            bonus += 1.0  # 当日正收益额外奖励
        return bonus

    def _force_close(self) -> float:
        """收盘强制平仓，使用收盘价，仅保留 3 份底仓

        1. 当日买入（pending_buys）全部强平结算（先买后卖/收盘平）；
        2. 先卖后买（sold_short）按收盘价买回补齐底仓到 3 份（卖出价高于买回价则赚）。

        Returns:
            平仓奖励（小负值，鼓励日内完成配对做T，而非拖到收盘）
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
                if buy_price > 0 and close_price > 0:
                    gross_return = (close_price - buy_price) / buy_price * 100
                    total_pnl += gross_return - self.config.transaction_cost
            self._realized_pnl += total_pnl
            n_pending = len(self._pending_buys)
            self._pending_buys = []
            self._today_bought = []  # 当日买入已全部强平，同步清空（保持 total_position = 底仓）
            # 小惩罚：鼓励日内配对卖出（SELL）实现做T，而非拖到收盘
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