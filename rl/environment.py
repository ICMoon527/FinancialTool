# -*- coding: utf-8 -*-
"""分时做T RL 环境

核心约束：
- 底仓管理：初始 3 份底仓，尾盘强制恢复为 3 份（卖出的底仓按收盘价买回补齐）
- 动作空间：0=HOLD, 1~3=BUY1/2/3, 4~6=SELL1/2/3（一次可买卖多份；当日累计买入 ≤ 3 份，卖出 ≤ 底仓）
- T+1 规则：当天买入的份额不可当天卖出，SELL1/2/3 只卖底仓（先卖后买），尾盘按收盘价买回补齐 3 份
- 未配对的当日买入尾盘强制平仓结算，仅保留 3 份底仓（维持现金流，防止一直买入/卖出）
- 预热期：前 warmup_steps 步强制 HOLD + reward=0
- 每个交易日为一个独立 episode

奖励机制（密集浮盈信号，r = ΔPnL - cost - λ·开仓敞口² + terminal_constraint）：
- ΔPnL：总持仓（底仓 + 当日买入）的逐分钟 mark-to-market 盈亏（% 刻度），
  让模型在每根K线都能感知持仓浮盈变化，而非仅事件驱动
- cost ：每笔成交的单边交易成本（佣金+滑点，每份 0.1%），含尾盘强平/买回成本
- λ·开仓敞口²：开仓惩罚（λ=reward_lambda），仅对本步「净新增」的日内敞口一次性收取
  （库存=agent可控的日内双向敞口=当日买入+先卖后买，不含恒定底仓）。改为开仓时收取而非
  逐steps持仓，避免「持有多日寸」在约240步/天的高频累加下被罚成最大负项、淹没做T信号。
- terminal_constraint：收盘仍有未平当日买入时惩罚 -κ·leftover²（κ=reward_terminal_coef），
  鼓励日内 SELL 配对平仓而非拖到收盘强平

性能簿记（不参与 reward）：_realized_pnl 仍按真实 T+1 账目精确结算当日做T已实现收益，
供训练验证与评估输出真实绩效数字。
"""

from __future__ import annotations

import logging
from pathlib import Path
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

        # 加载分时做T的 YAML 配置（指标阈值/周期 + 规则买卖点权重），
        # 与前端「分时做T」页面/策略共用同一份文件，使用户调整后训练侧同步生效；
        # 文件缺失或解析失败时为 None，下游组件各自回退类默认值。
        signal_config = self._load_signal_config()

        # 指标计算（阈值/周期取自 YAML，回退 IntradayIndicatorEngine 类默认值）
        self._indicator_engine = indicator_engine or IntradayIndicatorEngine(signal_config)
        # max_window=500：容纳「前日全天(约240根) + 当日全天(约240根)」的跨日K线拼接，
        # 保证开盘时 return_1/5/15/60、波动率等特征能看到完整前日上下文，
        # 且盘中不因裁剪挤掉前日数据导致中期特征断档
        self._data_buffer = IntradayDataBuffer(max_window=500)

        # 规则买卖点评分器（use_signal_scores=True 时为状态特征提供人工先验，权重取自 YAML）
        self._signal_evaluator = SignalEvaluator(signal_config)
        # 买卖分归一化基准：按当前权重满分求和，使特征落在 0~1，兼容用户对权重的任意调整。
        # 环境不传参考线、引力场恒为 0，故满分即权重之和。
        self._signal_buy_max = sum(self._signal_evaluator.BUY_WEIGHTS.values()) or 1.0
        self._signal_sell_max = sum(self._signal_evaluator.SELL_WEIGHTS.values()) or 1.0
        # 训练期信号统计（记录信号产生过程与有效特征分布，训练结束导出 signal_report）
        self._signal_stats = self._new_signal_stats()

        # 环境状态
        self._step: int = 0
        self._is_warmup: bool = False
        self._done: bool = False

        # 持仓状态
        self._base_position: int = 0          # 底仓（可卖出）
        self._today_bought: List[float] = []  # 当日买入的每笔成本
        self._realized_pnl: float = 0.0       # 已实现盈亏
        self._total_reward: float = 0.0       # episode 累计 reward

        # 当日价格统计
        self._day_open: float = 0.0
        self._day_high: float = 0.0
        self._day_low: float = float("inf")
        self._prev_close: float = 0.0
        # 形态窗口的固定归一化基准（前日收盘价，reset 时锁定）
        self._prev_close_ref: float = 0.0

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

    def _load_signal_config(self) -> Optional[Dict]:
        """加载分时做T的 YAML 配置（指标阈值/周期 + 规则买卖点权重）

        与分时做T页面/策略共用 watchdog/strategies/intraday_t0_config.yaml，
        用户在前端调整指标阈值或买卖规则权重后，训练侧同步生效；
        文件缺失或解析失败时返回 None，下游组件各自回退类默认值（保持旧行为）。
        """
        cfg_path = (
            Path(__file__).resolve().parent.parent
            / "watchdog"
            / "strategies"
            / "intraday_t0_config.yaml"
        )
        if cfg_path.exists():
            try:
                import yaml

                with open(cfg_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"加载分时做T信号配置失败，回退默认值: {e}")
        return None

    # ── 训练期信号统计 ──

    # 归一化特征直方图桶数（[0,1) 均分，末桶含 1.0）
    _FEAT_HIST_BUCKETS: int = 10

    def _new_signal_stats(self) -> Dict:
        """初始化信号统计累加器（跨 episode 累积，训练结束导出）"""
        return {
            "steps": 0,                 # 参与统计的步数（含无信号步）
            "absorption_gate": 0,       # 吸筹门槛通过次数（买点必备条件）
            "distribution_gate": 0,     # 出货门槛通过次数（卖点必备条件）
            "buy_hits": 0,              # 买分 > 0 的步数
            "sell_hits": 0,             # 卖分 > 0 的步数
            "buy_score_sum": 0.0,
            "buy_score_max": 0.0,
            "sell_score_sum": 0.0,
            "sell_score_max": 0.0,
            "buy_feat_sum": 0.0,
            "buy_feat_max": 0.0,
            "sell_feat_sum": 0.0,
            "sell_feat_max": 0.0,
            # 归一化特征直方图（只统计有效步，即分数>0）
            "buy_feat_hist": [0] * self._FEAT_HIST_BUCKETS,
            "sell_feat_hist": [0] * self._FEAT_HIST_BUCKETS,
            # 各规则触发次数 / 累计贡献分（仅 triggered 且 score>0，即有效特征）
            "buy_rule_counts": {},
            "buy_rule_scores": {},
            "sell_rule_counts": {},
            "sell_rule_scores": {},
        }

    def _accumulate_signal_stats(
        self,
        buy_score: float,
        sell_score: float,
        buy_details: List[Dict],
        sell_details: List[Dict],
        ind: "IndicatorSnapshot",
    ) -> None:
        """累加单步信号统计（信号怎么产生 + 有效特征数值分布）"""
        s = self._signal_stats
        s["steps"] += 1

        if ind.absorption_active:
            s["absorption_gate"] += 1
        if ind.distribution_active:
            s["distribution_gate"] += 1

        if buy_score > 0:
            s["buy_hits"] += 1
            s["buy_score_sum"] += buy_score
            s["buy_score_max"] = max(s["buy_score_max"], buy_score)
            feat = buy_score / self._signal_buy_max
            s["buy_feat_sum"] += feat
            s["buy_feat_max"] = max(s["buy_feat_max"], feat)
            s["buy_feat_hist"][min(int(feat * self._FEAT_HIST_BUCKETS), self._FEAT_HIST_BUCKETS - 1)] += 1
        if sell_score > 0:
            s["sell_hits"] += 1
            s["sell_score_sum"] += sell_score
            s["sell_score_max"] = max(s["sell_score_max"], sell_score)
            feat = sell_score / self._signal_sell_max
            s["sell_feat_sum"] += feat
            s["sell_feat_max"] = max(s["sell_feat_max"], feat)
            s["sell_feat_hist"][min(int(feat * self._FEAT_HIST_BUCKETS), self._FEAT_HIST_BUCKETS - 1)] += 1

        for detail in buy_details:
            if detail.get("triggered") and detail.get("score", 0) > 0:
                key = detail["key"]
                s["buy_rule_counts"][key] = s["buy_rule_counts"].get(key, 0) + 1
                s["buy_rule_scores"][key] = s["buy_rule_scores"].get(key, 0.0) + float(detail["score"])
        for detail in sell_details:
            if detail.get("triggered") and detail.get("score", 0) > 0:
                key = detail["key"]
                s["sell_rule_counts"][key] = s["sell_rule_counts"].get(key, 0) + 1
                s["sell_rule_scores"][key] = s["sell_rule_scores"].get(key, 0.0) + float(detail["score"])

    def get_signal_report(self) -> Optional[Dict]:
        """导出训练期规则信号统计（未启用先验买卖点时为 None）

        记录信号「怎么产生」：门槛通过次数、各规则触发次数与累计贡献分、
        买卖分与归一化特征的数值分布（均值/最大/直方图）。仅统计实际产生分数
        （triggered 且 score>0）的规则，即「产生效果的特征」。
        """
        if not self.config.use_signal_scores:
            return None
        s = self._signal_stats

        def _dist(hits: int, score_sum: float, score_max: float,
                  feat_sum: float, feat_max: float, hist: List[int]) -> Dict:
            return {
                "hits": hits,
                "score_mean": round(score_sum / hits, 6) if hits else 0.0,
                "score_max": round(score_max, 6),
                "feat_mean": round(feat_sum / hits, 6) if hits else 0.0,
                "feat_max": round(feat_max, 6),
                "feat_hist": hist,
            }

        def _nonzero(weights: Dict[str, float]) -> Dict[str, float]:
            return {k: v for k, v in weights.items() if v}

        def _sort_desc(counts: Dict[str, int], scores: Dict[str, float]) -> List[Dict]:
            rows = [
                {"rule": k, "count": v, "score_sum": round(scores.get(k, 0.0), 4)}
                for k, v in counts.items()
            ]
            return sorted(rows, key=lambda r: r["score_sum"], reverse=True)

        return {
            "enabled": True,
            "steps": s["steps"],
            "normalize_max": {"buy": self._signal_buy_max, "sell": self._signal_sell_max},
            "gate_pass": {
                "absorption_active": s["absorption_gate"],
                "distribution_active": s["distribution_gate"],
            },
            "buy": _dist(
                s["buy_hits"], s["buy_score_sum"], s["buy_score_max"],
                s["buy_feat_sum"], s["buy_feat_max"], s["buy_feat_hist"],
            ),
            "sell": _dist(
                s["sell_hits"], s["sell_score_sum"], s["sell_score_max"],
                s["sell_feat_sum"], s["sell_feat_max"], s["sell_feat_hist"],
            ),
            "buy_rules": _sort_desc(s["buy_rule_counts"], s["buy_rule_scores"]),
            "sell_rules": _sort_desc(s["sell_rule_counts"], s["sell_rule_scores"]),
            # 生效的规则权重（非零项），说明信号分由哪些规则构成
            "weights_used": {
                "buy": _nonzero(self._signal_evaluator.BUY_WEIGHTS),
                "sell": _nonzero(self._signal_evaluator.SELL_WEIGHTS),
            },
        }

    # ═══════════════════════════════════════════════
    #  公开接口
    # ═══════════════════════════════════════════════

    def reset(
        self,
        sample: Dict[str, object],
        prev_day_klines: Optional[List[Dict]] = None,
        prev_day_full_klines: Optional[List[Dict]] = None,
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
            prev_day_full_klines: 前一日全天分时K线（用于前日形态特征，
                use_prev_day_features=True 时生效；None 时前日特征补 0）

        Returns:
            state: np.ndarray 形状 (state_dim,)，初始状态向量
        """
        # 重置持仓状态
        self._base_position = self.MAX_BASE_POSITION
        self._today_bought = []
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

        # 预热指标：时间维度拼接前日K线（优先前日全天，兜底前日尾盘30根），
        # 使 return_1/5/15/60、波动率等特征在开盘时刻即包含前日上下文
        self._data_buffer = IntradayDataBuffer(max_window=500)
        self._warmup_bar_count = 0
        # 锁定前日收盘价作为形态窗口的归一化基准：
        # 旧实现按「窗口长度是否覆盖预热段」动态选取基准（len(df)=60 < 240 时
        # 回退为窗口首根收盘），导致基准随窗口滚动每步变化、同一根K线数值不稳定，
        # CNN 看到的是漂移的分布。锁定后整个交易日基准恒定。
        self._prev_close_ref = 0.0
        warmup_klines = prev_day_full_klines or prev_day_klines
        if warmup_klines:
            self._data_buffer.warmup(warmup_klines)
            self._warmup_bar_count = len(warmup_klines)
            latest = self._data_buffer.get_latest_price()
            self._prev_close_ref = float(latest) if latest else 0.0
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
        # 记录本步动作前敞口（用于开仓惩罚：只罚「新开仓」增量，不再逐step罚持仓）
        exp_before = len(self._today_bought) + len(self._sold_short)
        # R2：记录动作前已实现收益，用于计算本步「已实现收益增量」reward
        realized_before = self._realized_pnl

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

        # 3.5 做空日内硬止损：先卖后买做空若遇反弹超阈值，立即买回封死亏损，
        # 切断「先弱后强」下收盘强平造成单笔 -8% 灾难的尾部风险
        self._check_short_stop()

        # 4. 终态约束（奖励 r 的 terminal_constraint 项）与收盘强平簿记
        terminal_constraint = 0.0
        if self._step >= len(self._klines) or self._step >= self.MAX_STEPS:
            self._done = True
            # 先捕获未平当日买入数（_force_close 会清空），再强平结算
            leftover = len(self._today_bought)
            terminal_constraint = -self.config.reward_terminal_coef * leftover**2
            self._force_close()

        # 5. 奖励计算：r = Δrealized_pnl - λ·Δexposure² + terminal_constraint
        #    R2 修复（对齐评估口径，消除 reward hacking）：
        #    旧实现 r = mtm_delta - step_cost - inv_penalty，其中 mtm_delta 是
        #    「单根K线未实现浮动盈亏」。模型只需「买入后持有一阵」，未平仓的浮动
        #    就能刷正 reward（实测 16 笔交易全部集中在 600036、每天固定 1 笔，
        #    reward 累计 -75.65 而 realized -7.94% —— 训练目标与真实绩效严重错位），
        #    而真实做T收益要到配对/强平时才按 realized_pnl 结算（并扣双边成本 0.4%）。
        #    现改为：reward = 当日已实现收益增量（_realized_pnl 本步增量）。只有
        #    「真实平仓赚钱」才给正 reward；成本已在 _execute_action/_force_close
        #    结算时计入 realized（transaction_cost），与评估口径完全一致，杜绝刷浮动。
        cur_realized = self._realized_pnl
        realized_delta = cur_realized - realized_before
        # 开仓惩罚：只对本步净新增的日内敞口一次性收取 λ·Δexposure²，
        # 惩罚过度开仓/换手，不再对「持仓过程」逐 step 收取（逐step罚会把
        # 「持有日内仓位」（做T盈利前提）罚成最大负项，淹没一切正信号）。
        intraday_exposure = len(self._today_bought) + len(self._sold_short)
        open_exposure = max(0.0, intraday_exposure - exp_before)
        inv_penalty = self.config.reward_lambda * open_exposure**2

        reward = realized_delta - inv_penalty + terminal_constraint
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
        """构建状态向量（基础 8 维；use_signal_scores 时 10 维），所有特征股票无关

        组成：OHLCV(5, 相对前收的百分比 / 相对均量倍数) + 时间编码(1: 距收盘剩余比例)
              + 仓位状态(2: 底仓比例/有符号净敞口) [+ 规则买卖点得分(2)]

        设计分工：市场时序信息（动量、波动率、形态）全部由 CNN 编码器承担，
        state 只保留 CNN 看不到的「账户状态」与决策紧迫度。已移除：
        绝对价格 5 维（泄漏个股身份）、return/波动率 5 维（与 CNN 重复）、
        总持仓与平均成本 2 维（与其他维线性冗余）、时间 sin/cos 2 维（与剩余时间同源）、
        浮盈 1 维（sunk cost，不 gate 动作也不参与 reward）。
        """
        k = self._current_kline
        features = []

        # 归一化基准：前日收盘价（reset 时固定、全程不漂移，与 CNN 通道同一基准）
        ref = float(self._prev_close_ref) if self._prev_close_ref > 0 else 0.0
        if ref <= 0 and k:
            ref = float(k.get("Open") or 0.0)

        # ── OHLCV (5维)：全部改为股票无关的相对量 ──
        # 修复要点：此前为 `价格 / 100.0`，除以常数并非归一化——不同股票价格
        # 从 3 元到 1700 元相差数百倍，这 4 维等于给模型一条「个股身份证」通道，
        # 使其靠记忆个股而非学习通用日内形态（B 方案 35 个验证点全负的机制解释）。
        # 现统一以「前日收盘价」为参照取百分比偏离，任意股票的输入分布统计一致；
        # 成交量同理由绝对规模改为「相对 20 根均量」的倍数。
        if k and ref > 0:
            vol_ma = 0.0
            buf = self._data_buffer.data
            if len(buf) >= 1:
                vols = buf["Volume"].tail(20).tolist()
                vol_ma = float(np.mean(vols)) if vols else 0.0
            vol_ratio = (float(k["Volume"]) / vol_ma) if vol_ma > 0 else 1.0
            features.extend([
                (k["Open"] / ref - 1.0) * 100.0,
                (k["High"] / ref - 1.0) * 100.0,
                (k["Low"] / ref - 1.0) * 100.0,
                (k["Close"] / ref - 1.0) * 100.0,
                float(np.clip(vol_ratio, 0.0, 10.0)),  # 相对均量倍数，裁剪抑制异常放量
            ])
        else:
            features.extend([0.0] * 5)

        # 已删除「多尺度 return(4维) + 波动率(1维)」：
        # 本架构已启用 CNN 编码器（消费 480×8 的前日+当日K线序列），动量与波动率
        # 均能由 CNN 从原始序列自行推导，显式放进 state 属重复表达，徒增参数与
        # 过拟合风险。市场时序信息全部交由 CNN 承担，state 只保留账户状态。

        # ── 时间编码 (1维) ──
        # 只保留「距收盘剩余比例」：原 sin(2πt)/cos(2πt) 与本维承载同一信息
        # （由 bars_remaining 可反解 t，进而唯一确定 sin/cos），属完全冗余。
        # 且周期编码的核心价值在于消除跨周期边界跳变，而 episode 恰为单个交易日、
        # 无边界跳变，价值不成立；CNN 的 intraday_pos 通道亦已编码时间位置。
        bar_index = self._step
        max_steps = max(len(self._klines), self.MAX_STEPS)
        features.append((max_steps - bar_index) / max_steps)  # bars_remaining

        # ── 仓位状态 (2维) ──
        # state 的职责是「账户状态」：CNN 只能看到市场行情，看不到持仓与盈亏。
        # 第 7 维：底仓比例（决定 SELL 可用额度）；第 8 维：有符号净敞口
        # （正=当日买入待平，负=先卖后买待买回，决定 BUY 额度与尾盘买回义务）。
        # 由这两维可反解当日买入数 today_bought 与做空数 sold_short
        # （sold_short = 3·(1-base_ratio)，today_bought = 3·net_ratio + sold_short），
        # 账户状态信息完整。
        # 已删除「浮盈」维：属 sunk cost（最优择时只看未来价格），不 gate 任何动作、
        # 不参与 reward（R2 纯已实现），且环境唯一硬止损 short_stop 基于卖出价而非成本。
        net_exposure = len(self._today_bought) - len(self._sold_short)
        features.extend([
            self._base_position / self.MAX_BASE_POSITION,
            net_exposure / self.MAX_TODAY_BUY,
        ])

        # ── 规则买卖点得分 (2维，use_signal_scores=True 时启用) ──
        # 直接复用分时做T页面同款 SignalEvaluator 规则引擎的原始得分（权重取自
        # intraday_t0_config.yaml，随前端调整自动变化），作为人工先验注入状态；
        # 引力场部分不参与（环境无参考线数据）。
        # 除以「当前权重满分」归一化到 0~1，避免用户改权重后特征量纲漂移
        if self.config.use_signal_scores:
            buy_score = sell_score = 0.0
            ind = self._current_indicator
            if ind is not None:
                # 取完整返回值以拿到 weight_details（各规则触发/贡献），供 signal_report 记录
                buy_score, _, _, _, buy_details = self._signal_evaluator.evaluate_buy(ind)
                sell_score, _, _, _, sell_details = self._signal_evaluator.evaluate_sell(ind)
                buy_score = float(buy_score)
                sell_score = float(sell_score)
                self._accumulate_signal_stats(
                    buy_score, sell_score, buy_details, sell_details, ind
                )
            features.extend(
                [buy_score / self._signal_buy_max, sell_score / self._signal_sell_max]
            )

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
            if n > self._base_position:
                return False, self.HOLD
            # 空头尾部风险抑制：只在「当日已回落跌破开盘价」的弱势环境允许卖底仓做空，
            # 逢强势（price >= open）一律禁空。基线证实开盘价附近做空遇暴力拉升会被
            # 收盘强平买回造成单笔 -8% 灾难损失，故必须要求当日已确认弱势才可做空。
            if self.config.short_guard_enabled and self._day_open > 0:
                cur = self._current_kline.get("Close", 0.0) if self._current_kline else 0.0
                up_pct = (cur / self._day_open - 1.0) * 100.0
                if up_pct >= -self.config.short_down_margin:
                    return False, self.HOLD
            return True, action
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

    def _check_short_stop(self) -> None:
        """做空日内硬止损：对每笔先卖后买做空，若当前价格相对卖出价反弹幅度
        超过 short_stop(%)，立即按现价买回、结算利润/亏损并从 _sold_short 移除，
        同时恢复一份底仓。这从根本上把单笔做空的最大亏损封在 short_stop 附近，
        防止「先弱后强」的暴力拉升日被收盘强平买回造成单笔 -8% 的灾难性损失。

        注：动量守卫（short_guard_enabled）只挡得住「逢强势做空」，拦不住
        「盘中短暂跌破开盘价做空、随后暴涨」的漏网，硬止损是兜底机制。
        """
        if self.config.short_stop <= 0.0:
            return
        k = self._current_kline
        if not k:
            return
        cur_price = k["Close"]
        if cur_price <= 0.0 or not self._sold_short:
            return
        remaining: List[float] = []
        for sell_price in self._sold_short:
            if sell_price > 0.0 and (cur_price / sell_price - 1.0) * 100.0 > self.config.short_stop:
                # 止损买回：先卖后买，卖出价低于买回价则亏损，计入已实现做T收益
                gross_return = (sell_price - cur_price) / cur_price * 100 - self.config.transaction_cost
                self._realized_pnl += gross_return
                # reward cost 项：买回单边成本
                self._step_cost += self.config.per_side_cost
                # 已买回一份底仓 → 底仓恢复 1 份
                self._base_position += 1
                self._trades.append({
                    "time": k.get("time", k.get("datetime", "")),
                    "action": "SHORT_STOP",
                    "price": cur_price,
                    "pnl": gross_return,
                })
            else:
                remaining.append(sell_price)
        self._sold_short = remaining

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

    @property
    def kline_window(self) -> np.ndarray:
        """时间序列拼接的形态窗口，形状 (cnn_window, cnn_in_channels)

        语义：把「前日全天K线」与「当日已走过的K线」在时间轴上首尾相接，
        形成一条连续序列供 1D-CNN 读取（注意：这是时间维度的拼接，不是把前日
        特征拼进 8 维 state 向量；state 维度始终不变）。

        通道定义（8 通道）：
        - 0~3：OHLC 相对前日收盘价的收益率
        - 4  ：成交量相对窗口内均量的偏离比
        - 5  ：day_flag（0=前日，1=当日）——让 CNN 分辨该根属于哪一天
        - 6  ：intraday_pos（日内归一化位置 0~1）——让 CNN 锚定开盘/尾盘
        - 7  ：valid_mask（1=真实K线，0=补零）——避免 padding 被当成行情

        归一化基准固定为 reset 时锁定的前日收盘价，整个交易日不随窗口滚动变化。
        数据左对齐排列（前日固定占前置段），历史不足 cnn_window 根时在末尾补 0。
        """
        W = self.config.cnn_window
        C = self.config.cnn_in_channels
        window = np.zeros((W, C), dtype=np.float32)

        df_all = self._data_buffer.data
        if len(df_all) == 0:
            return window

        # 取最近 W 根；记录全局起始下标以便判定「前日 / 当日」归属
        start = max(0, len(df_all) - W)
        df = df_all.iloc[start:]
        n = len(df)

        pc = self._prev_close_ref
        if pc <= 0:
            pc = float(df["Close"].iloc[0])
        ohlc = df[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float64)
        vol = df["Volume"].to_numpy(dtype=np.float64)
        mean_vol = float(vol.mean()) if vol.size else 0.0

        window[:n, 0:4] = (ohlc / pc - 1.0).astype(np.float32)
        window[:n, 4] = (vol / (mean_vol + 1e-8) - 1.0).astype(np.float32)

        # 会话标记：全局下标 < 预热根数 视为前日，其余为当日
        warm = max(1, self._warmup_bar_count)
        global_idx = np.arange(start, start + n)
        is_today = global_idx >= self._warmup_bar_count
        window[:n, 5] = is_today.astype(np.float32)

        pos = np.where(
            is_today,
            (global_idx - self._warmup_bar_count + 1) / float(self.MAX_STEPS),
            (global_idx + 1) / float(warm),
        )
        window[:n, 6] = np.clip(pos, 0.0, 1.0).astype(np.float32)
        window[:n, 7] = 1.0
        return window

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
            # 与策略侧口径一致：两份镜像公式分别产出吸筹(≥0)与出货(≤0)，各自已在
            # calc_absorption/calc_distribution 内按 YAML 阈值过滤掉无效信号；
            # 再合并为单一有符号值，符号决定方向：
            #   >0 → 吸筹买入信号有效；<0 → 出货卖出信号有效；==0 → 无有效信号。
            # 因此二者严格互斥（对应策略 line 624 合并 + 1214-1215 取符号）。
            abs_data = self._indicator_engine.calc_absorption(data)
            dist_data = self._indicator_engine.calc_distribution(data)

            absorption_raw = (
                float(abs_data["absorption"].iloc[-1])
                if "absorption" in abs_data.columns
                else 0.0
            )
            distribution_raw = (
                float(dist_data["distribution"].iloc[-1])
                if "distribution" in dist_data.columns
                else 0.0
            )
            merged = absorption_raw + distribution_raw
            ind.absorption_value = merged
            ind.absorption_active = merged > 0
            ind.distribution_active = merged < 0

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