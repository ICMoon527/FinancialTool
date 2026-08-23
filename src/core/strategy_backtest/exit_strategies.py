# -*- coding: utf-8 -*-
"""
可插拔退出策略模块。

提供退出策略的抽象基类、注册表和内置实现：
- SimpleExitStrategy: 固定比例止盈止损
- TieredExitStrategy: 动态分级止盈（默认模式 + 强势模式）
- TiandaoJinniuExitStrategy: 天道金牛清仓（策略A）
- TiandaoBbiRollingExitStrategy: 天道BBI滚动（策略B）
- TiandaoJinniu2ProtectionExitStrategy: 天道金牛2保护（策略C）
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# 全局注册表
_exit_strategy_registry: Dict[str, Type["ExitStrategy"]] = {}


@dataclass
class ExitSignal:
    """退出信号"""
    stock_code: str
    reason: str                             # 退出原因
    exit_type: str = "full"                 # "full" | "partial" | "rebuy"
    exit_ratio: float = 1.0                 # 卖出比例 (1.0=全部, 0.5=一半)
    sell_price: float = 0.0
    price_type: str = "close"               # "open" | "intraday" | "close"


def register_exit_strategy(name: str):
    """注册退出策略的装饰器"""

    def decorator(cls: Type["ExitStrategy"]) -> Type["ExitStrategy"]:
        _exit_strategy_registry[name] = cls
        logger.debug("注册退出策略: %s -> %s", name, cls.__name__)
        return cls

    return decorator


class ExitStrategy(ABC):
    """退出策略抽象基类"""

    # 子类覆盖：前端显示名称
    display_name: str = ""
    # 子类覆盖：策略描述
    description: str = ""

    @abstractmethod
    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        """买入时初始化持仓状态"""
        ...

    @abstractmethod
    def check_exits(
        self,
        current_date: date,
        positions: Dict[str, Any],
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
        phase: str,
    ) -> List[ExitSignal]:
        """
        检查退出信号。

        Args:
            current_date: 当前交易日
            positions: 当前持仓 {stock_code: Position}
            price_provider: 价格获取函数 (stock_code, price_type) -> float|None
            trading_dates: 交易日列表
            current_date_index: 当前日期在 trading_dates 中的索引
            phase: 检查阶段 "open" | "intraday" | "close"

        Returns:
            退出信号列表
        """
        ...

    @abstractmethod
    def cleanup_position(self, stock_code: str) -> None:
        """卖出后清理持仓状态"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置所有状态"""
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """校验配置参数合法性"""
        ...

    @classmethod
    def get_registry(cls) -> Dict[str, Type["ExitStrategy"]]:
        """获取所有已注册的退出策略"""
        return _exit_strategy_registry.copy()

    @classmethod
    def create(cls, strategy_name: str, params: Optional[Dict[str, Any]] = None) -> "ExitStrategy":
        """
        工厂方法：根据名称和参数创建退出策略实例。

        Args:
            strategy_name: 策略名称（注册表中的 key）
            params: 策略参数

        Returns:
            ExitStrategy 实例
        """
        if params is None:
            params = {}
        strategy_cls = _exit_strategy_registry.get(strategy_name)
        if strategy_cls is None:
            logger.warning("未找到退出策略 '%s'，使用 SimpleExitStrategy 作为默认", strategy_name)
            strategy_cls = SimpleExitStrategy
        return strategy_cls(**params)


# ============================================================
# 内置策略实现
# ============================================================


@register_exit_strategy("simple")
class SimpleExitStrategy(ExitStrategy):
    """固定比例止盈止损"""

    display_name = "固定止盈止损"
    description = "设定固定止损/止盈比例，触及即卖出"

    def __init__(
        self,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_holding_days: int = 5,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self._entry_dates: Dict[str, date] = {}
        self._stop_loss_prices: Dict[str, float] = {}
        self._take_profit_prices: Dict[str, float] = {}

    def validate_config(self) -> bool:
        if self.stop_loss_pct is not None and (self.stop_loss_pct <= 0 or self.stop_loss_pct >= 1):
            logger.error("止损比例必须在 0-1 之间，当前值: %s", self.stop_loss_pct)
            return False
        if self.take_profit_pct is not None and (self.take_profit_pct <= 0 or self.take_profit_pct >= 1):
            logger.error("止盈比例必须在 0-1 之间，当前值: %s", self.take_profit_pct)
            return False
        if self.max_holding_days < 1:
            logger.error("最长持股天数必须 >= 1，当前值: %s", self.max_holding_days)
            return False
        return True

    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        self._entry_dates[stock_code] = entry_date
        if self.stop_loss_pct is not None:
            self._stop_loss_prices[stock_code] = entry_price * (1 - self.stop_loss_pct)
        if self.take_profit_pct is not None:
            self._take_profit_prices[stock_code] = entry_price * (1 + self.take_profit_pct)

    def check_exits(
        self,
        current_date: date,
        positions: Dict[str, Any],
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
        phase: str,
    ) -> List[ExitSignal]:
        signals: List[ExitSignal] = []

        for stock_code in list(positions.keys()):
            # T+1 规则：买入当天不能卖出
            if stock_code in self._entry_dates and self._entry_dates[stock_code] == current_date:
                continue

            if phase == "open":
                signals.extend(self._check_open(stock_code, price_provider))
            elif phase == "intraday":
                signals.extend(self._check_intraday(stock_code, price_provider))
            elif phase == "close":
                signals.extend(self._check_close(stock_code, price_provider, trading_dates, current_date_index))

        return signals

    def _check_open(
        self, stock_code: str, price_provider: Callable[[str, str], Optional[float]]
    ) -> List[ExitSignal]:
        """检查开盘价是否触发止盈止损"""
        price = price_provider(stock_code, "open")
        if price is None:
            return []

        # 止损检查
        if stock_code in self._stop_loss_prices and price <= self._stop_loss_prices[stock_code]:
            logger.info(
                "触发止损 (开盘价): %s 当前价 %.2f <= 止损价 %.2f",
                stock_code, price, self._stop_loss_prices[stock_code],
            )
            return [ExitSignal(
                stock_code=stock_code,
                reason="stop_loss",
                sell_price=price,
                price_type="open",
            )]

        # 止盈检查
        if stock_code in self._take_profit_prices and price >= self._take_profit_prices[stock_code]:
            logger.info(
                "触发止盈 (开盘价): %s 当前价 %.2f >= 止盈价 %.2f",
                stock_code, price, self._take_profit_prices[stock_code],
            )
            return [ExitSignal(
                stock_code=stock_code,
                reason="take_profit",
                sell_price=price,
                price_type="open",
            )]

        return []

    def _check_intraday(
        self, stock_code: str, price_provider: Callable[[str, str], Optional[float]]
    ) -> List[ExitSignal]:
        """检查盘中最高/最低价是否触发止盈止损"""
        high_price = price_provider(stock_code, "high")
        low_price = price_provider(stock_code, "low")
        if high_price is None or low_price is None:
            return []

        # 止损：最低价触发
        if stock_code in self._stop_loss_prices and low_price <= self._stop_loss_prices[stock_code]:
            logger.info(
                "触发盘中止损: %s 最低价 %.2f <= 止损价 %.2f",
                stock_code, low_price, self._stop_loss_prices[stock_code],
            )
            return [ExitSignal(
                stock_code=stock_code,
                reason="stop_loss_intraday",
                sell_price=self._stop_loss_prices[stock_code],
                price_type="intraday",
            )]

        # 止盈：最高价触发
        if stock_code in self._take_profit_prices and high_price >= self._take_profit_prices[stock_code]:
            logger.info(
                "触发盘中止盈: %s 最高价 %.2f >= 止盈价 %.2f",
                stock_code, high_price, self._take_profit_prices[stock_code],
            )
            return [ExitSignal(
                stock_code=stock_code,
                reason="take_profit_intraday",
                sell_price=self._take_profit_prices[stock_code],
                price_type="intraday",
            )]

        return []

    def _check_close(
        self,
        stock_code: str,
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
    ) -> List[ExitSignal]:
        """检查持股超时"""
        if stock_code not in self._entry_dates:
            return []

        entry_date = self._entry_dates[stock_code]
        try:
            entry_index = trading_dates.index(entry_date)
            holding_days = current_date_index - entry_index

            if holding_days >= self.max_holding_days:
                close_price = price_provider(stock_code, "close")
                if close_price is None:
                    return []
                logger.info(
                    "触发持股超时: %s 已持股 %d 个交易日 (最长: %d 天)",
                    stock_code, holding_days, self.max_holding_days,
                )
                return [ExitSignal(
                    stock_code=stock_code,
                    reason="hold_timeout",
                    sell_price=close_price,
                    price_type="close",
                )]
        except ValueError:
            logger.warning("股票 %s 的买入日期 %s 不在交易日列表中", stock_code, entry_date)

        return []

    def cleanup_position(self, stock_code: str) -> None:
        self._entry_dates.pop(stock_code, None)
        self._stop_loss_prices.pop(stock_code, None)
        self._take_profit_prices.pop(stock_code, None)

    def reset(self) -> None:
        self._entry_dates.clear()
        self._stop_loss_prices.clear()
        self._take_profit_prices.clear()


# ============================================================
# 动态分级止盈策略
# ============================================================


@dataclass
class _TieredPositionState:
    """分级止盈策略的持仓状态"""
    entry_date: date
    entry_price: float
    peak_price: float = 0.0                    # 持仓期间最高价
    last_trailing_stop: float = 0.0            # 上一次计算的移动止盈价，保证只上移不下移


@register_exit_strategy("tiered")
class TieredExitStrategy(ExitStrategy):
    """动态分级止盈策略。

    止损：
      - 买入价 × (1 - stop_loss_pct)，默认 12%
      - 收盘价跌破止损价 → 当日收盘价直接离场

    阶梯式移动止盈（基于持仓期间最高价）：
      - 盈利 < 30% 时，回撤容忍 10%
      - 盈利 >= 30% 时，回撤容忍 8%
      - 盈利 >= 50% 时，回撤容忍 5%
      - 当日收盘价触发移动止盈 → 当日收盘价直接离场

    动态时间止损：
      - 持股 20 天后，收益率 < 5% 则平仓
    """

    display_name = "动态分级止盈"
    description = "固定止损 + 阶梯式移动止盈 + 动态时间止损"

    def __init__(
        self,
        stop_loss_pct: float = 0.12,
        time_stop_days: int = 20,
        time_stop_min_return: float = 0.05,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days
        self.time_stop_min_return = time_stop_min_return
        self._states: Dict[str, _TieredPositionState] = {}
        self._date_to_index: Optional[Dict[date, int]] = None  # 交易日→索引缓存

    def _get_trailing_stop_pct(self, gain_pct: float) -> float:
        """根据当前盈利水平获取回撤容忍比例。"""
        if gain_pct >= 0.50:
            return 0.05
        elif gain_pct >= 0.30:
            return 0.08
        else:
            return 0.10

    def validate_config(self) -> bool:
        if self.stop_loss_pct <= 0 or self.stop_loss_pct >= 1:
            logger.error("stop_loss_pct 必须在 0-1 之间，当前值: %s", self.stop_loss_pct)
            return False
        if self.time_stop_days < 1:
            logger.error("time_stop_days 必须 >= 1")
            return False
        return True

    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        self._states[stock_code] = _TieredPositionState(
            entry_date=entry_date,
            entry_price=entry_price,
            peak_price=entry_price,
        )

    def check_exits(
        self,
        current_date: date,
        positions: Dict[str, Any],
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
        phase: str,
    ) -> List[ExitSignal]:
        signals: List[ExitSignal] = []

        # 缓存交易日→索引映射，避免每日 3 次重复构建
        if self._date_to_index is None:
            self._date_to_index = {d: i for i, d in enumerate(trading_dates)}
        date_to_index = self._date_to_index

        for stock_code in list(positions.keys()):
            if stock_code not in self._states:
                continue
            state = self._states[stock_code]

            # T+1 规则：买入当天不能卖出
            if state.entry_date == current_date:
                continue

            if phase == "open":
                # 开盘阶段无操作（止损/止盈均在收盘时直接执行）
                pass
            elif phase == "intraday":
                # 盘中更新峰值价格
                self._update_peak_intraday(stock_code, state, price_provider)
            elif phase == "close":
                s = self._check_close_tiered(
                    stock_code, state, price_provider, date_to_index, current_date_index
                )
                if s:
                    signals.append(s)

        return signals

    def _update_peak_intraday(
        self,
        stock_code: str,
        state: _TieredPositionState,
        price_provider: Callable[[str, str], Optional[float]],
    ) -> None:
        """盘中更新峰值价格（使用最高价）。"""
        high_price = price_provider(stock_code, "high")
        if high_price is not None and high_price > state.peak_price:
            state.peak_price = high_price

    def _check_close_tiered(
        self,
        stock_code: str,
        state: _TieredPositionState,
        price_provider: Callable[[str, str], Optional[float]],
        date_to_index: Dict[date, int],
        current_date_index: int,
    ) -> Optional[ExitSignal]:
        """检查收盘：更新峰值 + 止损 + 移动止盈 + 时间止损，全部以收盘价直接执行。"""
        close_price = price_provider(stock_code, "close")
        if close_price is None:
            return None

        # 即使 intraday 阶段未被调用，也获取 high 更新峰值，确保移动止盈线准确
        high_price = price_provider(stock_code, "high")
        if high_price is not None and high_price > state.peak_price:
            state.peak_price = high_price
        if close_price > state.peak_price:
            state.peak_price = close_price

        # 计算当前盈利和峰值曾达到的最大盈利
        gain_pct = (close_price - state.entry_price) / state.entry_price
        max_gain_pct = (state.peak_price - state.entry_price) / state.entry_price

        # 1. 止损检查：收盘价 < 止损价 → 当日收盘价直接离场
        stop_loss_price = state.entry_price * (1 - self.stop_loss_pct)
        if close_price <= stop_loss_price:
            if price_provider(stock_code, "is_limit_down_sealed") == 1.0:
                logger.info(
                    "分级止盈 [%s]: 收盘价 %.2f <= 止损价 %.2f，但跌停封死无法成交，继续持有",
                    stock_code, close_price, stop_loss_price,
                )
                return None
            logger.info(
                "分级止盈 [%s]: 收盘价 %.2f <= 止损价 %.2f，当日收盘价离场",
                stock_code, close_price, stop_loss_price,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason="止损出局",
                sell_price=close_price,
                price_type="close",
            )

        # 2. 阶梯式移动止盈检查
        # 基于峰值曾达到的最大盈利决定回撤容忍度，即使当前价格回落也锁定高等级保护
        if max_gain_pct > 0:
            trailing_dd = self._get_trailing_stop_pct(max_gain_pct)
            new_trailing_stop = state.peak_price * (1 - trailing_dd)

            # 移动止盈线只上移不下移（防御性编程，应对数据异常或未来逻辑变更）
            trailing_stop = max(new_trailing_stop, state.last_trailing_stop)
            state.last_trailing_stop = trailing_stop

            if close_price <= trailing_stop:
                if price_provider(stock_code, "is_limit_down_sealed") == 1.0:
                    logger.info(
                        "分级止盈 [%s]: 收盘价 %.2f <= 移动止盈价 %.2f，但跌停封死无法成交，继续持有",
                        stock_code, close_price, trailing_stop,
                    )
                    return None
                logger.info(
                    "分级止盈 [%s]: 收盘价 %.2f <= 移动止盈价 %.2f(峰值 %.2f, 峰值盈利 %.1f%%, 回撤 %.0f%%)，当日收盘价离场",
                    stock_code, close_price, trailing_stop, state.peak_price,
                    max_gain_pct * 100, trailing_dd * 100,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="移动止盈",
                    sell_price=close_price,
                    price_type="close",
                )

        # 3. 动态时间止损
        entry_index = date_to_index.get(state.entry_date)
        if entry_index is not None:
            holding_days = current_date_index - entry_index
            if holding_days >= self.time_stop_days and gain_pct < self.time_stop_min_return:
                logger.info(
                    "分级止盈 [%s]: 持股 %d 天 >= %d 天，收益率 %.1f%% < %.0f%%，时间止损平仓",
                    stock_code, holding_days, self.time_stop_days,
                    gain_pct * 100, self.time_stop_min_return * 100,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="时间止损",
                    sell_price=close_price,
                    price_type="close",
                )
        else:
            logger.warning(
                "分级止盈 [%s]: 买入日期 %s 不在交易日列表中，无法计算时间止损",
                stock_code, state.entry_date,
            )

        return None

    def cleanup_position(self, stock_code: str) -> None:
        self._states.pop(stock_code, None)

    def reset(self) -> None:
        self._states.clear()
        self._date_to_index = None


# ============================================================
# 天道指标数据快照
# ============================================================


@dataclass
class TiandaoIndicatorSnapshot:
    """天道指标快照（单日），供天道系列退出策略使用"""
    td_jinzuan: float = 0.0    # 金钻趋势（支撑位 / 通道下轨）
    td_jinniu: float = 0.0     # 金牛（压力位 / 通道上轨）
    td_jinniu2: float = 0.0    # 金牛2（慢速跟随线 / EMA(金钻趋势, 25)）
    td_bbi: float = 0.0        # BBI（多空分界线）
    pressure: float = 0.0      # REF(金牛, 1) — 昨日金牛值，用作压力位
    support: float = 0.0       # REF(金钻, 1) — 昨日金钻值，用作支撑位（仅监控）


# ============================================================
# 策略 D：通道压力线分批止盈（tiandao-pressure）
# ============================================================


@dataclass
class _TiandaoPressureState:
    """通道压力线分批止盈的持仓状态"""
    entry_date: date
    entry_price: float
    peak: float = 0.0
    last_trailing_stop: float = 0.0
    tier1_done: bool = False
    tier2_done: bool = False


@register_exit_strategy("tiandao_pressure")
class TiandaoPressureExitStrategy(ExitStrategy):
    """通道压力线分批止盈策略。

    在动态分级止盈的基础上，增加根据天道指标压力位的分批止盈：
      pressure = REF(金牛, 1)  — 昨日金牛值，作为压力位
      support  = REF(金钻, 1)  — 昨日金钻值，仅备用监控，不参与卖出

    卖出优先级（逐阶段检查，先触发先执行）：
      C > A/B > 7%止损 > 时间止损

    - A档：首次触碰压力位，以 pressure 价卖出 1/3
    - B档：从压力位回落，尾盘卖出剩余 1/2（即原仓 1/3）
    - C档：动态分级移动止盈，全程跟踪，对剩余仓位生效
    """

    display_name = "通道压力线分批止盈"
    description = "固定止损 + 通道压力线分批止盈(A/B/C档) + 动态时间止损"

    def __init__(
        self,
        stop_loss_pct: float = 0.07,
        time_stop_days: int = 30,
        time_stop_min_return: float = 0.0,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days
        self.time_stop_min_return = time_stop_min_return
        self._states: Dict[str, _TiandaoPressureState] = {}

        # ── 诊断统计字段 ──
        # 各卖出分支触发次数
        self._stop_loss_count = 0      # 7%止损
        self._time_stop_count = 0      # 时间止损
        self._tier1_count = 0          # A档压力位卖出1/3
        self._tier2_count = 0          # B档回落卖出1/2
        self._c_tier_count = 0         # C档移动止盈
        # 每笔完整交易的持仓天数
        self._holding_days: List[int] = []
        # peak 逐日轨迹（用于诊断 peak 初始化是否正确）: {stock_code: [(date, close, peak)]}
        self._daily_peak_log: Dict[str, List[tuple]] = {}
        # REF 检查采样: {stock_code: [(date, pressure)]}  pressure=REF(金牛,1)
        self._ref_check: Dict[str, List[tuple]] = {}
        # 记录持仓起始（买入）日期，用于分析 peak 是否从买入日开始
        self._entry_dates: Dict[str, date] = {}

    # ── 生命周期 ──

    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        self._states[stock_code] = _TiandaoPressureState(
            entry_date=entry_date,
            entry_price=entry_price,
            peak=entry_price,
        )
        self._entry_dates[stock_code] = entry_date

    def cleanup_position(self, stock_code: str) -> None:
        # 记录完整持仓周期（卖出时计算）
        state = self._states.get(stock_code)
        if state is not None and self._last_index is not None:
            try:
                idx = self._trading_dates.index(state.entry_date)
                holding = self._last_index - idx
                if holding >= 0:
                    self._holding_days.append(holding)
            except (ValueError, AttributeError):
                pass
        self._states.pop(stock_code, None)
        self._entry_dates.pop(stock_code, None)
        self._daily_peak_log.pop(stock_code, None)
        self._ref_check.pop(stock_code, None)

    def reset(self) -> None:
        self._states.clear()
        self._entry_dates.clear()
        self._daily_peak_log.clear()
        self._ref_check.clear()
        self._holding_days.clear()
        self._stop_loss_count = 0
        self._time_stop_count = 0
        self._tier1_count = 0
        self._tier2_count = 0
        self._c_tier_count = 0

    def validate_config(self) -> bool:
        if self.stop_loss_pct <= 0 or self.stop_loss_pct >= 1:
            logger.error("stop_loss_pct 必须在 0-1 之间，当前值: %s", self.stop_loss_pct)
            return False
        if self.time_stop_days < 1:
            logger.error("time_stop_days 必须 >= 1，当前值: %s", self.time_stop_days)
            return False
        return True

    # ── 核心逻辑 ──

    def check_exits(
        self,
        current_date: date,
        positions: Dict[str, Any],
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
        phase: str,
    ) -> List[ExitSignal]:
        if phase != "close":
            return []

        self._trading_dates = trading_dates
        self._last_index = current_date_index

        signals: List[ExitSignal] = []
        for stock_code in list(positions.keys()):
            state = self._states.get(stock_code)
            if state is None:
                continue

            # ── 获取价格数据 ──
            close_price = price_provider(stock_code, "close")
            high_price = price_provider(stock_code, "high")
            pressure = price_provider(stock_code, "pressure")
            if close_price is None:
                continue

            # ── 更新峰值（收盘价 + 最高价） ──
            if high_price is not None and high_price > state.peak:
                state.peak = high_price
            if close_price > state.peak:
                state.peak = close_price

            # ── 记录诊断轨迹 ──
            self._daily_peak_log.setdefault(stock_code, []).append(
                (current_date, close_price, state.peak)
            )
            # REF 检查采样：记录日期与 pressure（=REF(金牛,1)）
            # 注意：不依赖当日 td_jinniu 字段，因回测尾部数据缺口时该字段可能被算成 0，
            # 只依赖 pressure 序列本身做连续性与前日一致性校验
            self._ref_check.setdefault(stock_code, []).append(
                (current_date, pressure if pressure is not None else 0.0)
            )

            gain_pct = (close_price - state.entry_price) / state.entry_price
            max_gain_pct = (state.peak - state.entry_price) / state.entry_price

            # ── 优先级 C > A/B > 止损 > 时间止损 ──

            # --- C档：动态分级移动止盈（全程跟踪，无论 tier 状态） ---
            if max_gain_pct > 0:
                trailing_dd = self._get_trailing_stop_pct(max_gain_pct)
                new_trailing_stop = state.peak * (1 - trailing_dd)
                trailing_stop = max(new_trailing_stop, state.last_trailing_stop)
                state.last_trailing_stop = trailing_stop

                if close_price <= trailing_stop:
                    if price_provider(stock_code, "is_limit_down_sealed") == 1.0:
                        logger.info(
                            "通道压力线 [%s]: C档移动止盈触发，但跌停封死无法成交，继续持有",
                            stock_code,
                        )
                        continue
                    self._c_tier_count += 1
                    logger.info(
                        "通道压力线 [%s]: C档移动止盈，收盘价 %.2f <= 移动止盈价 %.2f(峰值 %.2f, 峰值盈利 %.1f%%)，全部卖出",
                        stock_code, close_price, trailing_stop, state.peak, max_gain_pct * 100,
                    )
                    signals.append(ExitSignal(
                        stock_code=stock_code,
                        reason="C档移动止盈",
                        exit_type="full",
                        exit_ratio=1.0,
                        sell_price=close_price,
                        price_type="close",
                    ))
                    continue  # C档已触发，跳过后续检查

            # --- A档：首次触碰压力位，卖 1/3 ---
            if not state.tier1_done and pressure is not None and pressure > 0 and high_price is not None and high_price >= pressure:
                if price_provider(stock_code, "is_limit_down_sealed") == 1.0:
                    logger.info(
                        "通道压力线 [%s]: A档压力位触发，但跌停封死无法成交，继续持有",
                        stock_code,
                    )
                    continue
                state.tier1_done = True
                self._tier1_count += 1
                logger.info(
                    "通道压力线 [%s]: A档触发，最高价 %.2f >= 压力位 %.2f，以压力位价格卖出 1/3",
                    stock_code, high_price, pressure,
                )
                signals.append(ExitSignal(
                    stock_code=stock_code,
                    reason="A档压力位卖出1/3",
                    exit_type="partial",
                    exit_ratio=1.0 / 3.0,
                    sell_price=pressure,
                    price_type="intraday",
                ))
                continue  # A档已触发，跳过B档

            # --- B档：突破后回落，再卖 1/2（即原仓的 1/3） ---
            if state.tier1_done and not state.tier2_done and pressure is not None and pressure > 0 and close_price < pressure:
                if price_provider(stock_code, "is_limit_down_sealed") == 1.0:
                    logger.info(
                        "通道压力线 [%s]: B档回落触发，但跌停封死无法成交，继续持有",
                        stock_code,
                    )
                    continue
                state.tier2_done = True
                self._tier2_count += 1
                logger.info(
                    "通道压力线 [%s]: B档触发，收盘价 %.2f < 压力位 %.2f，尾盘卖出剩余 1/2",
                    stock_code, close_price, pressure,
                )
                signals.append(ExitSignal(
                    stock_code=stock_code,
                    reason="B档压力位回落卖出1/2剩余",
                    exit_type="partial",
                    exit_ratio=0.5,
                    sell_price=close_price,
                    price_type="close",
                ))
                continue  # B档已触发，跳过止损

            # --- 止损检查（与基线完全一致） ---
            stop_loss_price = state.entry_price * (1 - self.stop_loss_pct)
            if close_price <= stop_loss_price:
                if price_provider(stock_code, "is_limit_down_sealed") == 1.0:
                    logger.info(
                        "通道压力线 [%s]: 止损触发，但跌停封死无法成交，继续持有",
                        stock_code,
                    )
                    continue
                self._stop_loss_count += 1
                logger.info(
                    "通道压力线 [%s]: 止损触发，收盘价 %.2f <= 止损价 %.2f(买入价 %.2f × %.0f%%)，全部卖出",
                    stock_code, close_price, stop_loss_price, state.entry_price, self.stop_loss_pct * 100,
                )
                signals.append(ExitSignal(
                    stock_code=stock_code,
                    reason="止损出局",
                    exit_type="full",
                    exit_ratio=1.0,
                    sell_price=close_price,
                    price_type="close",
                ))
                continue

            # --- 时间止损（与基线完全一致） ---
            entry_index = None
            for i, d in enumerate(trading_dates):
                if d == state.entry_date:
                    entry_index = i
                    break
            if entry_index is not None:
                holding_days = current_date_index - entry_index
                if holding_days >= self.time_stop_days and gain_pct <= self.time_stop_min_return:
                    self._time_stop_count += 1
                    logger.info(
                        "通道压力线 [%s]: 时间止损，持股 %d 天 >= %d 天，收益率 %.1f%% <= %.0f%%，全部卖出",
                        stock_code, holding_days, self.time_stop_days,
                        gain_pct * 100, self.time_stop_min_return * 100,
                    )
                    signals.append(ExitSignal(
                        stock_code=stock_code,
                        reason="时间止损",
                        exit_type="full",
                        exit_ratio=1.0,
                        sell_price=close_price,
                        price_type="close",
                    ))
                    continue

        return signals

    def _get_trailing_stop_pct(self, gain_pct: float) -> float:
        """阶梯式回撤容忍度（与 TieredExitStrategy 基线完全一致）。

        任何盈利 >0 时最低也有 10% 回撤容忍，避免峰值盈利不足 10%
        时移动止盈线等于当日最高价导致的"刚买就卖"。
        """
        if gain_pct >= 0.50:
            return 0.05
        elif gain_pct >= 0.30:
            return 0.08
        else:
            return 0.10

    def get_diagnostics(self) -> List[str]:
        """生成 tiandao-pressure 配置的诊断信息（供 drawdown_attribution.log 追加）。

        Returns:
            诊断文本行列表。
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"【tiandao-pressure 通道压力线诊断】 配置名: tiandao_pressure | 策略: {self.__class__.__name__}")
        lines.append(f"  stop_loss_pct={self.stop_loss_pct} | time_stop_days={self.time_stop_days} | time_stop_min_return={self.time_stop_min_return}")
        lines.append("  盈利保护(profit_protect): 未启用 ✅（本策略无 profit_protect_threshold/retracement 字段）")
        lines.append("")

        # 2. peak 初始化检查：选取 2024-10-08 买入的前 5 只股票
        target_buy_date = date(2024, 10, 8)
        candidates = [sc for sc, ed in self._entry_dates.items() if ed == target_buy_date]
        lines.append("### 2. peak 初始化检查（2024-10-08 买入，最多5只） ###")
        if not candidates:
            lines.append("  2024-10-08 当天无买入记录")
        else:
            for sc in candidates[:5]:
                log = self._daily_peak_log.get(sc, [])
                lines.append(f"  [{sc}] 买入日={target_buy_date} 轨迹（date | close | peak | peak*0.90）:")
                for d, close, peak in log:
                    lines.append(f"    {d} | {close:.2f} | {peak:.2f} | {peak*0.90:.2f}")
        lines.append("")

        # 3. 卖出分支触发统计
        lines.append("### 3. 卖出分支触发统计 ###")
        c_total = self._c_tier_count
        others = self._stop_loss_count + self._time_stop_count + self._tier1_count + self._tier2_count
        lines.append(
            f"  7%止损={self._stop_loss_count} | 时间止损={self._time_stop_count} | "
            f"Tier1(A档)={self._tier1_count} | Tier2(B档)={self._tier2_count} | C档(peak移动止盈)={self._c_tier_count}"
        )
        if others > 0 and c_total > others:
            lines.append(f"  ⚠️ C档次数({c_total}) >> 其他分支总和({others})，peak 逻辑可能是主要问题")
        lines.append("")

        # 4. 持仓周期分布
        lines.append("### 4. 持仓周期分布（完整清仓的持仓天数） ###")
        if self._holding_days:
            bins = {"1天": 0, "2-3天": 0, "4-7天": 0, "8-15天": 0, "15+天": 0}
            total = len(self._holding_days)
            for h in self._holding_days:
                if h <= 1:
                    bins["1天"] += 1
                elif h <= 3:
                    bins["2-3天"] += 1
                elif h <= 7:
                    bins["4-7天"] += 1
                elif h <= 15:
                    bins["8-15天"] += 1
                else:
                    bins["15+天"] += 1
            short_ratio = (bins["1天"] + bins["2-3天"]) / total
            lines.append(f"  清仓交易总数: {total}")
            for k, v in bins.items():
                lines.append(f"    {k}: {v}（{v/total:.1%}）")
            if short_ratio > 0.5:
                lines.append(f"  ⚠️ 1-3天占比 {short_ratio:.1%} > 50%，疑为'买入即清仓'循环")
        else:
            lines.append("  无完整清仓记录")
        lines.append("")

        # 5. REF 检查：任选一只持仓天数较长的股票，连续5天
        # 校验方案：pressure = REF(金牛,1)，故当日 pressure 应等于前一日快照的 td_jinniu。
        # 但前一日 td_jinniu 在数据缺口时可能为 0，无法可靠读取。改用 pressure 序列自身的
        # 连续性来确认 REF 平移正确：金牛为慢速平滑线，相邻交易日 pressure 应平滑过渡、
        # 变化幅度极小（XMA 慢线日变幅通常 < 2%），且不会出现跳变。
        lines.append("### 5. REF 检查（pressure 序列连续性校验，pressure=REF(金牛,1)） ###")
        sampled = None
        for sc, snap in self._ref_check.items():
            if len(snap) >= 5:
                sampled = sc
                break
        if sampled is None and self._ref_check:
            sampled = next(iter(self._ref_check))
            snap = self._ref_check[sampled]
            if len(snap) > 5:
                snap = snap[-5:]
        else:
            snap = self._ref_check.get(sampled, [])[-5:]
        if sampled is None:
            lines.append("  无可用采样数据")
        else:
            lines.append(f"  [{sampled}]（date | pressure=REF(金牛,1) | 相对前日变化 | 连续?）")
            prev_pressure = None
            for d, pressure in snap:
                if prev_pressure is not None and prev_pressure > 0:
                    change = abs(pressure - prev_pressure) / prev_pressure
                    cont = "✅" if change <= 0.03 else "❓"
                    lines.append(
                        f"    {d} | pressure={pressure:.2f} | {change:.2%} {cont}"
                    )
                else:
                    lines.append(f"    {d} | pressure={pressure:.2f} | --（首个样本） ✅")
                prev_pressure = pressure
        lines.append("")
        lines.append("=" * 60)
        return lines


