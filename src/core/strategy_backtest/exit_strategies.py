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


# ============================================================
# 天道系列退出策略公用状态
# ============================================================


@dataclass
class _TiandaoBaseState:
    """天道系列策略的公用持仓状态"""
    entry_date: date
    entry_price: float
    partial_sold: bool = False
    rebuy_count: int = 0       # 已加仓次数


# ============================================================
# 策略 A：天道金牛清仓
# ============================================================


@register_exit_strategy("tiandao_jinniu")
class TiandaoJinniuExitStrategy(ExitStrategy):
    """天道金牛清仓策略。

    买入后在通道下轨（金钻趋势）附近持有，反弹到通道上轨（金牛线）清仓止盈；
    止损设在买入价下方 5%（买入价 × 0.95），5% 固定止损，不依赖金钻趋势线。
    """

    display_name = "天道金牛清仓"
    description = "天道超跌买入后，反弹到金牛压力位清仓，买入价×0.95止损"

    def __init__(
        self,
        max_holding_days: int = 30,
        stop_loss_buffer: float = 0.05,
        indicator_provider: Optional[Callable[[str], Optional[TiandaoIndicatorSnapshot]]] = None,
    ):
        self.max_holding_days = max_holding_days
        self.stop_loss_buffer = stop_loss_buffer
        self._indicator_provider = indicator_provider
        self._states: Dict[str, _TiandaoBaseState] = {}

    def validate_config(self) -> bool:
        if self.max_holding_days < 1:
            logger.error("max_holding_days 必须 >= 1，当前值: %s", self.max_holding_days)
            return False
        if self.stop_loss_buffer <= 0 or self.stop_loss_buffer >= 0.5:
            logger.error("stop_loss_buffer 必须在 0-0.5 之间，当前值: %s", self.stop_loss_buffer)
            return False
        return True

    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        self._states[stock_code] = _TiandaoBaseState(
            entry_date=entry_date,
            entry_price=entry_price,
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

        for stock_code in list(positions.keys()):
            if stock_code not in self._states:
                continue
            state = self._states[stock_code]

            # T+1 规则
            if state.entry_date == current_date:
                continue

            # 获取天道指标
            indicators = None
            if self._indicator_provider:
                indicators = self._indicator_provider(stock_code)

            if phase == "open":
                s = self._check_open(stock_code, state, price_provider, indicators)
                if s:
                    signals.append(s)
            elif phase == "intraday":
                s = self._check_intraday(stock_code, state, price_provider, indicators)
                if s:
                    signals.append(s)
            elif phase == "close":
                s = self._check_close(stock_code, state, price_provider, trading_dates, current_date_index)
                if s:
                    signals.append(s)

        return signals

    def _check_open(
        self,
        stock_code: str,
        state: _TiandaoBaseState,
        price_provider: Callable[[str, str], Optional[float]],
        indicators: Optional[TiandaoIndicatorSnapshot],
    ) -> Optional[ExitSignal]:
        """检查开盘价：是否直接开在金牛线上方"""
        open_price = price_provider(stock_code, "open")
        if open_price is None:
            return None

        if indicators is None or indicators.td_jinniu <= 0:
            return None

        if open_price >= indicators.td_jinniu:
            logger.info(
                "天道金牛清仓 [%s]: 开盘价 %.2f >= 金牛线 %.2f，清仓止盈",
                stock_code, open_price, indicators.td_jinniu,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason="金牛清仓(开盘)",
                sell_price=open_price,
                price_type="open",
            )

        return None

    def _check_intraday(
        self,
        stock_code: str,
        state: _TiandaoBaseState,
        price_provider: Callable[[str, str], Optional[float]],
        indicators: Optional[TiandaoIndicatorSnapshot],
    ) -> Optional[ExitSignal]:
        """检查盘中：金牛止盈 或 买入价5%止损"""
        high_price = price_provider(stock_code, "high")
        low_price = price_provider(stock_code, "low")
        if high_price is None or low_price is None:
            return None

        if indicators is None:
            return None

        # 优先级1：金牛线止盈（最高价触发）
        if indicators.td_jinniu > 0 and high_price >= indicators.td_jinniu:
            logger.info(
                "天道金牛清仓 [%s]: 最高价 %.2f >= 金牛线 %.2f，清仓止盈",
                stock_code, high_price, indicators.td_jinniu,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason="金牛清仓(盘中)",
                sell_price=indicators.td_jinniu,
                price_type="intraday",
            )

        # 优先级2：买入价5%止损
        stop_loss_price = state.entry_price * (1 - self.stop_loss_buffer)
        if low_price <= stop_loss_price:
            logger.info(
                "天道金牛清仓 [%s]: 最低价 %.2f <= 止损价 %.2f (买入价%.2f × %.0f%%)，止损清仓",
                stock_code, low_price, stop_loss_price,
                state.entry_price, (1 - self.stop_loss_buffer) * 100,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason="买入价止损",
                sell_price=stop_loss_price,
                price_type="intraday",
            )

        return None

    def _check_close(
        self,
        stock_code: str,
        state: _TiandaoBaseState,
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
    ) -> Optional[ExitSignal]:
        """检查收盘：持股超时"""
        try:
            entry_index = trading_dates.index(state.entry_date)
            holding_days = current_date_index - entry_index
            if holding_days >= self.max_holding_days:
                close_price = price_provider(stock_code, "close")
                if close_price is None:
                    return None
                logger.info(
                    "天道金牛清仓 [%s]: 持股 %d 天 >= %d 天，到期清仓",
                    stock_code, holding_days, self.max_holding_days,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="到期清仓",
                    sell_price=close_price,
                    price_type="close",
                )
        except ValueError:
            logger.warning("股票 %s 的买入日期 %s 不在交易日列表中", stock_code, state.entry_date)

        return None

    def cleanup_position(self, stock_code: str) -> None:
        self._states.pop(stock_code, None)

    def reset(self) -> None:
        self._states.clear()


# ============================================================
# 策略 B：天道BBI滚动
# ============================================================


@dataclass
class _BbiRollingState(_TiandaoBaseState):
    """BBI滚动策略扩展状态"""
    released_cash: float = 0.0   # 减仓释放的现金（用于后续加仓）


@register_exit_strategy("tiandao_bbi_rolling")
class TiandaoBbiRollingExitStrategy(ExitStrategy):
    """天道BBI滚动策略。

    买入后反弹到BBI多空分界线减仓50%锁定利润；
    回踩金钻趋势支撑位加仓买回（低吸）；
    反弹到金牛线清仓；跌破支撑位下方5%止损。
    """

    display_name = "天道BBI滚动"
    description = "BBI多空线滚动操作：反弹减仓、回踩加仓、金牛清仓、破支撑止损"

    def __init__(
        self,
        max_holding_days: int = 40,
        max_rebuy_count: int = 1,
        stop_loss_buffer: float = 0.05,
        indicator_provider: Optional[Callable[[str], Optional[TiandaoIndicatorSnapshot]]] = None,
    ):
        self.max_holding_days = max_holding_days
        self.max_rebuy_count = max_rebuy_count
        self.stop_loss_buffer = stop_loss_buffer
        self._indicator_provider = indicator_provider
        self._states: Dict[str, _BbiRollingState] = {}

    def validate_config(self) -> bool:
        if self.max_holding_days < 1:
            logger.error("max_holding_days 必须 >= 1")
            return False
        if self.stop_loss_buffer <= 0 or self.stop_loss_buffer >= 0.5:
            logger.error("stop_loss_buffer 必须在 0-0.5 之间")
            return False
        if self.max_rebuy_count < 0:
            logger.error("max_rebuy_count 必须 >= 0")
            return False
        return True

    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        self._states[stock_code] = _BbiRollingState(
            entry_date=entry_date,
            entry_price=entry_price,
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

        for stock_code in list(positions.keys()):
            if stock_code not in self._states:
                continue
            state = self._states[stock_code]

            # T+1 规则
            if state.entry_date == current_date:
                continue

            # 获取天道指标
            indicators = None
            if self._indicator_provider:
                indicators = self._indicator_provider(stock_code)

            if phase == "intraday":
                s = self._check_intraday(stock_code, state, price_provider, indicators)
                if s:
                    signals.append(s)
            elif phase == "close":
                s = self._check_close(stock_code, state, price_provider, trading_dates, current_date_index)
                if s:
                    signals.append(s)

        return signals

    def _check_intraday(
        self,
        stock_code: str,
        state: _BbiRollingState,
        price_provider: Callable[[str, str], Optional[float]],
        indicators: Optional[TiandaoIndicatorSnapshot],
    ) -> Optional[ExitSignal]:
        """检查盘中：BBI减仓、回踩加仓、金牛止盈、止损"""
        high_price = price_provider(stock_code, "high")
        low_price = price_provider(stock_code, "low")
        if high_price is None or low_price is None:
            return None

        if indicators is None:
            return None

        # 优先级1：金牛线止盈（全清）
        if indicators.td_jinniu > 0 and high_price >= indicators.td_jinniu:
            logger.info(
                "天道BBI滚动 [%s]: 最高价 %.2f >= 金牛线 %.2f，清仓止盈",
                stock_code, high_price, indicators.td_jinniu,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason="金牛清仓",
                sell_price=indicators.td_jinniu,
                price_type="intraday",
            )

        if not state.partial_sold:
            # 未减仓状态：检查是否触发BBI减仓
            if indicators.td_bbi > 0 and high_price >= indicators.td_bbi:
                state.partial_sold = True
                logger.info(
                    "天道BBI滚动 [%s]: 最高价 %.2f >= BBI %.2f，减仓50%%",
                    stock_code, high_price, indicators.td_bbi,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="BBI减仓50%",
                    exit_type="partial",
                    exit_ratio=0.5,
                    sell_price=indicators.td_bbi,
                    price_type="intraday",
                )
        else:
            # 已减仓状态：检查回踩加仓 或 止损
            if indicators.td_jinzuan > 0:
                stop_loss_price = indicators.td_jinzuan * (1 - self.stop_loss_buffer)

                # 先检查止损（优先级高于加仓）
                if low_price <= stop_loss_price:
                    logger.info(
                        "天道BBI滚动 [%s]: 最低价 %.2f <= 止损价 %.2f (金钻%.2f × %.0f%%)，清仓止损",
                        stock_code, low_price, stop_loss_price,
                        indicators.td_jinzuan, (1 - self.stop_loss_buffer) * 100,
                    )
                    return ExitSignal(
                        stock_code=stock_code,
                        reason="跌破支撑止损",
                        sell_price=stop_loss_price,
                        price_type="intraday",
                    )

                # 回踩金钻趋势线 → 加仓买回
                if state.rebuy_count < self.max_rebuy_count and low_price <= indicators.td_jinzuan:
                    state.rebuy_count += 1
                    logger.info(
                        "天道BBI滚动 [%s]: 最低价 %.2f <= 金钻趋势 %.2f，加仓买回50%% (第%d次)",
                        stock_code, low_price, indicators.td_jinzuan, state.rebuy_count,
                    )
                    return ExitSignal(
                        stock_code=stock_code,
                        reason="回踩加仓",
                        exit_type="rebuy",
                        exit_ratio=0.5,
                        sell_price=indicators.td_jinzuan,
                        price_type="intraday",
                    )

        return None

    def _check_close(
        self,
        stock_code: str,
        state: _BbiRollingState,
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
    ) -> Optional[ExitSignal]:
        """检查收盘：持股超时"""
        try:
            entry_index = trading_dates.index(state.entry_date)
            holding_days = current_date_index - entry_index
            if holding_days >= self.max_holding_days:
                close_price = price_provider(stock_code, "close")
                if close_price is None:
                    return None
                logger.info(
                    "天道BBI滚动 [%s]: 持股 %d 天 >= %d 天，到期清仓",
                    stock_code, holding_days, self.max_holding_days,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="到期清仓",
                    sell_price=close_price,
                    price_type="close",
                )
        except ValueError:
            pass

        return None

    def cleanup_position(self, stock_code: str) -> None:
        self._states.pop(stock_code, None)

    def reset(self) -> None:
        self._states.clear()


# ============================================================
# 策略 C：天道金牛2分级保护
# ============================================================


@register_exit_strategy("tiandao_jinniu2_protection")
class TiandaoJinniu2ProtectionExitStrategy(ExitStrategy):
    """天道金牛2分级保护策略。

    以金牛2（EMA(金钻趋势, 25)）为动态保护线：
    - 跌破金牛2 → 趋势转弱，减仓50%
    - 跌破金牛2 × 0.95 → 趋势走坏，清仓止损
    - 收盘站上金牛2 → 趋势恢复，加仓买回
    - 反弹到金牛线 → 清仓止盈
    """

    display_name = "天道金牛2保护"
    description = "金牛2分级保护：趋势转弱减仓、趋势走坏清仓、趋势恢复加仓、金牛清仓"

    def __init__(
        self,
        max_holding_days: int = 35,
        stop_loss_buffer: float = 0.05,
        indicator_provider: Optional[Callable[[str], Optional[TiandaoIndicatorSnapshot]]] = None,
    ):
        self.max_holding_days = max_holding_days
        self.stop_loss_buffer = stop_loss_buffer
        self._indicator_provider = indicator_provider
        self._states: Dict[str, _TiandaoBaseState] = {}

    def validate_config(self) -> bool:
        if self.max_holding_days < 1:
            logger.error("max_holding_days 必须 >= 1")
            return False
        if self.stop_loss_buffer <= 0 or self.stop_loss_buffer >= 0.5:
            logger.error("stop_loss_buffer 必须在 0-0.5 之间")
            return False
        return True

    def initialize_position(self, stock_code: str, entry_price: float, entry_date: date) -> None:
        self._states[stock_code] = _TiandaoBaseState(
            entry_date=entry_date,
            entry_price=entry_price,
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

        for stock_code in list(positions.keys()):
            if stock_code not in self._states:
                continue
            state = self._states[stock_code]

            # T+1 规则
            if state.entry_date == current_date:
                continue

            # 获取天道指标
            indicators = None
            if self._indicator_provider:
                indicators = self._indicator_provider(stock_code)

            if phase == "intraday":
                s = self._check_intraday(stock_code, state, price_provider, indicators)
                if s:
                    signals.append(s)
            elif phase == "close":
                s = self._check_close(stock_code, state, price_provider, indicators, trading_dates, current_date_index)
                if s:
                    signals.append(s)

        return signals

    def _check_intraday(
        self,
        stock_code: str,
        state: _TiandaoBaseState,
        price_provider: Callable[[str, str], Optional[float]],
        indicators: Optional[TiandaoIndicatorSnapshot],
    ) -> Optional[ExitSignal]:
        """检查盘中：金牛止盈、金牛2减仓、金牛2下方止损"""
        high_price = price_provider(stock_code, "high")
        low_price = price_provider(stock_code, "low")
        if high_price is None or low_price is None:
            return None

        if indicators is None:
            return None

        # 获取开盘价，用于计算实际可成交的卖出价
        # 当股票跳空低开直接跌破触发价时，不能以触发价卖出（当天该价格不存在）
        open_price = price_provider(stock_code, "open")

        # 优先级1：金牛线止盈（全清）
        if indicators.td_jinniu > 0 and high_price >= indicators.td_jinniu:
            logger.info(
                "天道金牛2保护 [%s]: 最高价 %.2f >= 金牛线 %.2f，清仓止盈",
                stock_code, high_price, indicators.td_jinniu,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason="金牛清仓",
                sell_price=indicators.td_jinniu,
                price_type="intraday",
            )

        if indicators.td_jinniu2 <= 0:
            return None

        if not state.partial_sold:
            # 未减仓：检查是否跌破金牛2（趋势转弱）
            if low_price <= indicators.td_jinniu2:
                state.partial_sold = True
                # 卖出价不能高于开盘价（如果跳空低开跌破金牛2，金牛2当天不存在）
                actual_sell_price = (
                    min(open_price, indicators.td_jinniu2)
                    if open_price is not None
                    else indicators.td_jinniu2
                )
                logger.info(
                    "天道金牛2保护 [%s]: 最低价 %.2f <= 金牛2 %.2f，趋势转弱减仓50%%（卖出价=%.2f）",
                    stock_code, low_price, indicators.td_jinniu2, actual_sell_price,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="趋势转弱减仓",
                    exit_type="partial",
                    exit_ratio=0.5,
                    sell_price=actual_sell_price,
                    price_type="intraday",
                )
        else:
            # 已减仓：检查是否跌破金牛2 × 0.95（趋势走坏）
            stop_loss_price = indicators.td_jinniu2 * (1 - self.stop_loss_buffer)
            if low_price <= stop_loss_price:
                # 卖出价不能高于开盘价（如果跳空低开跌破止损价，止损价当天不存在）
                actual_sell_price = (
                    min(open_price, stop_loss_price)
                    if open_price is not None
                    else stop_loss_price
                )
                logger.info(
                    "天道金牛2保护 [%s]: 最低价 %.2f <= 止损价 %.2f (金牛2 %.2f × %.0f%%)，趋势走坏清仓（卖出价=%.2f）",
                    stock_code, low_price, stop_loss_price,
                    indicators.td_jinniu2, (1 - self.stop_loss_buffer) * 100,
                    actual_sell_price,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="趋势走坏清仓",
                    sell_price=actual_sell_price,
                    price_type="intraday",
                )

        return None

    def _check_close(
        self,
        stock_code: str,
        state: _TiandaoBaseState,
        price_provider: Callable[[str, str], Optional[float]],
        indicators: Optional[TiandaoIndicatorSnapshot],
        trading_dates: List[date],
        current_date_index: int,
    ) -> Optional[ExitSignal]:
        """检查收盘：趋势恢复加仓 或 持股超时"""
        if indicators is not None and state.partial_sold:
            close_price = price_provider(stock_code, "close")
            if close_price is not None and indicators.td_jinniu2 > 0:
                # 收盘价重新站上金牛2 → 趋势恢复，加仓买回
                if close_price > indicators.td_jinniu2:
                    state.partial_sold = False
                    logger.info(
                        "天道金牛2保护 [%s]: 收盘价 %.2f > 金牛2 %.2f，趋势恢复加仓买回",
                        stock_code, close_price, indicators.td_jinniu2,
                    )
                    return ExitSignal(
                        stock_code=stock_code,
                        reason="趋势恢复加仓",
                        exit_type="rebuy",
                        exit_ratio=1.0,
                        sell_price=close_price,
                        price_type="close",
                    )

        # 持股超时
        try:
            entry_index = trading_dates.index(state.entry_date)
            holding_days = current_date_index - entry_index
            if holding_days >= self.max_holding_days:
                close_price = price_provider(stock_code, "close")
                if close_price is None:
                    return None
                logger.info(
                    "天道金牛2保护 [%s]: 持股 %d 天 >= %d 天，到期清仓",
                    stock_code, holding_days, self.max_holding_days,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="到期清仓",
                    sell_price=close_price,
                    price_type="close",
                )
        except ValueError:
            pass

        return None

    def cleanup_position(self, stock_code: str) -> None:
        self._states.pop(stock_code, None)

    def reset(self) -> None:
        self._states.clear()