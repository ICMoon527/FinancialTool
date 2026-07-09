# -*- coding: utf-8 -*-
"""
可插拔退出策略模块。

提供退出策略的抽象基类、注册表和两种内置实现：
- SimpleExitStrategy: 固定比例止盈止损
- TieredExitStrategy: 动态分级止盈（默认模式 + 强势模式）
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
    exit_type: str = "full"                 # "full" | "partial"
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
    tier: str = "default"           # "default" | "big_winner"
    breakeven_activated: bool = False
    partial_sold: bool = False
    peak_price: float = 0.0         # 持仓期间最高收盘价


@register_exit_strategy("tiered")
class TieredExitStrategy(ExitStrategy):
    """动态分级止盈策略。

    默认模式：
      - 止损 -12%
      - 涨幅达到 +10% → 保本止损
      - 涨幅达到 +20% → 卖出 50% 仓位，剩余 50% 启动 10% 移动止盈
      - 最长持股 44 个交易日

    强势模式（买入后 10 个交易日内收盘价涨幅 ≥15% 触发）：
      - 止损 -15%（给更多波动空间）
      - 涨幅达到 +10% → 保本止损（不变）
      - 不触发 +20% 部分止盈（全部仓位保留）
      - 15% 移动止盈（给强势股更多回撤空间）
    """

    display_name = "动态分级止盈"
    description = "根据持仓表现自动切换模式：默认严格止损 + 强势股放宽追踪止盈"

    def __init__(
        self,
        # 默认模式参数
        default_stop_loss_pct: float = 0.12,
        default_breakeven_trigger_pct: float = 0.10,
        default_partial_take_profit_pct: float = 0.20,
        default_partial_ratio: float = 0.50,
        default_trailing_stop_pct: float = 0.10,
        # 强势模式参数
        big_winner_trigger_days: int = 10,
        big_winner_trigger_pct: float = 0.15,
        big_winner_stop_loss_pct: float = 0.15,
        big_winner_trailing_stop_pct: float = 0.15,
        # 通用参数
        time_stop_days: int = 44,
    ):
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_breakeven_trigger_pct = default_breakeven_trigger_pct
        self.default_partial_take_profit_pct = default_partial_take_profit_pct
        self.default_partial_ratio = default_partial_ratio
        self.default_trailing_stop_pct = default_trailing_stop_pct
        self.big_winner_trigger_days = big_winner_trigger_days
        self.big_winner_trigger_pct = big_winner_trigger_pct
        self.big_winner_stop_loss_pct = big_winner_stop_loss_pct
        self.big_winner_trailing_stop_pct = big_winner_trailing_stop_pct
        self.time_stop_days = time_stop_days
        self._states: Dict[str, _TieredPositionState] = {}

    def validate_config(self) -> bool:
        for pct_name, val in [
            ("default_stop_loss_pct", self.default_stop_loss_pct),
            ("default_partial_take_profit_pct", self.default_partial_take_profit_pct),
            ("big_winner_stop_loss_pct", self.big_winner_stop_loss_pct),
        ]:
            if val <= 0 or val >= 1:
                logger.error("%s 必须在 0-1 之间，当前值: %s", pct_name, val)
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

        for stock_code in list(positions.keys()):
            if stock_code not in self._states:
                continue
            state = self._states[stock_code]

            # T+1 规则
            if state.entry_date == current_date:
                continue

            if phase == "open":
                s = self._check_open_tiered(stock_code, state, price_provider)
                if s:
                    signals.append(s)
            elif phase == "intraday":
                s = self._check_intraday_tiered(stock_code, state, price_provider)
                if s:
                    signals.append(s)
            elif phase == "close":
                s = self._check_close_tiered(
                    stock_code, state, price_provider, trading_dates, current_date_index, current_date
                )
                if s:
                    signals.append(s)

        return signals

    def _check_open_tiered(
        self,
        stock_code: str,
        state: _TieredPositionState,
        price_provider: Callable[[str, str], Optional[float]],
    ) -> Optional[ExitSignal]:
        """检查开盘价：止损 + 保本"""
        open_price = price_provider(stock_code, "open")
        if open_price is None:
            return None

        stop_loss_pct = (
            self.big_winner_stop_loss_pct if state.tier == "big_winner"
            else self.default_stop_loss_pct
        )
        stop_loss_price = state.entry_price * (1 - stop_loss_pct)

        # 保本止损
        if state.breakeven_activated:
            stop_loss_price = state.entry_price

        if open_price <= stop_loss_price:
            reason = "保本出局" if state.breakeven_activated else "止损出局"
            logger.info(
                "分级止盈 [%s][%s]: 开盘价 %.2f <= %s %.2f",
                stock_code, state.tier, open_price, "保本价" if state.breakeven_activated else "止损价", stop_loss_price,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason=reason,
                sell_price=open_price,
                price_type="open",
            )

        return None

    def _check_intraday_tiered(
        self,
        stock_code: str,
        state: _TieredPositionState,
        price_provider: Callable[[str, str], Optional[float]],
    ) -> Optional[ExitSignal]:
        """检查盘中：止损 + 部分止盈 + 移动止盈"""
        high_price = price_provider(stock_code, "high")
        low_price = price_provider(stock_code, "low")
        if high_price is None or low_price is None:
            return None

        if state.tier == "default" and not state.partial_sold:
            # 默认模式：检查部分止盈（最高价触发）
            take_profit_price = state.entry_price * (1 + self.default_partial_take_profit_pct)
            if high_price >= take_profit_price:
                state.partial_sold = True
                state.breakeven_activated = True
                state.peak_price = max(state.peak_price, high_price)
                logger.info(
                    "分级止盈 [%s][default]: 触发部分止盈 最高价 %.2f >= %.2f, 卖出 %.0f%%",
                    stock_code, high_price, take_profit_price, self.default_partial_ratio * 100,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="部分止盈",
                    exit_type="partial",
                    exit_ratio=self.default_partial_ratio,
                    sell_price=take_profit_price,
                    price_type="intraday",
                )

        # 移动止盈（partial_sold 的默认模式，或 big_winner 模式）
        if state.partial_sold or state.tier == "big_winner":
            trailing_dd = (
                self.big_winner_trailing_stop_pct if state.tier == "big_winner"
                else self.default_trailing_stop_pct
            )
            trailing_stop = state.peak_price * (1 - trailing_dd)

            # 止损线
            stop_loss_pct = (
                self.big_winner_stop_loss_pct if state.tier == "big_winner"
                else self.default_stop_loss_pct
            )
            stop_loss_price = state.entry_price * (1 - stop_loss_pct)
            if state.breakeven_activated:
                stop_loss_price = state.entry_price

            actual_stop = max(stop_loss_price, trailing_stop)

            if low_price <= actual_stop:
                reason = (
                    "强势移动止盈" if state.tier == "big_winner"
                    else "移动止盈"
                )
                logger.info(
                    "分级止盈 [%s][%s]: 最低价 %.2f <= 止盈价 %.2f",
                    stock_code, state.tier, low_price, actual_stop,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason=reason,
                    sell_price=actual_stop,
                    price_type="intraday",
                )
            return None

        # 默认模式未部分止盈：检查止损
        stop_loss_price = state.entry_price * (1 - self.default_stop_loss_pct)
        if state.breakeven_activated:
            stop_loss_price = state.entry_price

        if low_price <= stop_loss_price:
            reason = "保本出局" if state.breakeven_activated else "止损出局"
            logger.info(
                "分级止盈 [%s][default]: 最低价 %.2f <= %s %.2f",
                stock_code, low_price, "保本价" if state.breakeven_activated else "止损价", stop_loss_price,
            )
            return ExitSignal(
                stock_code=stock_code,
                reason=reason,
                sell_price=stop_loss_price,
                price_type="intraday",
            )

        return None

    def _check_close_tiered(
        self,
        stock_code: str,
        state: _TieredPositionState,
        price_provider: Callable[[str, str], Optional[float]],
        trading_dates: List[date],
        current_date_index: int,
        current_date: date,
    ) -> Optional[ExitSignal]:
        """检查收盘：更新峰值 + 时间止损 + 强势模式检测"""
        close_price = price_provider(stock_code, "close")
        if close_price is None:
            return None

        # 更新最高收盘价
        if close_price > state.peak_price:
            state.peak_price = close_price

        # 检查保本激活
        if not state.breakeven_activated:
            gain_pct = (close_price - state.entry_price) / state.entry_price
            if gain_pct >= self.default_breakeven_trigger_pct:
                state.breakeven_activated = True
                logger.info(
                    "分级止盈 [%s]: 收盘价涨幅 %.1f%% >= %.0f%%, 保本止损激活",
                    stock_code, gain_pct * 100, self.default_breakeven_trigger_pct * 100,
                )

        # 强势模式检测（仅在默认模式且未部分止盈时检测）
        if state.tier == "default" and not state.partial_sold:
            gain_pct = (close_price - state.entry_price) / state.entry_price
            # 计算从买入至今的交易日数
            try:
                entry_index = trading_dates.index(state.entry_date)
                days_held = current_date_index - entry_index
            except ValueError:
                days_held = 999

            if days_held <= self.big_winner_trigger_days and gain_pct >= self.big_winner_trigger_pct:
                state.tier = "big_winner"
                logger.info(
                    "分级止盈 [%s]: 触发强势模式! %d天内涨幅 %.1f%% >= %.0f%%",
                    stock_code, days_held, gain_pct * 100, self.big_winner_trigger_pct * 100,
                )

        # 时间止损
        try:
            entry_index = trading_dates.index(state.entry_date)
            holding_days = current_date_index - entry_index
            if holding_days >= self.time_stop_days:
                logger.info(
                    "分级止盈 [%s][%s]: 持股 %d 天 >= %d 天，到期平仓",
                    stock_code, state.tier, holding_days, self.time_stop_days,
                )
                return ExitSignal(
                    stock_code=stock_code,
                    reason="到期平仓",
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