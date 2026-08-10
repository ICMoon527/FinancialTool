# -*- coding: utf-8 -*-
"""
策略回测核心引擎。

实现基于时间步进的回测框架，包含交易执行逻辑。
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .data_access import TimeIsolatedDataProvider

# 天道指标（供天道系列退出策略使用）
from indicators.indicators.tiandao import Tiandao
from .exit_strategies import TiandaoIndicatorSnapshot

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """订单"""
    order_id: str
    stock_code: str
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_date: date = field(default_factory=date.today)
    filled_date: Optional[date] = None
    filled_price: Optional[float] = None


@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    stock_code: str
    order_type: OrderType
    quantity: int
    price: float
    date: date
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """持仓"""
    stock_code: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0


class Portfolio:
    """投资组合"""

    def __init__(self, initial_capital: float = 1000000.0):
        """
        初始化投资组合。

        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.equity_history: List[Tuple[date, float]] = []

    def get_position(self, stock_code: str) -> Optional[Position]:
        """
        获取持仓。

        Args:
            stock_code: 股票代码

        Returns:
            持仓对象，如果没有持仓则返回None
        """
        return self.positions.get(stock_code)

    def update_position_price(self, stock_code: str, current_price: float) -> None:
        """
        更新持仓价格。

        Args:
            stock_code: 股票代码
            current_price: 当前价格
        """
        if stock_code in self.positions:
            self.positions[stock_code].current_price = current_price

    def get_total_equity(self) -> float:
        """
        获取总权益。

        Returns:
            总权益（现金 + 持仓市值）
        """
        equity = self.cash
        for pos in self.positions.values():
            equity += pos.quantity * pos.current_price
        return equity

    def record_equity(self, current_date: date) -> None:
        """
        记录当前权益。

        Args:
            current_date: 当前日期
        """
        self.equity_history.append((current_date, self.get_total_equity()))

    def execute_trade(
        self,
        trade: Trade,
    ) -> None:
        """
        执行交易。

        Args:
            trade: 交易对象
        """
        self.trades.append(trade)

        if trade.order_type == OrderType.BUY:
            total_cost = trade.quantity * trade.price + trade.commission + trade.slippage
            self.cash -= total_cost

            if trade.stock_code in self.positions:
                pos = self.positions[trade.stock_code]
                total_qty = pos.quantity + trade.quantity
                total_cost = pos.quantity * pos.avg_cost + trade.quantity * trade.price
                pos.quantity = total_qty
                pos.avg_cost = total_cost / total_qty
                pos.current_price = trade.price
            else:
                self.positions[trade.stock_code] = Position(
                    stock_code=trade.stock_code,
                    quantity=trade.quantity,
                    avg_cost=trade.price,
                    current_price=trade.price,
                )
        else:
            if trade.stock_code in self.positions:
                pos = self.positions[trade.stock_code]
                total_proceeds = trade.quantity * trade.price - trade.commission - trade.slippage
                self.cash += total_proceeds

                pos.quantity -= trade.quantity
                if pos.quantity <= 0:
                    del self.positions[trade.stock_code]


class StrategyBacktestEngine:
    """
    策略回测引擎。
    """

    def __init__(
        self,
        data_provider: Any,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.001,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_positions: Optional[int] = None,
        max_holding_days: int = 5,
        exit_strategy: Optional["ExitStrategy"] = None,
        market_trend_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化回测引擎。

        Args:
            data_provider: 数据提供器
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            start_date: 回测开始日期
            end_date: 回测结束日期
            stop_loss_pct: 止损百分比（已弃用，请使用 exit_strategy）
            take_profit_pct: 止盈百分比（已弃用，请使用 exit_strategy）
            max_positions: 最大持仓数量
            max_holding_days: 最长持股时间（已弃用，请使用 exit_strategy）
            exit_strategy: 退出策略实例（推荐使用）
            market_trend_filter: 大盘趋势过滤器配置，如 {"index_code": "000001.SH", "ma_period": 20}
                                 为 None 或空字典时不启用
        """
        self._original_data_provider = data_provider
        self._data_provider = TimeIsolatedDataProvider(data_provider)
        self.portfolio = Portfolio(initial_capital)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.start_date = start_date
        self.end_date = end_date
        self.trading_dates: List[date] = []
        self.current_date_index = 0
        self._strategy = None
        self._strategies: List[Any] = []  # 多策略支持
        self._stock_pool: List[str] = []
        self.max_positions = max_positions
        self._should_stop = False

        # 安装 Ctrl+C 信号处理器，支持安全停止回测（仅在主线程中有效）
        import threading
        self._sigint_installed = False
        if threading.current_thread() is threading.main_thread():
            self._original_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
            self._sigint_installed = True

        # 大盘趋势过滤器
        self.market_trend_filter = market_trend_filter or {}
        self._index_cache: Dict[str, pd.DataFrame] = {}  # 指数数据缓存

        # 退出策略
        if exit_strategy is not None:
            self.exit_strategy = exit_strategy
        else:
            # 向后兼容：从旧参数构造 SimpleExitStrategy
            from .exit_strategies import SimpleExitStrategy
            self.exit_strategy = SimpleExitStrategy(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_holding_days=max_holding_days,
            )

        # 天道指标计算器（供天道系列退出策略使用）
        self._tiandao_indicator = Tiandao()
        # 天道指标缓存：{stock_code: TiandaoIndicatorSnapshot}
        self._tiandao_cache: Dict[str, TiandaoIndicatorSnapshot] = {}

    def set_strategy(self, strategy: Any) -> None:
        """
        设置回测策略。

        Args:
            strategy: 策略对象
        """
        self._strategy = strategy
        self._strategies = [strategy]  # 同时更新多策略列表
        if hasattr(strategy, "_data_provider"):
            strategy._data_provider = self._data_provider

    def set_strategies(self, strategies: List[Any]) -> None:
        """
        设置多个回测策略（多策略支持）。

        Args:
            strategies: 策略对象列表
        """
        self._strategies = strategies
        if strategies:
            self._strategy = strategies[0]  # 保持向后兼容
        # 为每个策略设置数据提供者
        for strategy in strategies:
            if hasattr(strategy, "_data_provider"):
                strategy._data_provider = self._data_provider

    def set_stock_pool(self, stock_pool: List[str]) -> None:
        """
        设置股票池。

        Args:
            stock_pool: 股票代码列表
        """
        self._stock_pool = stock_pool

    def set_trading_dates(self, trading_dates: List[date]) -> None:
        """
        设置交易日历。

        Args:
            trading_dates: 交易日列表
        """
        self.trading_dates = sorted(trading_dates)
        if self.start_date:
            self.trading_dates = [d for d in self.trading_dates if d >= self.start_date]
        if self.end_date:
            self.trading_dates = [d for d in self.trading_dates if d <= self.end_date]

    def stop(self) -> None:
        """
        停止回测。
        """
        logger.info("收到停止回测信号")
        self._should_stop = True

    def _sigint_handler(self, signum, frame):
        """
        Ctrl+C 信号处理器，安全停止回测。
        首次 Ctrl+C：设置停止标志，等待当前批次完成后退出。
        再次 Ctrl+C：强制退出。
        """
        if self._should_stop:
            logger.warning("强制退出回测引擎")
            # 恢复原始处理器，避免 os._exit 也被拦截
            if self._sigint_installed:
                signal.signal(signal.SIGINT, self._original_sigint)
            os._exit(1)
        logger.warning("收到 Ctrl+C 终止信号，正在安全停止回测...（再次按 Ctrl+C 强制退出）")
        self._should_stop = True

    def get_current_date(self) -> Optional[date]:
        """
        获取当前回测日期。

        Returns:
            当前日期
        """
        if 0 <= self.current_date_index < len(self.trading_dates):
            return self.trading_dates[self.current_date_index]
        return None

    def _get_stock_price(
        self, 
        stock_code: str, 
        trade_date: date,
        price_type: str = "close"
    ) -> Optional[float]:
        """
        获取股票在指定日期的价格。

        Args:
            stock_code: 股票代码
            trade_date: 交易日期
            price_type: 价格类型 ("close" 收盘价, "open" 开盘价, "high" 最高价, "low" 最低价)

        Returns:
            价格
        """
        try:
            # logger.debug(f"获取股票 {stock_code} 在 {trade_date} 的{price_type}价")
            current_date = self.get_current_date()
            if current_date and trade_date > current_date:
                logger.warning(f"尝试获取未来日期 {trade_date} 的股票价格，当前日期 {current_date}")
                return None

            data = self._data_provider.get_daily_data(stock_code, days=30)
            if isinstance(data, tuple):
                df, _ = data
            else:
                df = data

            if df is None or df.empty:
                logger.warning(f"股票 {stock_code} 数据为空或None")
                return None

            # logger.debug(f"股票 {stock_code} 数据列: {list(df.columns)}")

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                target_df = df[df["date"] == trade_date]
                if not target_df.empty:
                    # 尝试读取价格列（支持大小写）
                    def get_price(col_name):
                        if col_name in target_df.columns:
                            return float(target_df[col_name].iloc[-1])
                        elif col_name.capitalize() in target_df.columns:
                            return float(target_df[col_name.capitalize()].iloc[-1])
                        return None
                    
                    if price_type == "open":
                        return get_price("open")
                    elif price_type == "close":
                        return get_price("close")
                    elif price_type == "high":
                        return get_price("high")
                    elif price_type == "low":
                        return get_price("low")

            # 如果没有找到目标日期的数据，使用最新数据
            def get_latest_price(col_name):
                if col_name in df.columns:
                    return float(df[col_name].iloc[-1])
                elif col_name.capitalize() in df.columns:
                    return float(df[col_name.capitalize()].iloc[-1])
                return None
            
            if price_type == "open":
                return get_latest_price("open")
            elif price_type == "close":
                return get_latest_price("close")
            elif price_type == "high":
                return get_latest_price("high")
            elif price_type == "low":
                return get_latest_price("low")
            
            logger.warning(f"股票 {stock_code} 没有找到{price_type}列")
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"获取股票{price_type}价失败 {stock_code}: {e}")
        return None

    def _check_market_trend(self, current_date: date) -> Optional[bool]:
        """
        检查大盘趋势过滤器。

        当指数收盘价站上其 MA(ma_period) 时返回 True（允许开仓）；
        否则返回 False（禁止开仓）。
        如果过滤器未配置或数据获取失败，返回 None（不拦截）。

        Returns:
            True  - 趋势允许开仓
            False - 趋势禁止开仓
            None  - 无法判断（不拦截）
        """
        if not self.market_trend_filter:
            return None

        index_code = self.market_trend_filter.get("index_code", "000001.SH")
        ma_period = self.market_trend_filter.get("ma_period", 20)

        # 从缓存获取指数数据
        if index_code not in self._index_cache:
            try:
                raw = self._original_data_provider.get_daily_data(index_code, days=ma_period + 30)
                if isinstance(raw, tuple) and len(raw) == 2:
                    df, _ = raw
                else:
                    df = raw
                if df is not None and not df.empty and "date" in df.columns:
                    df = df.copy()
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df = df.sort_values("date").reset_index(drop=True)
                    self._index_cache[index_code] = df
                else:
                    logger.warning("大盘趋势过滤器: 无法获取指数 %s 数据，跳过过滤", index_code)
                    return None
            except Exception as e:
                logger.warning("大盘趋势过滤器: 获取指数 %s 数据失败: %s，跳过过滤", index_code, e)
                return None

        df = self._index_cache[index_code]

        # 找到当前日期在指数数据中的位置
        mask = df["date"] <= current_date
        if not mask.any():
            logger.debug("大盘趋势过滤器: 当前日期 %s 无指数数据，跳过过滤", current_date)
            return None

        df_up_to_date = df[mask].tail(ma_period + 5)
        if len(df_up_to_date) < ma_period:
            logger.debug("大盘趋势过滤器: 指数数据不足 %d 天，跳过过滤", ma_period)
            return None

        # 获取收盘价列
        close_col = "close" if "close" in df_up_to_date.columns else "Close"
        if close_col not in df_up_to_date.columns:
            return None

        closes = df_up_to_date[close_col].astype(float)
        ma_value = closes.tail(ma_period).mean()
        current_close = closes.iloc[-1]

        passed = current_close > ma_value
        logger.info(
            "大盘趋势过滤器: %s 收盘 %.2f, MA%d %.2f, %s → %s",
            index_code, current_close, ma_period, ma_value,
            "站上均线" if passed else "跌破均线",
            "允许开仓" if passed else "禁止开仓",
        )
        return passed

    def _compute_tiandao_indicators(self, current_date: date) -> None:
        """
        为所有持仓股票计算天道指标并缓存。

        在每个交易日的盘中阶段调用一次，后续 exit strategy 的
        check_exits 通过 indicator_provider 回调获取缓存数据。

        Args:
            current_date: 当前交易日
        """
        self._tiandao_cache.clear()
        for stock_code in list(self.portfolio.positions.keys()):
            try:
                data = self._data_provider.get_daily_data(stock_code, days=200)
                if isinstance(data, tuple):
                    df, _ = data
                else:
                    df = data
                if df is None or df.empty:
                    continue

                # 确保有日期列
                if "date" not in df.columns:
                    continue

                # Tiandao 类期望列名为大写首字母：Open, High, Low, Close, Volume
                # 数据提供器可能返回小写列名，需做映射
                column_mapping = {
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume",
                }
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                # 同时处理首字母大写形式
                df = df.rename(columns={
                    k.capitalize(): v for k, v in column_mapping.items()
                    if k.capitalize() in df.columns and k.capitalize() not in column_mapping.values()
                })

                # 计算天道指标
                result = self._tiandao_indicator.calculate(df)
                if result is None or result.empty:
                    continue

                # 找到当前日期对应的指标值
                if "date" in result.columns:
                    result["date"] = pd.to_datetime(result["date"]).dt.date
                    row = result[result["date"] == current_date]
                    if row.empty:
                        continue
                    row = row.iloc[-1]
                else:
                    # 如果没有日期列，使用最后一行
                    row = result.iloc[-1]

                snapshot = TiandaoIndicatorSnapshot(
                    td_jinzuan=float(row.get("td_jinzuan", 0) or 0),
                    td_jinniu=float(row.get("td_jinniu", 0) or 0),
                    td_jinniu2=float(row.get("td_jinniu2", 0) or 0),
                    td_bbi=float(row.get("td_bbi", 0) or 0),
                )
                self._tiandao_cache[stock_code] = snapshot
                logger.debug(
                    "天道指标 [%s]: 金钻=%.2f 金牛=%.2f 金牛2=%.2f BBI=%.2f",
                    stock_code, snapshot.td_jinzuan, snapshot.td_jinniu,
                    snapshot.td_jinniu2, snapshot.td_bbi,
                )
            except Exception as e:
                logger.debug("计算天道指标失败 [%s]: %s", stock_code, e)

    def _setup_indicator_provider(self) -> None:
        """
        为退出策略设置天道指标提供器。

        如果退出策略有 _indicator_provider 属性，则设置一个回调函数，
        从缓存中获取对应股票的天道指标快照。
        """
        if hasattr(self.exit_strategy, "_indicator_provider"):
            cache = self._tiandao_cache

            def indicator_provider(stock_code: str) -> Optional[TiandaoIndicatorSnapshot]:
                return cache.get(stock_code)

            self.exit_strategy._indicator_provider = indicator_provider

    def _process_exit_signal(
        self,
        signal: "ExitSignal",
        current_date: date,
        sold_stocks: Set[str],
        price_type: str,
    ) -> None:
        """
        处理退出信号（卖出或加仓买回）。

        支持三种信号类型：
        - "full": 全部卖出
        - "partial": 部分卖出
        - "rebuy": 加仓买回（将之前减仓的部分买回）

        Args:
            signal: 退出信号
            current_date: 当前交易日
            sold_stocks: 已卖出股票集合（用于跟踪）
            price_type: 价格类型标签（用于日志）
        """
        stock_code = signal.stock_code
        if stock_code in sold_stocks:
            return

        # 处理加仓买回信号
        if signal.exit_type == "rebuy":
            self._process_rebuy_signal(signal, current_date)
            return

        # 处理卖出信号
        pos = self.portfolio.get_position(stock_code)
        if not pos:
            return

        sell_price = signal.sell_price
        if signal.exit_type == "full":
            sell_qty = pos.quantity
        else:
            # partial: exit_ratio 是卖出比例（如 0.5 表示卖出 50%）
            sell_qty = max(100, int(pos.quantity * signal.exit_ratio / 100) * 100)

        if sell_qty <= 0:
            sell_qty = pos.quantity

        logger.info(
            "以%s价执行%s: %s %d股 @ %.2f (%s)",
            price_type,
            "卖出" if signal.exit_type != "rebuy" else "加仓",
            stock_code, sell_qty, sell_price, signal.reason,
        )
        self.place_order(stock_code, OrderType.SELL, sell_qty, price=sell_price, price_type=signal.price_type)

        if signal.exit_type == "full":
            sold_stocks.add(stock_code)
            self.exit_strategy.cleanup_position(stock_code)

    def _process_rebuy_signal(self, signal: "ExitSignal", current_date: date) -> None:
        """
        处理加仓买回信号。

        使用可用现金买入指定数量的股票，不增加新的持仓记录。

        Args:
            signal: 加仓信号
            current_date: 当前交易日
        """
        stock_code = signal.stock_code
        pos = self.portfolio.get_position(stock_code)
        if not pos:
            return

        buy_price = signal.sell_price  # rebuy信号中 sell_price 表示买入价格

        if signal.exit_ratio >= 1.0:
            # 使用全部可用现金加仓买回（天道金牛2保护策略）
            target_qty = int(
                self.portfolio.cash / (buy_price * (1 + self.commission_rate + self.slippage_rate)) / 100
            ) * 100
            if target_qty < 100:
                logger.debug("天道策略 [%s]: 可用现金不足，无法加仓", stock_code)
                return
            logger.info(
                "天道策略加仓买回(全部现金): %s %d股 @ %.2f，可用现金 %.2f (%s)",
                stock_code, target_qty, buy_price, self.portfolio.cash, signal.reason,
            )
        else:
            # 按比例加仓买回（普通策略）
            target_qty = int(pos.quantity * signal.exit_ratio / 100) * 100
            if target_qty < 100:
                return
            # 检查可用资金是否足够
            estimated_cost = target_qty * buy_price * (1 + self.commission_rate + self.slippage_rate)
            if self.portfolio.cash < estimated_cost:
                logger.debug(
                    "天道策略 [%s]: 加仓资金不足，需要 %.2f，可用 %.2f",
                    stock_code, estimated_cost, self.portfolio.cash,
                )
                return
            logger.info(
                "天道策略加仓买回: %s %d股 @ %.2f (%s)",
                stock_code, target_qty, buy_price, signal.reason,
            )

        self.place_order(stock_code, OrderType.BUY, target_qty, price=buy_price, price_type=signal.price_type)

    def _calculate_commission(self, amount: float) -> float:
        """
        计算手续费。

        Args:
            amount: 交易金额

        Returns:
            手续费
        """
        return max(5.0, amount * self.commission_rate)

    def _calculate_slippage(self, price: float, is_buy: bool) -> float:
        """
        计算滑点。

        Args:
            price: 原始价格
            is_buy: 是否买入

        Returns:
            调整后的价格
        """
        if is_buy:
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)

    def place_order(
        self,
        stock_code: str,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        price_type: str = "close",
    ) -> Optional[Trade]:
        """
        下单。

        Args:
            stock_code: 股票代码
            order_type: 订单类型
            quantity: 数量
            price: 可选，指定价格（如果不指定则从数据获取）
            price_type: 价格类型 ("close" 收盘价, "open" 开盘价)

        Returns:
            交易记录
        """
        current_date = self.get_current_date()
        if current_date is None:
            return None

        if price is None:
            price = self._get_stock_price(stock_code, current_date, price_type)
            if price is None:
                return None

        adjusted_price = self._calculate_slippage(price, order_type == OrderType.BUY)
        amount = quantity * adjusted_price
        commission = self._calculate_commission(amount)

        if order_type == OrderType.BUY:
            total_cost = amount + commission
            if self.portfolio.cash < total_cost:
                logger.warning(f"资金不足，无法买入 {stock_code}")
                return None
            
            # 记录入场价并初始化退出策略的状态
            self.exit_strategy.initialize_position(stock_code, adjusted_price, current_date)

        trade = Trade(
            trade_id=f"trade_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            stock_code=stock_code,
            order_type=order_type,
            quantity=quantity,
            price=adjusted_price,
            date=current_date,
            commission=commission,
            slippage=abs(adjusted_price - price) * quantity,
        )

        self.portfolio.execute_trade(trade)
        logger.info(f"执行交易: {trade.order_type.value} {trade.quantity} {trade.stock_code} @ {trade.price:.2f} ({price_type}价)")

        return trade

    def rebalance(self, target_stocks: List[str]) -> None:
        """
        调仓。

        在信号日当天以收盘价直接买入，不再挂单到次日开盘。

        Args:
            target_stocks: 目标持仓股票列表（按优先级排序）
        """
        logger.info(f"=== 开始调仓（当天收盘价买入） ===")

        if not target_stocks:
            logger.info("目标股票列表为空，不进行调仓")
            return

        current_holdings = set(self.portfolio.positions.keys())

        # 记录持仓详细信息（包含成本）
        holding_details = []
        for stock_code in current_holdings:
            pos = self.portfolio.get_position(stock_code)
            if pos:
                holding_details.append(f"{stock_code}(成本:{pos.avg_cost:.2f})")

        logger.info(f"当前持仓: {holding_details} ({len(current_holdings)} 只)")

        # 计算还可以买入多少只新股票
        available_slots = 0
        if self.max_positions is not None:
            available_slots = self.max_positions - len(current_holdings)
            if available_slots <= 0:
                logger.info(f"当前持仓数 {len(current_holdings)} 已达最高持仓数 {self.max_positions}，不买入新股")
                return
            logger.info(f"当前持仓数 {len(current_holdings)}，最高持仓数 {self.max_positions}，可买入 {available_slots} 只新股")
        else:
            logger.info("未设置最高持仓数限制，可以买入所有目标股票")
            available_slots = len(target_stocks)

        # 按排序顺序直接选择前N只股票（已按控盘度优先排序）
        eligible_stocks = [
            stock_code for stock_code in target_stocks
            if stock_code not in current_holdings
        ]

        if not eligible_stocks:
            stocks_to_buy = []
        else:
            # 直接取前 available_slots 只
            stocks_to_buy = eligible_stocks[:available_slots]

            logger.info(f"按控盘度优先排序，选择前 {len(stocks_to_buy)} 只股票: {stocks_to_buy}")

        if not stocks_to_buy:
            logger.info("没有符合条件的新股可买入")
            return

        logger.info(f"准备买入 {len(stocks_to_buy)} 只新股: {stocks_to_buy}")

        # 计算每只股票可分配的资金（预留手续费和滑点）
        current_date = self.get_current_date()

        # 大盘趋势过滤器：指数在 MA20 下方时禁止开仓
        trend_check = self._check_market_trend(current_date)
        if trend_check is False:
            logger.info("大盘趋势过滤器: 指数在均线下方，跳过本次买入")
            return

        available_cash = self.portfolio.cash
        logger.info(f"可用现金: {available_cash:.2f}")
        if available_cash <= 0:
            logger.warning("可用现金不足，无法买入新股")
            return

        # 预留约 0.5% 的交易成本（手续费 + 滑点）
        reserved_cost_pct = 0.005
        equity_per_stock = (available_cash * (1 - reserved_cost_pct)) / len(stocks_to_buy)
        logger.info(f"每只股票分配资金（已预留交易成本）: {equity_per_stock:.2f}")

        for stock_code in stocks_to_buy:
            # 获取当天收盘价
            close_price = self._get_stock_price(stock_code, current_date, "close")
            if close_price is None or close_price <= 0:
                logger.warning(f"股票 {stock_code} 今天收盘价无效，跳过买入")
                continue

            # 计算买入数量（考虑交易成本）
            estimated_price_with_slippage = close_price * (1 + self.slippage_rate)
            effective_equity = equity_per_stock * 0.99
            target_qty = int(effective_equity / estimated_price_with_slippage / 100) * 100
            if target_qty > 0:
                logger.info(f"以收盘价执行买入: {stock_code} {target_qty} 股 @ {close_price:.2f}")
                self.place_order(stock_code, OrderType.BUY, target_qty, price=close_price, price_type="close")
            else:
                logger.warning(f"股票 {stock_code} 目标数量为0，跳过买入")

        logger.info(f"=== 调仓完成 ===")

    def _check_stop_loss_take_profit(
        self, 
        price_type: str = "close"
    ) -> List[Dict[str, Any]]:
        """
        检查持仓股票是否触发止盈或止损（不包含持股超时）。

        Args:
            price_type: 价格类型 ("close" 收盘价, "open" 开盘价)

        Returns:
            需要卖出的股票列表，每个元素包含:
                - stock_code: 股票代码
                - reason: 触发原因 ("stop_loss" 止损, "take_profit" 止盈)
                - trigger_price: 触发价格
                - target_price: 目标价（止损价或止盈价）
        """
        stocks_to_sell = []
        current_date = self.get_current_date()
        
        for stock_code in list(self.portfolio.positions.keys()):
            # T+1规则：买入当天不能卖出
            if stock_code in self._entry_dates and self._entry_dates[stock_code] == current_date:
                continue
            
            price = self._get_stock_price(stock_code, current_date, price_type)
            if price is None:
                continue
            
            # 检查止损
            if stock_code in self._stop_loss_prices and price <= self._stop_loss_prices[stock_code]:
                pos = self.portfolio.get_position(stock_code)
                if pos:
                    sell_price = price if price_type == "open" else self._stop_loss_prices[stock_code]
                    logger.info(f"触发止损 ({price_type}价): {stock_code} 当前价 {price:.2f} <= 止损价 {self._stop_loss_prices[stock_code]:.2f}")
                    stocks_to_sell.append({
                        "stock_code": stock_code,
                        "reason": "stop_loss",
                        "trigger_price": price,
                        "sell_price": sell_price,
                        "price_type": price_type
                    })
                continue
            
            # 检查止盈
            if stock_code in self._take_profit_prices and price >= self._take_profit_prices[stock_code]:
                pos = self.portfolio.get_position(stock_code)
                if pos:
                    sell_price = price if price_type == "open" else self._take_profit_prices[stock_code]
                    logger.info(f"触发止盈 ({price_type}价): {stock_code} 当前价 {price:.2f} >= 止盈价 {self._take_profit_prices[stock_code]:.2f}")
                    stocks_to_sell.append({
                        "stock_code": stock_code,
                        "reason": "take_profit",
                        "trigger_price": price,
                        "sell_price": sell_price,
                        "price_type": price_type
                    })
                continue
        
        return stocks_to_sell

    def _check_intraday_stop_loss_take_profit(self) -> List[Dict[str, Any]]:
        """
        检查盘中最高价和最低价是否触发止盈或止损。

        Returns:
            需要卖出的股票列表
        """
        stocks_to_sell = []
        current_date = self.get_current_date()
        
        for stock_code in list(self.portfolio.positions.keys()):
            # T+1规则：买入当天不能卖出
            if stock_code in self._entry_dates and self._entry_dates[stock_code] == current_date:
                continue
            
            high_price = self._get_stock_price(stock_code, current_date, "high")
            low_price = self._get_stock_price(stock_code, current_date, "low")
            
            if high_price is None or low_price is None:
                continue
            
            # 优先级1：检查止损（最低价触发）
            if stock_code in self._stop_loss_prices and low_price <= self._stop_loss_prices[stock_code]:
                pos = self.portfolio.get_position(stock_code)
                if pos:
                    logger.info(f"触发盘中止损: {stock_code} 最低价 {low_price:.2f} <= 止损价 {self._stop_loss_prices[stock_code]:.2f}")
                    stocks_to_sell.append({
                        "stock_code": stock_code,
                        "reason": "stop_loss_intraday",
                        "trigger_price": low_price,
                        "sell_price": self._stop_loss_prices[stock_code],
                        "price_type": "intraday"
                    })
                continue
            
            # 优先级2：检查止盈（最高价触发）
            if stock_code in self._take_profit_prices and high_price >= self._take_profit_prices[stock_code]:
                pos = self.portfolio.get_position(stock_code)
                if pos:
                    logger.info(f"触发盘中止盈: {stock_code} 最高价 {high_price:.2f} >= 止盈价 {self._take_profit_prices[stock_code]:.2f}")
                    stocks_to_sell.append({
                        "stock_code": stock_code,
                        "reason": "take_profit_intraday",
                        "trigger_price": high_price,
                        "sell_price": self._take_profit_prices[stock_code],
                        "price_type": "intraday"
                    })
                continue
        
        return stocks_to_sell

    def _check_hold_timeout_only(self) -> List[Dict[str, Any]]:
        """
        仅检查持股超时（配置的最长持股时间）。

        Returns:
            需要卖出的股票列表
        """
        stocks_to_sell = []
        current_date = self.get_current_date()
        
        for stock_code in list(self.portfolio.positions.keys()):
            # T+1规则：买入当天不能卖出
            if stock_code in self._entry_dates and self._entry_dates[stock_code] == current_date:
                continue
            
            # 检查持股时间（配置的最长持股时间）
            if stock_code in self._entry_dates:
                entry_date = self._entry_dates[stock_code]
                try:
                    entry_index = self.trading_dates.index(entry_date)
                    current_index = self.current_date_index
                    holding_days = current_index - entry_index
                    
                    if holding_days >= self.max_holding_days:
                        pos = self.portfolio.get_position(stock_code)
                        if pos:
                            # 按收盘价卖出
                            close_price = self._get_stock_price(stock_code, current_date, "close")
                            if close_price is None:
                                continue
                            logger.info(f"触发持股超时: {stock_code} 已持股 {holding_days} 个交易日 (买入日期: {entry_date}, 最长持股: {self.max_holding_days} 天)")
                            stocks_to_sell.append({
                                "stock_code": stock_code,
                                "reason": "hold_timeout",
                                "trigger_price": close_price,
                                "sell_price": close_price,
                                "price_type": "close"
                            })
                except ValueError:
                    logger.warning(f"股票 {stock_code} 的买入日期 {entry_date} 不在交易日列表中")
        
        return stocks_to_sell

    def _step(self) -> bool:
        """
        执行一个时间步。

        执行流程：
        1. 检查开盘价是否触发止盈止损（如果触发，按开盘价卖出）
        2. 检查盘中最高价和最低价，如果触发止盈/止损则按照止盈/止损价卖出
        3. 更新持仓价格到今天收盘价
        4. 检查收盘条件（时间止损、收盘价止损等）
        5. 执行策略选股，当天以收盘价买入

        Returns:
            是否还有下一个时间步
        """
        # 检查是否应该停止
        if self._should_stop:
            logger.info("回测已被用户终止")
            return False

        if self.current_date_index >= len(self.trading_dates):
            return False

        current_date = self.trading_dates[self.current_date_index]
        self._data_provider.set_current_date(current_date)

        logger.info(f"=== 开始日期 {current_date} ===")

        # 步骤 1：检查今天开盘价是否触发止盈止损（如果触发，按开盘价卖出）
        logger.info("检查开盘价是否触发止盈止损...")
        # 先记录所有持仓的开盘价和涨跌幅
        for stock_code in list(self.portfolio.positions.keys()):
            pos = self.portfolio.get_position(stock_code)
            if pos:
                open_price = self._get_stock_price(stock_code, current_date, "open")
                if open_price:
                    change_pct = (open_price - pos.avg_cost) / pos.avg_cost * 100
                    logger.info(f"  {stock_code}: 开盘价={open_price:.2f}, 成本={pos.avg_cost:.2f}, 涨跌={change_pct:+.2f}%")

        # 价格提供函数（绑定当前日期）
        def price_provider(stock_code: str, price_type: str) -> Optional[float]:
            return self._get_stock_price(stock_code, current_date, price_type)

        # 计算天道指标并设置 indicator_provider
        self._compute_tiandao_indicators(current_date)
        self._setup_indicator_provider()

        stocks_to_sell_open = self.exit_strategy.check_exits(
            current_date=current_date,
            positions=self.portfolio.positions,
            price_provider=price_provider,
            trading_dates=self.trading_dates,
            current_date_index=self.current_date_index,
            phase="open",
        )
        sold_stocks = set()
        for signal in stocks_to_sell_open:
            self._process_exit_signal(signal, current_date, sold_stocks, "开盘")

        # 步骤 2：检查盘中最高价和最低价，如果触发止盈/止损则按照止盈/止损价卖出
        logger.info("检查盘中最高价和最低价是否触发止盈止损...")
        # 先记录所有持仓的最高价和最低价
        for stock_code in list(self.portfolio.positions.keys()):
            if stock_code in sold_stocks:
                continue
            pos = self.portfolio.get_position(stock_code)
            if pos:
                high_price = self._get_stock_price(stock_code, current_date, "high")
                low_price = self._get_stock_price(stock_code, current_date, "low")
                if high_price and low_price:
                    logger.info(f"  {stock_code}: 最高价={high_price:.2f}, 最低价={low_price:.2f}, 成本={pos.avg_cost:.2f}")

        stocks_to_sell_intraday = self.exit_strategy.check_exits(
            current_date=current_date,
            positions=self.portfolio.positions,
            price_provider=price_provider,
            trading_dates=self.trading_dates,
            current_date_index=self.current_date_index,
            phase="intraday",
        )
        for signal in stocks_to_sell_intraday:
            self._process_exit_signal(signal, current_date, sold_stocks, "盘中")

        # 步骤 3：更新持仓价格到今天收盘价
        logger.info("更新持仓价格到今天收盘价...")
        for stock_code in self.portfolio.positions.keys():
            price = self._get_stock_price(stock_code, current_date, "close")
            if price:
                self.portfolio.update_position_price(stock_code, price)

        # 步骤 4：检查收盘条件（时间止损、收盘价止损等）
        logger.info("检查收盘条件...")
        # 先记录所有持仓的收盘价和涨跌幅
        for stock_code in list(self.portfolio.positions.keys()):
            if stock_code in sold_stocks:
                continue
            pos = self.portfolio.get_position(stock_code)
            if pos:
                close_price = self._get_stock_price(stock_code, current_date, "close")
                if close_price:
                    change_pct = (close_price - pos.avg_cost) / pos.avg_cost * 100
                    logger.info(f"  {stock_code}: 收盘价={close_price:.2f}, 成本={pos.avg_cost:.2f}, 涨跌={change_pct:+.2f}%")

        stocks_to_sell_close = self.exit_strategy.check_exits(
            current_date=current_date,
            positions=self.portfolio.positions,
            price_provider=price_provider,
            trading_dates=self.trading_dates,
            current_date_index=self.current_date_index,
            phase="close",
        )
        for signal in stocks_to_sell_close:
            self._process_exit_signal(signal, current_date, sold_stocks, "收盘")

        # 步骤 5：判断是否需要跑策略选股
        need_run_strategy = False

        # 条件1：今天有股票被卖出
        if len(sold_stocks) > 0:
            need_run_strategy = True
            logger.info(f"今天有 {len(sold_stocks)} 只股票被卖出，需要跑策略选股")

        # 条件2：持仓未满（还有空仓可以买新股）
        if not need_run_strategy and self.max_positions is not None:
            current_holdings = len(self.portfolio.positions)
            if current_holdings < self.max_positions:
                need_run_strategy = True
                logger.info(f"持仓未满（当前持仓: {current_holdings}, 最高持仓: {self.max_positions}），需要跑策略选股")
        
        # 执行策略选股（如果需要）
        has_strategies = (self._strategy is not None) or (len(self._strategies) > 0)
        if has_strategies and need_run_strategy:
            selected_stocks = []
            from tqdm import tqdm
            
            strategies_to_use = self._strategies if len(self._strategies) > 0 else [self._strategy]
            logger.info(f"开始策略选股，共 {len(self._stock_pool)} 只股票，使用 {len(strategies_to_use)} 个策略...")
            
            # 使用进度条遍历股票池
            selected_with_scores = []
            for stock_code in tqdm(
                self._stock_pool,
                desc=f"策略选股 [{current_date}]",
                unit="只",
                leave=False,
                ncols=100
            ):
                # 检查停止标志
                if self._should_stop:
                    logger.info("策略选股被用户终止")
                    break
                try:
                    # 计算综合得分：所有匹配策略的平均得分
                    total_score = 0.0
                    all_matched = True
                    
                    for strategy in strategies_to_use:
                        # 回测中启用金钻趋势 > 金牛2 条件，过滤趋势已被破坏的标的
                        match = strategy.select(stock_code, require_jinzuan_above_jinniu2=True)
                        if not match or not match.matched:
                            all_matched = False
                            break
                        total_score += match.score
                        # 取策略的控盘度（用于排序）
                        control_degree = match.control_degree
                    
                    # 只有当所有策略都匹配时才选中（与选股页面保持一致）
                    if all_matched:
                        avg_score = total_score / len(strategies_to_use)
                        selected_with_scores.append((stock_code, avg_score, control_degree))
                except Exception as e:
                    logger.warning(f"策略执行失败 {stock_code}: {e}")
            
            # 按控盘度从高到低排序，控盘度相同时按策略得分从高到低排序
            selected_with_scores_sorted = sorted(
                selected_with_scores,
                key=lambda x: (
                    -(x[2] if x[2] is not None else 0),  # 控盘度
                    -x[1],  # 策略得分
                )
            )
            selected_stocks = [stock_code for stock_code, score, cd in selected_with_scores_sorted]
            
            if selected_with_scores_sorted:
                logger.info(f"选中股票的分数（前10只）: {[(stock, round(score, 2), round(cd, 2) if cd else 0) for stock, score, cd in selected_with_scores_sorted[:10]]}")

            logger.info(f"策略选股完成: 选中 {len(selected_stocks)} 只股票")
            if selected_stocks:
                logger.info(f"选中的股票: {selected_stocks[:10]}{'...' if len(selected_stocks) > 10 else ''}")
            
            self.rebalance(selected_stocks)
        elif has_strategies and not need_run_strategy:
            logger.info("无需跑策略选股：没有股票被卖出且持仓已满")

        self.portfolio.record_equity(current_date)
        
        # 计算账户金额信息
        cash = self.portfolio.cash
        position_value = sum(pos.quantity * pos.current_price for pos in self.portfolio.positions.values())
        total_equity = self.portfolio.get_total_equity()
        
        self.current_date_index += 1
        
        logger.info(f"=== 日期 {current_date} 完成 | 现金: {cash:.2f} | 持仓市值: {position_value:.2f} | 总金额: {total_equity:.2f} ===")

        return self.current_date_index < len(self.trading_dates)

    def run(self) -> Portfolio:
        """
        运行完整回测。

        Returns:
            回测后的投资组合
        """
        logger.info("开始回测")
        logger.info(f"初始资金: {self.portfolio.initial_capital:.2f}")
        logger.info(f"回测期间: {self.start_date} 至 {self.end_date}")
        logger.info(f"交易日数: {len(self.trading_dates)}")

        self.current_date_index = 0

        try:
            while self._step():
                if self._should_stop:
                    logger.info("回测已被用户终止")
                    break
        finally:
            # 恢复原始 SIGINT 处理器，避免影响后续操作
            if self._sigint_installed:
                signal.signal(signal.SIGINT, self._original_sigint)

        logger.info("回测完成")
        logger.info(f"最终权益: {self.portfolio.get_total_equity():.2f}")

        return self.portfolio

    def reset(self) -> None:
        """
        重置回测引擎。
        """
        self.portfolio = Portfolio(self.portfolio.initial_capital)
        self.current_date_index = 0
        self._data_provider.clear_cache()
        self.exit_strategy.reset()
