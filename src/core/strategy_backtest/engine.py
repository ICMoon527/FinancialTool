# -*- coding: utf-8 -*-
"""
策略回测核心引擎。

实现基于时间步进的回测框架，包含交易执行逻辑。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .data_access import TimeIsolatedDataProvider

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
            stop_loss_pct: 止损百分比（例如：0.05 表示 5%）
            take_profit_pct: 止盈百分比（例如：0.10 表示 10%）
            max_positions: 最大持仓数量（例如：3 表示最多同时持有3只股票）
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
        self._stock_pool: List[str] = []
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self._entry_prices: Dict[str, float] = {}  # 记录每个股票的入场价
        self._stop_loss_prices: Dict[str, float] = {}  # 记录每个股票的止损价
        self._take_profit_prices: Dict[str, float] = {}  # 记录每个股票的止盈价
        self._pending_orders: List[Dict[str, Any]] = []  # 挂单列表，第二天以开盘价买入
        # 终止标志
        self._should_stop = False

    def set_strategy(self, strategy: Any) -> None:
        """
        设置回测策略。

        Args:
            strategy: 策略对象
        """
        self._strategy = strategy
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
            price_type: 价格类型 ("close" 收盘价, "open" 开盘价)

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
                    if price_type == "open":
                        if "open" in target_df.columns:
                            price = float(target_df["open"].iloc[-1])
                            # logger.debug(f"股票 {stock_code} 在 {trade_date} 的开盘价: {price}")
                            return price
                        elif "Open" in target_df.columns:
                            price = float(target_df["Open"].iloc[-1])
                            # logger.debug(f"股票 {stock_code} 在 {trade_date} 的开盘价: {price}")
                            return price
                    else:
                        if "close" in target_df.columns:
                            price = float(target_df["close"].iloc[-1])
                            # logger.debug(f"股票 {stock_code} 在 {trade_date} 的收盘价: {price}")
                            return price
                        elif "Close" in target_df.columns:
                            price = float(target_df["Close"].iloc[-1])
                            # logger.debug(f"股票 {stock_code} 在 {trade_date} 的收盘价: {price}")
                            return price

            if price_type == "open":
                if "open" in df.columns:
                    price = float(df["open"].iloc[-1])
                    logger.debug(f"股票 {stock_code} 最新开盘价: {price}")
                    return price
                elif "Open" in df.columns:
                    price = float(df["Open"].iloc[-1])
                    logger.debug(f"股票 {stock_code} 最新开盘价: {price}")
                    return price
            else:
                if "close" in df.columns:
                    price = float(df["close"].iloc[-1])
                    logger.debug(f"股票 {stock_code} 最新收盘价: {price}")
                    return price
                elif "Close" in df.columns:
                    price = float(df["Close"].iloc[-1])
                    logger.debug(f"股票 {stock_code} 最新收盘价: {price}")
                    return price
            
            logger.warning(f"股票 {stock_code} 没有找到{price_type}列")
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"获取股票{price_type}价失败 {stock_code}: {e}")
        return None

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
            
            # 记录入场价并计算止盈止损价
            self._entry_prices[stock_code] = adjusted_price
            # 计算止损价
            if self.stop_loss_pct is not None:
                self._stop_loss_prices[stock_code] = adjusted_price * (1 - self.stop_loss_pct)
                logger.info(f"设置止损价: {stock_code} @ {self._stop_loss_prices[stock_code]:.2f} ({self.stop_loss_pct*100:.1f}%)")
            # 计算止盈价
            if self.take_profit_pct is not None:
                self._take_profit_prices[stock_code] = adjusted_price * (1 + self.take_profit_pct)
                logger.info(f"设置止盈价: {stock_code} @ {self._take_profit_prices[stock_code]:.2f} ({self.take_profit_pct*100:.1f}%)")

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

        新逻辑：
        1. 持仓中的股票只有触发止盈或止损时才卖出（在 _check_stop_loss_take_profit 中处理）
        2. 只有当持仓数量 < 最高持仓数时，才可以根据策略的最新排名挂单买入新股
        3. 挂单会在第二天以开盘价成交

        Args:
            target_stocks: 目标持仓股票列表（按优先级排序）
        """
        logger.info(f"=== 开始调仓（挂单） ===")
        
        if not target_stocks:
            logger.info("目标股票列表为空，不进行调仓")
            return

        current_holdings = set(self.portfolio.positions.keys())
        pending_stocks = set([order["stock_code"] for order in self._pending_orders])
        
        # 记录持仓详细信息（包含成本）
        holding_details = []
        for stock_code in current_holdings:
            pos = self.portfolio.get_position(stock_code)
            if pos:
                holding_details.append(f"{stock_code}(成本:{pos.avg_cost:.2f})")
        
        logger.info(f"当前持仓: {holding_details} ({len(current_holdings)} 只)")
        logger.info(f"已挂单: {list(pending_stocks)} ({len(pending_stocks)} 只)")
        
        # 计算还可以买入多少只新股票
        total_holdings = len(current_holdings) + len(pending_stocks)
        available_slots = 0
        if self.max_positions is not None:
            available_slots = self.max_positions - total_holdings
            if available_slots <= 0:
                logger.info(f"当前持仓+挂单数 {total_holdings} 已达最高持仓数 {self.max_positions}，不买入新股")
                return
            logger.info(f"当前持仓+挂单数 {total_holdings}，最高持仓数 {self.max_positions}，可买入 {available_slots} 只新股")
        else:
            logger.info("未设置最高持仓数限制，可以买入所有目标股票")
            available_slots = len(target_stocks)

        # 从目标股票中选择还没有持仓也没有挂单的股票，按优先级排序
        stocks_to_buy = []
        for stock_code in target_stocks:
            if stock_code not in current_holdings and stock_code not in pending_stocks:
                stocks_to_buy.append(stock_code)
                if self.max_positions is not None and len(stocks_to_buy) >= available_slots:
                    break

        if not stocks_to_buy:
            logger.info("没有符合条件的新股可买入")
            return

        logger.info(f"准备挂单买入 {len(stocks_to_buy)} 只新股: {stocks_to_buy}")

        # 计算每只股票可分配的资金
        if stocks_to_buy:
            # 使用可用现金来计算每只股票的分配资金
            available_cash = self.portfolio.cash
            logger.info(f"可用现金: {available_cash:.2f}")
            if available_cash <= 0:
                logger.warning("可用现金不足，无法挂单买入新股")
                return

            equity_per_stock = available_cash / len(stocks_to_buy)
            logger.info(f"每只股票分配资金: {equity_per_stock:.2f}")

            # 先获取今天的收盘价，计算大概需要买入多少股（明天会用开盘价重新计算精确数量）
            for stock_code in stocks_to_buy:
                close_price = self._get_stock_price(stock_code, self.get_current_date(), "close")
                if close_price is None or close_price <= 0:
                    logger.warning(f"股票 {stock_code} 今天收盘价无效，跳过挂单")
                    continue
                
                # 估算数量，但明天会用开盘价重新计算精确数量
                self._pending_orders.append({
                    "stock_code": stock_code,
                    "equity_per_stock": equity_per_stock
                })
                logger.info(f"挂单买入: {stock_code}（明天以开盘价成交）")
        
        logger.info(f"=== 调仓完成 ===")

    def _check_stop_loss_take_profit(
        self, 
        price_type: str = "close"
    ) -> List[Dict[str, Any]]:
        """
        检查持仓股票是否触发止盈或止损。

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
        
        return stocks_to_sell

    def _step(self) -> bool:
        """
        执行一个时间步。

        新的执行流程：
        1. 执行昨天挂的买单（以今天开盘价成交）
        2. 检查今天开盘价是否触发止盈止损（如果触发，按开盘价卖出）
        3. 更新持仓价格到今天收盘价
        4. 检查今天收盘价是否触发止盈止损（如果触发，按止盈止损价卖出）
        5. 执行策略选股，挂买单（明天成交）

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
        
        # 步骤 1：执行昨天挂的买单（以今天开盘价成交）
        if self._pending_orders:
            logger.info(f"执行昨天挂的 {len(self._pending_orders)} 个买单...")
            pending_orders = self._pending_orders.copy()
            self._pending_orders.clear()
            
            # 计算当前持仓数量，确保不超过最高持仓数限制
            current_holdings = len(self.portfolio.positions)
            available_slots = None
            if self.max_positions is not None:
                available_slots = self.max_positions - current_holdings
                if available_slots <= 0:
                    logger.info(f"当前持仓数 {current_holdings} 已达最高持仓数 {self.max_positions}，不执行任何挂单")
                    available_slots = 0
            
            executed_count = 0
            for order in pending_orders:
                # 检查是否还可以继续买入
                if self.max_positions is not None and executed_count >= available_slots:
                    logger.info(f"已执行 {executed_count} 个买单，达到最高持仓数限制，跳过剩余 {len(pending_orders) - executed_count} 个挂单")
                    break
                
                stock_code = order["stock_code"]
                equity_per_stock = order["equity_per_stock"]
                
                # 如果已经持仓，跳过
                if stock_code in self.portfolio.positions:
                    logger.info(f"股票 {stock_code} 已持仓，跳过买入")
                    continue
                
                # 获取今天的开盘价
                open_price = self._get_stock_price(stock_code, current_date, "open")
                if open_price is None or open_price <= 0:
                    logger.warning(f"股票 {stock_code} 今天开盘价无效，跳过买入")
                    continue
                
                # 计算买入数量
                target_qty = int(equity_per_stock / open_price / 100) * 100
                if target_qty > 0:
                    logger.info(f"以开盘价执行买入: {stock_code} {target_qty} 股 @ {open_price:.2f}")
                    self.place_order(stock_code, OrderType.BUY, target_qty, price=open_price, price_type="open")
                    executed_count += 1
                else:
                    logger.warning(f"股票 {stock_code} 目标数量为0，跳过买入")
        
        # 步骤 2：检查今天开盘价是否触发止盈止损（如果触发，按开盘价卖出）
        logger.info("检查开盘价是否触发止盈止损...")
        # 先记录所有持仓的开盘价和涨跌幅
        for stock_code in list(self.portfolio.positions.keys()):
            pos = self.portfolio.get_position(stock_code)
            if pos:
                open_price = self._get_stock_price(stock_code, current_date, "open")
                if open_price:
                    change_pct = (open_price - pos.avg_cost) / pos.avg_cost * 100
                    logger.info(f"  {stock_code}: 开盘价={open_price:.2f}, 成本={pos.avg_cost:.2f}, 涨跌={change_pct:+.2f}%")
        
        stocks_to_sell_open = self._check_stop_loss_take_profit(price_type="open")
        for sell_info in stocks_to_sell_open:
            stock_code = sell_info["stock_code"]
            sell_price = sell_info["sell_price"]
            pos = self.portfolio.get_position(stock_code)
            if pos:
                logger.info(f"以开盘价执行止盈止损卖出: {stock_code} {pos.quantity} 股 @ {sell_price:.2f}")
                self.place_order(stock_code, OrderType.SELL, pos.quantity, price=sell_price, price_type="open")
                # 清理记录
                if stock_code in self._entry_prices:
                    del self._entry_prices[stock_code]
                if stock_code in self._stop_loss_prices:
                    del self._stop_loss_prices[stock_code]
                if stock_code in self._take_profit_prices:
                    del self._take_profit_prices[stock_code]
        
        # 步骤 3：更新持仓价格到今天收盘价
        logger.info("更新持仓价格到今天收盘价...")
        for stock_code in self.portfolio.positions.keys():
            price = self._get_stock_price(stock_code, current_date, "close")
            if price:
                self.portfolio.update_position_price(stock_code, price)
        
        # 步骤 4：检查今天收盘价是否触发止盈止损（如果触发，按止盈止损价卖出）
        logger.info("检查收盘价是否触发止盈止损...")
        # 先记录所有持仓的收盘价和涨跌幅
        for stock_code in list(self.portfolio.positions.keys()):
            pos = self.portfolio.get_position(stock_code)
            if pos:
                close_price = self._get_stock_price(stock_code, current_date, "close")
                if close_price:
                    change_pct = (close_price - pos.avg_cost) / pos.avg_cost * 100
                    logger.info(f"  {stock_code}: 收盘价={close_price:.2f}, 成本={pos.avg_cost:.2f}, 涨跌={change_pct:+.2f}%")
        
        stocks_to_sell_close = self._check_stop_loss_take_profit(price_type="close")
        for sell_info in stocks_to_sell_close:
            stock_code = sell_info["stock_code"]
            sell_price = sell_info["sell_price"]
            pos = self.portfolio.get_position(stock_code)
            if pos:
                logger.info(f"以止盈止损价执行卖出: {stock_code} {pos.quantity} 股 @ {sell_price:.2f}")
                self.place_order(stock_code, OrderType.SELL, pos.quantity, price=sell_price, price_type="close")
                # 清理记录
                if stock_code in self._entry_prices:
                    del self._entry_prices[stock_code]
                if stock_code in self._stop_loss_prices:
                    del self._stop_loss_prices[stock_code]
                if stock_code in self._take_profit_prices:
                    del self._take_profit_prices[stock_code]
        
        # 步骤 5：执行策略选股，挂买单（明天成交）
        if self._strategy:
            selected_stocks = []
            from tqdm import tqdm
            
            logger.info(f"开始策略选股，共 {len(self._stock_pool)} 只股票...")
            
            # 使用进度条遍历股票池
            selected_with_scores = []
            for stock_code in tqdm(
                self._stock_pool,
                desc=f"策略选股 [{current_date}]",
                unit="只",
                leave=False,
                ncols=100
            ):
                try:
                    match = self._strategy.select(stock_code)
                    if match and match.matched:
                        selected_with_scores.append((stock_code, match.score))
                except Exception as e:
                    logger.warning(f"策略执行失败 {stock_code}: {e}")
            
            # 按分数从高到低排序
            selected_with_scores_sorted = sorted(selected_with_scores, key=lambda x: x[1], reverse=True)
            selected_stocks = [stock_code for stock_code, score in selected_with_scores_sorted]
            
            if selected_with_scores_sorted:
                logger.info(f"选中股票的分数（前10只）: {[(stock, round(score, 2)) for stock, score in selected_with_scores_sorted[:10]]}")

            logger.info(f"策略选股完成: 选中 {len(selected_stocks)} 只股票")
            if selected_stocks:
                logger.info(f"选中的股票: {selected_stocks[:10]}{'...' if len(selected_stocks) > 10 else ''}")
            
            self.rebalance(selected_stocks)

        self.portfolio.record_equity(current_date)
        self.current_date_index += 1
        
        logger.info(f"=== 日期 {current_date} 完成 ===")

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

        while self._step():
            pass

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
        self._entry_prices.clear()
        self._stop_loss_prices.clear()
        self._take_profit_prices.clear()
        self._pending_orders.clear()
