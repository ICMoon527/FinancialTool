# -*- coding: utf-8 -*-
"""
===================================
风险监控与风险管理模块
===================================

职责：
1. 持仓管理
2. 止损止盈策略
3. 风险计算引擎（VaR、CVaR、最大回撤
4. 实时风险监控
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionStatus(Enum):
    """持仓状态"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"


@dataclass
class Position:
    """持仓信息"""
    position_id: str
    stock_code: str
    stock_name: str
    quantity: int
    entry_price: float
    current_price: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    entry_date: date = field(default_factory=date.today)
    close_date: Optional[date] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_runup: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """未实现盈亏"""
        if self.status != PositionStatus.OPEN:
            return self.realized_pnl
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """未实现盈亏百分比"""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    @property
    def position_value(self) -> float:
        """持仓市值"""
        return self.current_price * self.quantity
    
    @property
    def cost_basis(self) -> float:
        """成本基准"""
        return self.entry_price * self.quantity


@dataclass
class PortfolioMetrics:
    """组合风险指标"""
    total_value: float = 0.0
    total_cost: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    cash_available: float = 0.0
    position_count: int = 0
    open_position_count: int = 0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # 95% 置信度 VaR
    var_99: float = 0.0  # 99% 置信度 VaR
    cvar_95: float = 0.0  # 95% 置信度 CVaR
    leverage_ratio: float = 0.0  # 杠杆率
    concentration_ratio: float = 0.0  # 集中度比率
    risk_level: RiskLevel = RiskLevel.LOW


class RiskAlert:
    """风险预警"""
    def __init__(
        self,
        alert_id: str,
        alert_type: str,
        risk_level: RiskLevel,
        message: str,
        stock_code: Optional[str] = None,
        stock_name: Optional[str] = None,
        position_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        trigger_time: datetime = field(default_factory=datetime.now),
    ):
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.risk_level = risk_level
        self.message = message
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.position_id = position_id
        self.metrics = metrics or {}
        self.trigger_time = trigger_time


class RiskManager:
    """
    风险管理器
    
    职责：
    1. 持仓管理
    2. 止损止盈监控
    3. 组合风险计算
    4. 风险预警生成
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        max_position_pct: float = 0.2,  # 单只股票最大持仓比例
        max_total_positions: int = 10,  # 最大持仓数量
        var_confidence_levels: List[float] = None,
        var_window_days: int = 252,  # VaR 计算窗口（交易日）
    ):
        self.initial_capital = initial_capital
        self.cash_available = initial_capital
        self.max_position_pct = max_position_pct
        self.max_total_positions = max_total_positions
        self.var_confidence_levels = var_confidence_levels or [0.95, 0.99]
        self.var_window_days = var_window_days
        
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.position_history: List[Position] = []
        self.portfolio_history: List[Tuple[datetime, float]] = []  # 组合历史净值
        self.risk_alerts: List[RiskAlert] = []
        self.equity_curve: List[float] = []
        
        self._position_counter = 0
        
    def open_position(
        self,
        stock_code: str,
        stock_name: str,
        quantity: int,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Optional[Position]:
        """
        开仓
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            quantity: 数量
            entry_price: 入场价格
            stop_loss_price: 止损价
            take_profit_price: 止盈价
            
        Returns:
            Position: 持仓对象
        """
        # 检查是否超过最大持仓数量
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        if len(open_positions) >= self.max_total_positions:
            logger.warning(f"Cannot open position: max positions reached ({self.max_total_positions})")
            return None
        
        # 检查单只股票持仓比例
        position_value = quantity * entry_price
        max_position_value = self.initial_capital * self.max_position_pct
        if position_value > max_position_value:
            logger.warning(f"Cannot open position: exceeds max position percentage")
            return None
        
        # 检查现金是否足够
        if position_value > self.cash_available:
            logger.warning(f"Cannot open position: insufficient cash")
            return None
        
        # 创建持仓
        self._position_counter += 1
        position_id = f"POS_{self._position_counter}"
        
        position = Position(
            position_id=position_id,
            stock_code=stock_code,
            stock_name=stock_name,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            entry_date=date.today(),
        )
        
        self.positions[position_id] = position
        self.cash_available -= position_value
        
        logger.info(f"Opened position: {position_id} - {stock_code} {quantity}股 @ {entry_price}")
        
        return position
    
    def close_position(
        self,
        position_id: str,
        close_price: float,
        close_reason: str = "manual",
    ) -> Optional[Position]:
        """
        平仓
        
        Args:
            position_id: 持仓 ID
            close_price: 平仓价格
            close_reason: 平仓原因
            
        Returns:
            Position: 持仓对象
        """
        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return None
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            logger.warning(f"Position not open: {position_id}")
            return None
        
        # 更新持仓
        position.close_date = date.today()
        position.realized_pnl = (close_price - position.entry_price) * position.quantity
        position.realized_pnl_pct = (close_price - position.entry_price) / position.entry_price * 100
        position.current_price = close_price
        
        # 更新状态
        if close_reason == "stop_loss":
            position.status = PositionStatus.STOPPED_OUT
        elif close_reason == "take_profit":
            position.status = PositionStatus.TAKE_PROFIT
        else:
            position.status = PositionStatus.CLOSED
        
        # 返还现金
        self.cash_available += close_price * position.quantity
        
        # 移动到历史记录
        self.position_history.append(position)
        
        logger.info(f"Closed position: {position_id} - {position.stock_code} @ {close_price}, PnL: {position.realized_pnl:.2f}")
        
        return position
    
    def update_position_price(self, position_id: str, current_price: float) -> Optional[RiskAlert]:
        """
        更新持仓价格并检查止损止盈
        
        Args:
            position_id: 持仓 ID
            current_price: 当前价格
            
        Returns:
            RiskAlert: 风险预警（如果触发）
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            return None
        
        old_price = position.current_price
        position.current_price = current_price
        
        # 更新最大回撤和最大涨幅
        if position.entry_price > 0:
            current_pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            if current_pnl_pct < position.max_drawdown:
                position.max_drawdown = current_pnl_pct
            if current_pnl_pct > position.max_runup:
                position.max_runup = current_pnl_pct
        
        # 检查止损
        if position.stop_loss_price is not None and current_price <= position.stop_loss_price:
            alert = RiskAlert(
                alert_id=f"STOP_LOSS_{position_id}",
                alert_type="stop_loss_triggered",
                risk_level=RiskLevel.CRITICAL,
                message=f"{position.stock_name}({position.stock_code}) 触发止损: {current_price:.2f} <= {position.stop_loss_price:.2f}",
                stock_code=position.stock_code,
                stock_name=position.stock_name,
                position_id=position_id,
                metrics={
                    "current_price": current_price,
                    "stop_loss_price": position.stop_loss_price,
                    "pnl_pct": position.unrealized_pnl_pct,
                },
            )
            self.risk_alerts.append(alert)
            logger.warning(alert.message)
            return alert
        
        # 检查止盈
        if position.take_profit_price is not None and current_price >= position.take_profit_price:
            alert = RiskAlert(
                alert_id=f"TAKE_PROFIT_{position_id}",
                alert_type="take_profit_triggered",
                risk_level=RiskLevel.HIGH,
                message=f"{position.stock_name}({position.stock_code}) 触发止盈: {current_price:.2f} >= {position.take_profit_price:.2f}",
                stock_code=position.stock_code,
                stock_name=position.stock_name,
                position_id=position_id,
                metrics={
                    "current_price": current_price,
                    "take_profit_price": position.take_profit_price,
                    "pnl_pct": position.unrealized_pnl_pct,
                },
            )
            self.risk_alerts.append(alert)
            logger.info(alert.message)
            return alert
        
        return None
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """
        计算组合风险指标
        
        Returns:
            PortfolioMetrics: 组合风险指标
        """
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        
        total_value = self.cash_available
        total_cost = 0.0
        total_pnl = 0.0
        
        for position in open_positions:
            total_value += position.position_value
            total_cost += position.cost_basis
            total_pnl += position.unrealized_pnl
        
        total_pnl_pct = 0.0
        if total_cost > 0:
            total_pnl_pct = total_pnl / total_cost * 100
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # 计算 VaR（简化版本，基于历史数据）
        var_95, var_99 = self._calculate_var()
        
        # 计算集中度（前3只股票占比）
        concentration_ratio = self._calculate_concentration_ratio(open_positions)
        
        # 计算杠杆率
        leverage_ratio = total_value / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # 确定风险级别
        risk_level = self._determine_risk_level(
            total_pnl_pct, max_drawdown, var_95, concentration_ratio, leverage_ratio
        )
        
        return PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            cash_available=self.cash_available,
            position_count=len(self.positions),
            open_position_count=len(open_positions),
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            cvar_95=var_95 * 1.2,  # 简化 CVaR 估算
            leverage_ratio=leverage_ratio,
            concentration_ratio=concentration_ratio,
            risk_level=risk_level,
        )
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤（简化版本）"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # 也考虑当前持仓的最大回撤
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        for position in open_positions:
            if abs(position.max_drawdown) > max_dd:
                max_dd = abs(position.max_drawdown)
        
        return max_dd
    
    def _calculate_var(self) -> Tuple[float, float]:
        """
        计算 VaR（简化版本）
        
        使用简单的蒙特卡洛模拟或历史模拟
        这里使用简化的正态分布假设
        """
        # 简化估算
        # 假设日收益率标准差为 2%
        daily_volatility = 0.02
        
        # 95% 置信度 VaR = -1.645 * 标准差 * sqrt(窗口)
        var_95 = -1.645 * daily_volatility * np.sqrt(self.var_window_days / 252) * 100
        
        # 99% 置信度 VaR = -2.326 * 标准差 * sqrt(窗口)
        var_99 = -2.326 * daily_volatility * np.sqrt(self.var_window_days / 252) * 100
        
        return abs(var_95), abs(var_99)
    
    def _calculate_concentration_ratio(self, positions: List[Position]) -> float:
        """计算集中度比率（前3只股票占比）"""
        if not positions:
            return 0.0
        
        total_value = sum(p.position_value for p in positions)
        if total_value <= 0:
            return 0.0
        
        # 按市值排序
        sorted_positions = sorted(positions, key=lambda p: p.position_value, reverse=True)
        
        # 前3只股票占比
        top3_value = sum(p.position_value for p in sorted_positions[:3])
        
        return top3_value / total_value * 100
    
    def _determine_risk_level(
        self,
        total_pnl_pct: float,
        max_drawdown: float,
        var_95: float,
        concentration_ratio: float,
        leverage_ratio: float,
    ) -> RiskLevel:
        """确定风险级别"""
        high_risk_count = 0
        
        # 检查最大回撤
        if max_drawdown > 20:
            high_risk_count += 2
        elif max_drawdown > 10:
            high_risk_count += 1
        
        # 检查 VaR
        if var_95 > 10:
            high_risk_count += 1
        
        # 检查集中度
        if concentration_ratio > 60:
            high_risk_count += 1
        
        # 检查杠杆
        if leverage_ratio > 1.5:
            high_risk_count += 2
        elif leverage_ratio > 1.2:
            high_risk_count += 1
        
        if high_risk_count >= 4:
            return RiskLevel.CRITICAL
        elif high_risk_count >= 2:
            return RiskLevel.HIGH
        elif high_risk_count >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_portfolio_risk(self) -> List[RiskAlert]:
        """
        检查组合风险
        
        Returns:
            List[RiskAlert]: 风险预警列表
        """
        alerts = []
        metrics = self.calculate_portfolio_metrics()
        
        # 检查最大回撤
        if metrics.max_drawdown > 10:
            alert = RiskAlert(
                alert_id=f"PORTFOLIO_DRAWDOWN_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                alert_type="portfolio_drawdown_high",
                risk_level=RiskLevel.HIGH if metrics.max_drawdown > 20 else RiskLevel.MEDIUM,
                message=f"组合最大回撤过高: {metrics.max_drawdown:.2f}%",
                metrics={"max_drawdown": metrics.max_drawdown},
            )
            alerts.append(alert)
            self.risk_alerts.append(alert)
        
        # 检查 VaR
        if metrics.var_95 > 10:
            alert = RiskAlert(
                alert_id=f"PORTFOLIO_VAR_HIGH_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                alert_type="portfolio_var_high",
                risk_level=RiskLevel.MEDIUM,
                message=f"组合 VaR(95%)过高: {metrics.var_95:.2f}%",
                metrics={"var_95": metrics.var_95},
            )
            alerts.append(alert)
            self.risk_alerts.append(alert)
        
        # 检查集中度
        if metrics.concentration_ratio > 60:
            alert = RiskAlert(
                alert_id=f"PORTFOLIO_CONCENTRATION_HIGH_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                alert_type="portfolio_concentration_high",
                risk_level=RiskLevel.MEDIUM,
                message=f"组合集中度过高: {metrics.concentration_ratio:.2f}%",
                metrics={"concentration_ratio": metrics.concentration_ratio},
            )
            alerts.append(alert)
            self.risk_alerts.append(alert)
        
        # 检查杠杆率
        if metrics.leverage_ratio > 1.2:
            alert = RiskAlert(
                alert_id=f"PORTFOLIO_LEVERAGE_HIGH_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                alert_type="portfolio_leverage_high",
                risk_level=RiskLevel.HIGH if metrics.leverage_ratio > 1.5 else RiskLevel.MEDIUM,
                message=f"组合杠杆率过高: {metrics.leverage_ratio:.2f}x",
                metrics={"leverage_ratio": metrics.leverage_ratio},
            )
            alerts.append(alert)
            self.risk_alerts.append(alert)
        
        return alerts
    
    def get_open_positions(self) -> List[Position]:
        """获取所有开仓持仓"""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """获取指定持仓"""
        return self.positions.get(position_id)
    
    def get_recent_alerts(self, max_count: int = 50) -> List[RiskAlert]:
        """获取最近的风险预警"""
        return sorted(self.risk_alerts, key=lambda a: a.trigger_time, reverse=True)[:max_count]
    
    def record_equity_snapshot(self):
        """记录当前组合净值快照"""
        metrics = self.calculate_portfolio_metrics()
        self.equity_curve.append(metrics.total_value)
        self.portfolio_history.append((datetime.now(), metrics.total_value))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于持久化）"""
        return {
            "initial_capital": self.initial_capital,
            "cash_available": self.cash_available,
            "positions": {pid: pos.__dict__ for pid, pos in self.positions.items()},
            "position_history": [pos.__dict__ for pos in self.position_history],
            "equity_curve": self.equity_curve,
        }
