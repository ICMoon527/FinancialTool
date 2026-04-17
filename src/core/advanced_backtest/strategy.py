# -*- coding: utf-8 -*-
"""
===================================
多因子策略模块
===================================

提供多因子选股和交易策略实现。
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import numpy as np

from .factor import Factor, FactorCombination, FactorConfig, FactorDirection

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """调仓频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class PositionSizingMethod(Enum):
    """仓位管理方法"""
    EQUAL_WEIGHT = "equal_weight"  # 等权
    RISK_PARITY = "risk_parity"  # 风险平价
    MARKET_CAP = "market_cap"  # 市值加权
    FIXED_NUMBER = "fixed_number"  # 固定数量


@dataclass
class MultiFactorStrategyConfig:
    """多因子策略配置"""
    name: str = "multi_factor_strategy"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    position_sizing: PositionSizingMethod = PositionSizingMethod.EQUAL_WEIGHT
    max_positions: int = 20  # 最大持仓数量
    min_factor_score: float = 0.0  # 最小因子分数
    turnover_limit: float = 0.5  # 换手率限制（0-1）
    transaction_cost: float = 0.001  # 交易成本（单边）
    slippage: float = 0.001  # 滑点
    initial_capital: float = 1000000.0  # 初始资金
    description: str = ""


class MultiFactorStrategy:
    """
    多因子策略
    
    基于多因子组合的选股和交易策略。
    """
    
    def __init__(
        self,
        factor_combination: FactorCombination,
        config: Optional[MultiFactorStrategyConfig] = None,
    ):
        self.factor_combination = factor_combination
        self.config = config or MultiFactorStrategyConfig()
        
        # 状态
        self.current_positions: Dict[str, float] = {}  # 股票代码 -> 持仓权重
        self.current_holdings: Dict[str, int] = {}  # 股票代码 -> 持仓数量
        self.cash: float = self.config.initial_capital
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
    
    def calculate_factor_scores(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        计算所有股票的因子分数
        
        Args:
            data: 股票数据字典 {股票代码: DataFrame}
            
        Returns:
            因子分数 Series
        """
        all_scores = {}
        
        for stock_code, stock_data in data.items():
            try:
                score = self.factor_combination.combine(stock_data)
                if not score.empty:
                    # 使用最新的因子分数
                    all_scores[stock_code] = score.iloc[-1]
            except Exception as e:
                logger.warning(f"计算 {stock_code} 因子分数失败: {e}")
                continue
        
        return pd.Series(all_scores)
    
    def select_stocks(self, factor_scores: pd.Series) -> List[str]:
        """
        根据因子分数选择股票
        
        Args:
            factor_scores: 因子分数 Series
            
        Returns:
            选中的股票代码列表
        """
        # 过滤掉分数低于阈值的股票
        valid_scores = factor_scores[factor_scores >= self.config.min_factor_score]
        
        if valid_scores.empty:
            return []
        
        # 排序并选择前 N 只股票
        sorted_stocks = valid_scores.sort_values(ascending=False)
        selected = sorted_stocks.head(self.config.max_positions).index.tolist()
        
        return selected
    
    def calculate_position_weights(
        self,
        selected_stocks: List[str],
        factor_scores: pd.Series,
        market_caps: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        计算持仓权重
        
        Args:
            selected_stocks: 选中的股票列表
            factor_scores: 因子分数
            market_caps: 市值字典（可选）
            volatilities: 波动率字典（可选）
            
        Returns:
            股票代码 -> 权重的字典
        """
        if not selected_stocks:
            return {}
        
        weights = {}
        
        if self.config.position_sizing == PositionSizingMethod.EQUAL_WEIGHT:
            # 等权
            weight = 1.0 / len(selected_stocks)
            for stock in selected_stocks:
                weights[stock] = weight
        
        elif self.config.position_sizing == PositionSizingMethod.RISK_PARITY:
            # 风险平价（基于波动率）
            if volatilities is None:
                # 如果没有波动率数据，回退到等权
                weight = 1.0 / len(selected_stocks)
                for stock in selected_stocks:
                    weights[stock] = weight
            else:
                # 按波动率倒数分配权重
                inv_vols = {}
                for stock in selected_stocks:
                    vol = volatilities.get(stock, 0.01)
                    inv_vols[stock] = 1.0 / vol
                
                total_inv_vol = sum(inv_vols.values())
                for stock in selected_stocks:
                    weights[stock] = inv_vols[stock] / total_inv_vol
        
        elif self.config.position_sizing == PositionSizingMethod.MARKET_CAP:
            # 市值加权
            if market_caps is None:
                # 如果没有市值数据，回退到等权
                weight = 1.0 / len(selected_stocks)
                for stock in selected_stocks:
                    weights[stock] = weight
            else:
                total_cap = sum(market_caps.get(stock, 0) for stock in selected_stocks)
                if total_cap > 0:
                    for stock in selected_stocks:
                        weights[stock] = market_caps.get(stock, 0) / total_cap
                else:
                    weight = 1.0 / len(selected_stocks)
                    for stock in selected_stocks:
                        weights[stock] = weight
        
        elif self.config.position_sizing == PositionSizingMethod.FIXED_NUMBER:
            # 固定数量（等权）
            weight = 1.0 / len(selected_stocks)
            for stock in selected_stocks:
                weights[stock] = weight
        
        return weights
    
    def generate_trades(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        current_date: date,
    ) -> List[Dict]:
        """
        生成交易指令
        
        Args:
            target_weights: 目标权重
            current_prices: 当前价格
            current_date: 当前日期
            
        Returns:
            交易指令列表
        """
        trades = []
        
        # 计算当前组合总价值
        current_portfolio_value = self.cash
        for stock_code, quantity in self.current_holdings.items():
            price = current_prices.get(stock_code, 0)
            current_portfolio_value += quantity * price
        
        # 计算目标持仓
        for stock_code, target_weight in target_weights.items():
            target_value = current_portfolio_value * target_weight
            current_price = current_prices.get(stock_code, 0)
            
            if current_price <= 0:
                continue
            
            target_quantity = int(target_value / current_price)
            current_quantity = self.current_holdings.get(stock_code, 0)
            quantity_diff = target_quantity - current_quantity
            
            if abs(quantity_diff) > 0:
                trade = {
                    "date": current_date,
                    "stock_code": stock_code,
                    "action": "buy" if quantity_diff > 0 else "sell",
                    "quantity": abs(quantity_diff),
                    "price": current_price,
                }
                trades.append(trade)
        
        # 卖出不在目标列表中的持仓
        for stock_code in list(self.current_holdings.keys()):
            if stock_code not in target_weights:
                current_price = current_prices.get(stock_code, 0)
                if current_price > 0:
                    trade = {
                        "date": current_date,
                        "stock_code": stock_code,
                        "action": "sell",
                        "quantity": self.current_holdings[stock_code],
                        "price": current_price,
                    }
                    trades.append(trade)
        
        return trades
    
    def execute_trades(
        self,
        trades: List[Dict],
        current_prices: Dict[str, float],
    ) -> None:
        """
        执行交易
        
        Args:
            trades: 交易指令列表
            current_prices: 当前价格
        """
        for trade in trades:
            stock_code = trade["stock_code"]
            action = trade["action"]
            quantity = trade["quantity"]
            price = trade["price"]
            
            # 计算交易成本
            transaction_value = quantity * price
            cost = transaction_value * (self.config.transaction_cost + self.config.slippage)
            
            if action == "buy":
                # 买入
                total_cost = transaction_value + cost
                if self.cash >= total_cost:
                    self.current_holdings[stock_code] = self.current_holdings.get(stock_code, 0) + quantity
                    self.cash -= total_cost
                    self.trade_history.append(trade)
            
            elif action == "sell":
                # 卖出
                if stock_code in self.current_holdings and self.current_holdings[stock_code] >= quantity:
                    self.current_holdings[stock_code] -= quantity
                    if self.current_holdings[stock_code] == 0:
                        del self.current_holdings[stock_code]
                    self.cash += (transaction_value - cost)
                    self.trade_history.append(trade)
    
    def update_portfolio(
        self,
        current_prices: Dict[str, float],
        current_date: date,
    ) -> None:
        """
        更新组合状态
        
        Args:
            current_prices: 当前价格
            current_date: 当前日期
        """
        # 计算当前组合价值
        portfolio_value = self.cash
        for stock_code, quantity in self.current_holdings.items():
            price = current_prices.get(stock_code, 0)
            portfolio_value += quantity * price
        
        # 记录组合历史
        self.portfolio_history.append({
            "date": current_date,
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "holdings": dict(self.current_holdings),
        })
    
    def rebalance(
        self,
        data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        current_date: date,
        market_caps: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        执行调仓
        
        Args:
            data: 股票数据
            current_prices: 当前价格
            current_date: 当前日期
            market_caps: 市值（可选）
            volatilities: 波动率（可选）
            
        Returns:
            执行的交易列表
        """
        # 1. 计算因子分数
        factor_scores = self.calculate_factor_scores(data)
        
        # 2. 选择股票
        selected_stocks = self.select_stocks(factor_scores)
        
        # 3. 计算目标权重
        target_weights = self.calculate_position_weights(
            selected_stocks, factor_scores, market_caps, volatilities
        )
        
        # 4. 生成交易
        trades = self.generate_trades(target_weights, current_prices, current_date)
        
        # 5. 执行交易
        self.execute_trades(trades, current_prices)
        
        # 6. 更新组合
        self.update_portfolio(current_prices, current_date)
        
        return trades
