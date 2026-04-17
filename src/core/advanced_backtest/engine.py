# -*- coding: utf-8 -*-
"""
===================================
高级回测引擎模块
===================================

提供多因子回测、绩效评估等功能。
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import numpy as np

from .strategy import MultiFactorStrategy, MultiFactorStrategyConfig
from .factor import FactorCombination

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: date
    end_date: date
    initial_capital: float = 1000000.0
    benchmark_code: Optional[str] = None  # 基准指数代码
    rebalance_on_start: bool = True
    warmup_period: int = 60  # 预热期（交易日）
    name: str = "advanced_backtest"
    description: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    strategy_config: MultiFactorStrategyConfig
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Dict] = field(default_factory=list)
    portfolio_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    benchmark_curve: Optional[pd.DataFrame] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class AdvancedBacktestEngine:
    """
    高级回测引擎
    
    支持多因子策略回测、绩效评估等功能。
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.result = BacktestResult(
            config=config,
            strategy_config=MultiFactorStrategyConfig(initial_capital=config.initial_capital),
            start_date=config.start_date,
            end_date=config.end_date,
        )
    
    def run(
        self,
        strategy: MultiFactorStrategy,
        data_provider: Any,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            strategy: 多因子策略
            data_provider: 数据提供者
            benchmark_data: 基准数据（可选）
            
        Returns:
            回测结果
        """
        logger.info(f"开始回测: {self.config.start_date} 至 {self.config.end_date}")
        
        # 获取交易日历
        trading_days = self._get_trading_days(data_provider)
        
        # 预热期
        warmup_days = trading_days[:self.config.warmup_period]
        actual_start = trading_days[self.config.warmup_period] if len(trading_days) > self.config.warmup_period else self.config.start_date
        
        # 主回测循环
        current_day_index = self.config.warmup_period
        
        for i, current_date in enumerate(trading_days[self.config.warmup_period:]):
            if current_date < self.config.start_date:
                continue
            if current_date > self.config.end_date:
                break
            
            logger.debug(f"回测日期: {current_date}")
            
            # 1. 获取当前数据
            current_data = self._get_data_for_date(data_provider, current_date, trading_days, i + self.config.warmup_period)
            
            # 2. 获取当前价格
            current_prices = self._get_current_prices(data_provider, current_date)
            
            # 3. 检查是否需要调仓
            if self._should_rebalance(current_date, i, trading_days):
                logger.info(f"调仓日期: {current_date}")
                
                # 获取调仓所需数据（历史数据）
                rebalance_data = self._get_rebalance_data(data_provider, current_date, trading_days, i + self.config.warmup_period)
                
                # 执行调仓
                trades = strategy.rebalance(
                    rebalance_data,
                    current_prices,
                    current_date,
                )
                self.result.trades.extend(trades)
            
            # 4. 更新组合
            strategy.update_portfolio(current_prices, current_date)
            
            # 5. 记录组合历史
            self.result.portfolio_history.extend(strategy.portfolio_history)
        
        # 计算绩效指标
        self._calculate_performance_metrics(strategy)
        
        # 生成权益曲线
        self._build_equity_curve(strategy)
        
        # 处理基准数据
        if benchmark_data is not None:
            self.result.benchmark_curve = self._process_benchmark(benchmark_data)
        
        logger.info("回测完成")
        return self.result
    
    def _get_trading_days(self, data_provider: Any) -> List[date]:
        """获取交易日历"""
        # 这里应该从数据提供者获取交易日历
        # 简化实现：生成日期序列
        days = []
        current = self.config.start_date - timedelta(days=self.config.warmup_period * 2)
        while current <= self.config.end_date:
            if current.weekday() < 5:  # 周一到周五
                days.append(current)
            current += timedelta(days=1)
        return days
    
    def _get_data_for_date(
        self,
        data_provider: Any,
        current_date: date,
        trading_days: List[date],
        day_index: int,
    ) -> Dict[str, pd.DataFrame]:
        """获取指定日期的数据"""
        # 这里应该从数据提供者获取数据
        # 简化实现：返回空字典
        return {}
    
    def _get_current_prices(self, data_provider: Any, current_date: date) -> Dict[str, float]:
        """获取当前价格"""
        # 这里应该从数据提供者获取价格
        # 简化实现：返回空字典
        return {}
    
    def _should_rebalance(
        self,
        current_date: date,
        day_index: int,
        trading_days: List[date],
    ) -> bool:
        """检查是否需要调仓"""
        # 第一天总是调仓
        if day_index == 0 and self.config.rebalance_on_start:
            return True
        
        # 根据调仓频率判断
        # 这里简化为每月调仓
        if day_index > 0:
            prev_date = trading_days[day_index - 1]
            if current_date.month != prev_date.month:
                return True
        
        return False
    
    def _get_rebalance_data(
        self,
        data_provider: Any,
        current_date: date,
        trading_days: List[date],
        day_index: int,
    ) -> Dict[str, pd.DataFrame]:
        """获取调仓所需的历史数据"""
        # 这里应该从数据提供者获取历史数据
        # 简化实现：返回空字典
        return {}
    
    def _calculate_performance_metrics(self, strategy: MultiFactorStrategy) -> None:
        """计算绩效指标"""
        if not strategy.portfolio_history:
            return
        
        # 提取权益数据
        dates = [h["date"] for h in strategy.portfolio_history]
        values = [h["portfolio_value"] for h in strategy.portfolio_history]
        
        df = pd.DataFrame({"date": dates, "equity": values})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 计算收益率
        df["return"] = df["equity"].pct_change()
        
        # 总收益率
        initial = df["equity"].iloc[0]
        final = df["equity"].iloc[-1]
        total_return = (final - initial) / initial
        
        # 年化收益率
        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 年化波动率
        daily_volatility = df["return"].std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # 最大回撤
        df["cummax"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["cummax"]) / df["cummax"]
        max_drawdown = df["drawdown"].min()
        
        # 胜率
        winning_trades = [t for t in strategy.trade_history if t["action"] == "sell"]
        # 这里简化计算，实际需要计算每笔交易的盈亏
        win_rate = 0.5
        
        self.result.performance_metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(strategy.trade_history),
            "final_equity": final,
        }
    
    def _build_equity_curve(self, strategy: MultiFactorStrategy) -> None:
        """构建权益曲线"""
        if not strategy.portfolio_history:
            return
        
        dates = [h["date"] for h in strategy.portfolio_history]
        values = [h["portfolio_value"] for h in strategy.portfolio_history]
        
        self.result.equity_curve = pd.DataFrame({
            "date": dates,
            "equity": values,
        })
    
    def _process_benchmark(self, benchmark_data: pd.DataFrame) -> pd.DataFrame:
        """处理基准数据"""
        # 标准化基准数据格式
        if "close" in benchmark_data.columns:
            benchmark_data = benchmark_data.rename(columns={"close": "equity"})
        
        if "date" not in benchmark_data.columns and benchmark_data.index.name == "date":
            benchmark_data = benchmark_data.reset_index()
        
        return benchmark_data
