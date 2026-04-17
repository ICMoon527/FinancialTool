# -*- coding: utf-8 -*-
"""
多维度评价指标计算模块。

实现年化收益率、最大回撤率、夏普比率、胜率、盈亏比、信息比率等核心指标。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.engine import Portfolio, Trade, OrderType


class PerformanceMetrics:
    """
    绩效指标计算器。
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk_free_rate: float = 0.03,
        benchmark_returns: Optional[List[float]] = None,
    ):
        """
        初始化绩效指标计算器。

        Args:
            portfolio: 投资组合
            risk_free_rate: 无风险利率（年化）
            benchmark_returns: 基准收益率序列
        """
        self.portfolio = portfolio
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns

        self._equity_df = self._build_equity_dataframe()
        self._returns = self._calculate_daily_returns()

    def _build_equity_dataframe(self) -> pd.DataFrame:
        """
        构建权益数据DataFrame。

        Returns:
            权益数据DataFrame
        """
        if not self.portfolio.equity_history:
            return pd.DataFrame(columns=["date", "equity"])

        dates, equities = zip(*self.portfolio.equity_history)
        df = pd.DataFrame({"date": dates, "equity": equities})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _calculate_daily_returns(self) -> pd.Series:
        """
        计算日收益率。

        Returns:
            日收益率序列
        """
        if self._equity_df.empty or len(self._equity_df) < 2:
            return pd.Series(dtype="float64")

        returns = self._equity_df["equity"].pct_change().dropna()
        returns.index = self._equity_df["date"].iloc[1:]
        return returns

    def get_total_return(self) -> float:
        """
        计算总收益率。

        Returns:
            总收益率
        """
        if self._equity_df.empty:
            return 0.0

        initial = self._equity_df["equity"].iloc[0]
        final = self._equity_df["equity"].iloc[-1]
        return (final - initial) / initial

    def get_annualized_return(self, days_per_year: int = 252) -> float:
        """
        计算年化收益率。

        Args:
            days_per_year: 一年的交易日数

        Returns:
            年化收益率
        """
        total_return = self.get_total_return()
        if self._equity_df.empty:
            return 0.0

        num_days = (self._equity_df["date"].iloc[-1] - self._equity_df["date"].iloc[0]).days
        if num_days <= 0:
            return 0.0

        years = num_days / days_per_year
        if years <= 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def get_volatility(self, days_per_year: int = 252) -> float:
        """
        计算年化波动率。

        Args:
            days_per_year: 一年的交易日数

        Returns:
            年化波动率
        """
        if self._returns.empty:
            return 0.0

        return self._returns.std() * math.sqrt(days_per_year)

    def get_sharpe_ratio(self, days_per_year: int = 252) -> float:
        """
        计算夏普比率。

        Args:
            days_per_year: 一年的交易日数

        Returns:
            夏普比率
        """
        annualized_return = self.get_annualized_return(days_per_year)
        volatility = self.get_volatility(days_per_year)

        if volatility == 0:
            return 0.0

        return (annualized_return - self.risk_free_rate) / volatility

    def get_max_drawdown(self) -> Tuple[float, date, date]:
        """
        计算最大回撤率。

        Returns:
            (最大回撤率, 回撤开始日期, 回撤结束日期)
        """
        if self._equity_df.empty or len(self._equity_df) < 2:
            return 0.0, date.today(), date.today()

        df = self._equity_df.copy()
        df["cummax"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["cummax"]) / df["cummax"]

        max_drawdown = df["drawdown"].min()
        end_idx = df["drawdown"].idxmin()
        start_idx = df.loc[:end_idx, "cummax"].idxmax()

        start_date = df.loc[start_idx, "date"].date()
        end_date = df.loc[end_idx, "date"].date()

        return abs(max_drawdown), start_date, end_date

    def get_max_drawdown_rate(self) -> float:
        """
        获取最大回撤率。

        Returns:
            最大回撤率
        """
        return self.get_max_drawdown()[0]

    def get_calmar_ratio(self, days_per_year: int = 252) -> float:
        """
        计算卡尔马比率（年化收益率/最大回撤）。

        Args:
            days_per_year: 一年的交易日数

        Returns:
            卡尔马比率
        """
        annualized_return = self.get_annualized_return(days_per_year)
        max_drawdown = self.get_max_drawdown_rate()

        if max_drawdown == 0:
            return float("inf") if annualized_return > 0 else 0.0

        return annualized_return / max_drawdown

    def get_win_rate(self) -> float:
        """
        计算胜率。

        Returns:
            胜率
        """
        if not self.portfolio.trades:
            return 0.0

        buy_trades = [t for t in self.portfolio.trades if t.order_type == OrderType.BUY]
        sell_trades = [t for t in self.portfolio.trades if t.order_type == OrderType.SELL]

        if not sell_trades:
            return 0.0

        win_count = 0
        total_trades = len(sell_trades)

        for sell_trade in sell_trades:
            for buy_trade in reversed(buy_trades):
                if buy_trade.stock_code == sell_trade.stock_code and buy_trade.date <= sell_trade.date:
                    if sell_trade.price > buy_trade.price:
                        win_count += 1
                    break

        return win_count / total_trades if total_trades > 0 else 0.0

    def get_profit_loss_ratio(self) -> float:
        """
        计算盈亏比。

        Returns:
            盈亏比
        """
        if not self.portfolio.trades:
            return 0.0

        buy_trades = [t for t in self.portfolio.trades if t.order_type == OrderType.BUY]
        sell_trades = [t for t in self.portfolio.trades if t.order_type == OrderType.SELL]

        if not sell_trades:
            return 0.0

        profits = []
        losses = []

        for sell_trade in sell_trades:
            for buy_trade in reversed(buy_trades):
                if buy_trade.stock_code == sell_trade.stock_code and buy_trade.date <= sell_trade.date:
                    pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity
                    if pnl > 0:
                        profits.append(pnl)
                    elif pnl < 0:
                        losses.append(abs(pnl))
                    break

        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0

        return avg_profit / avg_loss if avg_loss > 0 else float("inf") if avg_profit > 0 else 0.0

    def get_information_ratio(self, days_per_year: int = 252) -> float:
        """
        计算信息比率。

        Args:
            days_per_year: 一年的交易日数

        Returns:
            信息比率
        """
        if self.benchmark_returns is None or len(self.benchmark_returns) != len(self._returns):
            return 0.0

        active_returns = self._returns.values - np.array(self.benchmark_returns)
        tracking_error = np.std(active_returns) * math.sqrt(days_per_year)

        if tracking_error == 0:
            return 0.0

        annualized_active_return = np.mean(active_returns) * days_per_year
        return annualized_active_return / tracking_error

    def get_sortino_ratio(self, days_per_year: int = 252) -> float:
        """
        计算索提诺比率。

        Args:
            days_per_year: 一年的交易日数

        Returns:
            索提诺比率
        """
        if self._returns.empty:
            return 0.0

        annualized_return = self.get_annualized_return(days_per_year)

        downside_returns = self._returns[self._returns < 0]
        downside_risk = downside_returns.std() * math.sqrt(days_per_year) if not downside_returns.empty else 0.0

        if downside_risk == 0:
            return float("inf") if annualized_return > 0 else 0.0

        return (annualized_return - self.risk_free_rate) / downside_risk

    def get_beta(self) -> float:
        """
        计算贝塔（Beta）。

        贝塔是衡量投资组合相对于基准的系统性风险的指标。
        - Beta = 1：投资组合的风险与基准相同
        - Beta > 1：投资组合的风险高于基准
        - Beta < 1：投资组合的风险低于基准

        Returns:
            贝塔值
        """
        if self._returns.empty or self.benchmark_returns is None or len(self.benchmark_returns) != len(self._returns):
            return 0.0

        portfolio_returns = self._returns.values
        benchmark_returns = np.array(self.benchmark_returns)

        # 计算协方差和方差
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def get_alpha(self, days_per_year: int = 252) -> float:
        """
        计算阿尔法（Alpha）。

        阿尔法是衡量投资组合超额收益的指标，即超过基准收益的部分。
        Alpha > 0：投资组合表现优于基准
        Alpha = 0：投资组合表现与基准相同
        Alpha < 0：投资组合表现不如基准

        计算公式：
        Alpha = Rp - [Rf + Beta * (Rm - Rf)]
        其中：
        - Rp：投资组合年化收益率
        - Rf：无风险利率
        - Beta：贝塔值
        - Rm：基准年化收益率

        Args:
            days_per_year: 一年的交易日数

        Returns:
            阿尔法值（年化）
        """
        if self._returns.empty or self.benchmark_returns is None or len(self.benchmark_returns) != len(self._returns):
            return 0.0

        # 计算投资组合年化收益率
        portfolio_annual_return = self.get_annualized_return(days_per_year)

        # 计算基准年化收益率
        benchmark_returns = np.array(self.benchmark_returns)
        total_benchmark_return = (1 + benchmark_returns).prod() - 1
        num_days = len(benchmark_returns)
        years = num_days / days_per_year
        if years > 0:
            benchmark_annual_return = (1 + total_benchmark_return) ** (1 / years) - 1
        else:
            benchmark_annual_return = 0.0

        # 计算贝塔
        beta = self.get_beta()

        # 计算 CAPM 预期收益率
        capm_expected_return = self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate)

        # 计算阿尔法
        alpha = portfolio_annual_return - capm_expected_return

        return alpha

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        获取所有绩效指标。

        Returns:
            包含所有指标的字典
        """
        max_drawdown, drawdown_start, drawdown_end = self.get_max_drawdown()

        return {
            "total_return": self.get_total_return(),
            "annualized_return": self.get_annualized_return(),
            "volatility": self.get_volatility(),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "max_drawdown": max_drawdown,
            "max_drawdown_start": drawdown_start,
            "max_drawdown_end": drawdown_end,
            "calmar_ratio": self.get_calmar_ratio(),
            "win_rate": self.get_win_rate(),
            "profit_loss_ratio": self.get_profit_loss_ratio(),
            "information_ratio": self.get_information_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "beta": self.get_beta(),
            "alpha": self.get_alpha(),
            "total_trades": len(self.portfolio.trades),
            "initial_capital": self.portfolio.initial_capital,
            "final_equity": self.portfolio.get_total_equity(),
        }
