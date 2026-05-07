# -*- coding: utf-8 -*-
"""
模拟交易测试模块

基于历史分时K线数据回放策略信号生成过程，
输出交易统计报告，验证策略有效性和稳定性。

使用示例:
    from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy
    from watchdog.strategies.simulator import T0Simulator

    strategy = IntradayT0Strategy(stock_code="000001")
    simulator = T0Simulator(strategy)
    report = simulator.run(historical_klines)
    print(report)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy, T0Signal

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """单笔模拟交易记录"""

    buy_time: datetime
    buy_price: float
    sell_time: Optional[datetime] = None
    sell_price: Optional[float] = None
    return_pct: Optional[float] = None
    buy_signal: Optional[T0Signal] = None
    sell_signal: Optional[T0Signal] = None
    status: str = "open"  # "open" / "closed"

    def close(self, sell_price: float, sell_time: datetime, sell_signal: T0Signal) -> None:
        """平仓"""
        self.sell_price = sell_price
        self.sell_time = sell_time
        self.sell_signal = sell_signal
        self.return_pct = (sell_price - self.buy_price) / self.buy_price * 100
        self.status = "closed"


@dataclass
class SimulationReport:
    """模拟交易统计报告"""

    stock_code: str
    total_klines: int
    total_signals: int
    buy_signals: int
    sell_signals: int
    total_trades: int
    win_trades: int
    lose_trades: int
    win_rate: float
    avg_return_pct: float
    max_return_pct: float
    min_return_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    trades: List[TradeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "total_klines": self.total_klines,
            "total_signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "total_trades": self.total_trades,
            "win_trades": self.win_trades,
            "lose_trades": self.lose_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_return_pct": round(self.avg_return_pct, 4),
            "max_return_pct": round(self.max_return_pct, 4),
            "min_return_pct": round(self.min_return_pct, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
        }

    def __repr__(self) -> str:
        lines = [
            "=" * 60,
            f"  模拟交易报告 - {self.stock_code}",
            "=" * 60,
            f"  总K线数:       {self.total_klines}",
            f"  总信号数:       {self.total_signals} (买入{self.buy_signals} / 卖出{self.sell_signals})",
            f"  总交易次数:     {self.total_trades}",
            f"  盈利次数:       {self.win_trades}",
            f"  亏损次数:       {self.lose_trades}",
            f"  胜率:           {self.win_rate:.2%}",
            f"  平均收益率:     {self.avg_return_pct:+.2f}%",
            f"  最大单笔收益:   {self.max_return_pct:+.2f}%",
            f"  最小单笔收益:   {self.min_return_pct:+.2f}%",
            f"  累计收益率:     {self.total_return_pct:+.2f}%",
            f"  最大回撤:       {self.max_drawdown_pct:.2f}%",
            f"  夏普比率:       {self.sharpe_ratio:.2f}",
            f"  盈亏比:         {self.profit_factor:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class T0Simulator:
    """
    做T策略模拟交易器

    基于历史分时K线数据逐根回放，模拟策略信号生成和虚拟交易过程，
    输出完整的统计报告。
    """

    def __init__(self, strategy: IntradayT0Strategy):
        """
        Args:
            strategy: 已初始化的 IntradayT0Strategy 实例
        """
        self.strategy = strategy
        self.trades: List[TradeRecord] = []
        self._current_position: Optional[TradeRecord] = None
        self._equity_curve: List[float] = []  # 净值曲线
        self._cumulative_return: float = 0.0
        self._peak_return: float = 0.0

    def run(self, klines: List[Dict[str, Any]], verbose: bool = False) -> SimulationReport:
        """
        运行模拟交易

        Args:
            klines: 历史K线数据列表，每项为包含 Open,High,Low,Close,Volume 的字典
            verbose: 是否打印每笔交易详情

        Returns:
            SimulationReport 统计报告
        """
        self.strategy.reset()
        self.trades.clear()
        self._current_position = None
        self._equity_curve = [0.0]
        self._cumulative_return = 0.0
        self._peak_return = 0.0

        buy_signals_count = 0
        sell_signals_count = 0

        for i, kline in enumerate(klines):
            signal = self.strategy.feed_kline(kline)

            if signal is None:
                continue

            if signal.signal_type == "buy":
                buy_signals_count += 1
                if self._current_position is None:
                    trade = TradeRecord(
                        buy_time=signal.trigger_time,
                        buy_price=signal.price,
                        buy_signal=signal,
                    )
                    self._current_position = trade
                    self.trades.append(trade)
                    if verbose:
                        logger.info(f"  [买] {signal.trigger_time} @ {signal.price:.2f} 得分={signal.score}")

            elif signal.signal_type == "sell":
                sell_signals_count += 1
                if self._current_position is not None:
                    self._current_position.close(
                        sell_price=signal.price,
                        sell_time=signal.trigger_time,
                        sell_signal=signal,
                    )
                    self._cumulative_return += self._current_position.return_pct or 0.0
                    self._equity_curve.append(self._cumulative_return)
                    self._peak_return = max(self._peak_return, self._cumulative_return)
                    if verbose:
                        logger.info(
                            f"  [卖] {signal.trigger_time} @ {signal.price:.2f} 收益率={self._current_position.return_pct:+.2f}%"
                        )
                    self._current_position = None

        # 如果最后还持有仓位，以最后价格平仓
        last_price = klines[-1]["Close"] if klines else 0
        if self._current_position is not None:
            self._current_position.close(
                sell_price=last_price,
                sell_time=datetime.now(),
                sell_signal=T0Signal(stock_code=self.strategy.stock_code, signal_type="sell", price=last_price),
            )
            self._cumulative_return += self._current_position.return_pct or 0.0
            self._equity_curve.append(self._cumulative_return)

        return self._build_report(
            total_klines=len(klines),
            buy_signals=buy_signals_count,
            sell_signals=sell_signals_count,
        )

    def _build_report(self, total_klines: int, buy_signals: int, sell_signals: int) -> SimulationReport:
        """构建统计报告"""
        closed_trades = [t for t in self.trades if t.status == "closed"]
        total_trades = len(closed_trades)

        if total_trades == 0:
            return SimulationReport(
                stock_code=self.strategy.stock_code,
                total_klines=total_klines,
                total_signals=buy_signals + sell_signals,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                total_trades=0,
                win_trades=0,
                lose_trades=0,
                win_rate=0.0,
                avg_return_pct=0.0,
                max_return_pct=0.0,
                min_return_pct=0.0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                trades=self.trades,
            )

        returns = [t.return_pct or 0.0 for t in closed_trades]
        win_trades = sum(1 for r in returns if r > 0)
        lose_trades = sum(1 for r in returns if r <= 0)
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0

        avg_return = np.mean(returns) if returns else 0.0
        max_return = float(np.max(returns)) if returns else 0.0
        min_return = float(np.min(returns)) if returns else 0.0
        total_return = self._cumulative_return

        # 最大回撤
        max_drawdown = 0.0
        peak = 0.0
        for eq in self._equity_curve:
            peak = max(peak, eq)
            drawdown = peak - eq
            max_drawdown = max(max_drawdown, drawdown)

        # 夏普比率（简化版，假设无风险利率为0）
        if returns and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(total_trades)
        else:
            sharpe = 0.0

        # 盈亏比
        total_profit = sum(r for r in returns if r > 0)
        total_loss = abs(sum(r for r in returns if r <= 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf") if total_profit > 0 else 0.0

        return SimulationReport(
            stock_code=self.strategy.stock_code,
            total_klines=total_klines,
            total_signals=buy_signals + sell_signals,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            total_trades=total_trades,
            win_trades=win_trades,
            lose_trades=lose_trades,
            win_rate=win_rate,
            avg_return_pct=avg_return,
            max_return_pct=max_return,
            min_return_pct=min_return,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            trades=self.trades,
        )


# ============================================================
# 便捷函数
# ============================================================


def generate_simulated_intraday_data(
    stock_code: str = "000001",
    num_bars: int = 240,
    base_price: float = 10.0,
    volatility: float = 0.003,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    生成模拟的分时K线数据用于测试

    Args:
        stock_code: 股票代码
        num_bars: K线数量（默认240根=4小时的1分钟K线）
        base_price: 基准价格
        volatility: 每根K线的波动率
        seed: 随机种子

    Returns:
        模拟K线数据列表
    """
    import pandas as pd

    np.random.seed(seed)

    prices = [base_price]
    for _ in range(num_bars - 1):
        change = np.random.normal(0, volatility)
        # 加入一个小的均值回归倾向
        mean_reversion = (base_price - prices[-1]) * 0.001
        new_price = prices[-1] * (1 + change + mean_reversion)
        prices.append(new_price)

    klines = []
    start_time = pd.Timestamp("2026-01-15 09:30:00")

    for i in range(num_bars):
        close_price = prices[i]
        open_price = prices[i - 1] if i > 0 else base_price
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility * 0.5)))
        volume = int(np.random.uniform(100000, 500000) * (1 + 0.5 * np.sin(i / 20)))

        klines.append(
            {
                "Open": round(open_price, 2),
                "High": round(high_price, 2),
                "Low": round(low_price, 2),
                "Close": round(close_price, 2),
                "Volume": volume,
                "timestamp": start_time + pd.Timedelta(minutes=i),
            }
        )

    return klines
