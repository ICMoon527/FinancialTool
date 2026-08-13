# -*- coding: utf-8 -*-
"""
合成场景单元测试：验证跌停封死拦截代码路径 100% 正确。

测试场景：
  Case A: 真封死 — 跌停缩量 (0.8% 均量)，被 _is_limit_down_sealed 拦截，次日开板后止损
  Case B: 放量开板 (对照组) — 收盘价相同但成交量达到 30% 均量，判断为未封死，当日正常卖出
"""

import sys
import logging
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.strategy_backtest.engine import StrategyBacktestEngine
from src.core.strategy_backtest.exit_strategies import TieredExitStrategy
from stock_selector.base import StrategyMatch, StockSelectorStrategy, StrategyMetadata, StrategyType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Mock 数据提供器
# ============================================================


class MockDataProvider:
    """返回合成日线数据的 Mock 数据提供器。"""

    def __init__(self, synthetic_df: pd.DataFrame):
        self._synthetic_df = synthetic_df

    def get_daily_data(self, stock_code: str, days: Any = None, start_date: Any = None, end_date: Any = None, **kwargs: Any) -> Any:
        # 返回完整数据，TimeIsolatedDataProvider 会按当前日期队列过滤
        return self._synthetic_df.copy(), "mock"

    def get_index_daily_data(self, index_symbol: str, **kwargs: Any) -> Any:
        return pd.DataFrame(), "mock"

    def get_realtime_quote(self, stock_code: str, **kwargs: Any) -> Any:
        return None

    def set_current_date(self, current_date: date) -> None:
        pass


# ============================================================
# Mock 选股策略
# ============================================================


class MockBuyStrategy(StockSelectorStrategy):
    """在第一次调用 select() 时返回买入信号，后续不再返回。"""

    def __init__(self) -> None:
        metadata = StrategyMetadata(
            id="mock_buy",
            name="mock_buy",
            display_name="Mock Buy",
            description="Only returns buy signal on first call",
            strategy_type=StrategyType.PYTHON,
        )
        super().__init__(metadata)
        self._has_signaled = False

    def select(self, stock_code: str, **kwargs: Any) -> StrategyMatch:
        if not self._has_signaled:
            self._has_signaled = True
            return StrategyMatch(
                strategy_id=self.metadata.id,
                strategy_name=self.metadata.name,
                matched=True,
                score=60.0,
                control_degree=50.0,
            )
        return StrategyMatch(
            strategy_id=self.metadata.id,
            strategy_name=self.metadata.name,
            matched=False,
            score=0.0,
        )


# ============================================================
# 辅助：合成数据生成
# ============================================================


def _make_synthetic_df(
    base_date: date,
    day_data: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    生成合成日线数据。

    Args:
        base_date: 起始日期
        day_data: 每项为 {open, high, low, close, volume} 的字典列表

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    records = []
    for i, dd in enumerate(day_data):
        d = base_date + timedelta(days=i)
        c = dd["close"]
        v = dd["volume"]
        records.append({
            "date": d,
            "open": dd["open"],
            "high": dd["high"],
            "low": dd["low"],
            "close": c,
            "volume": v,
            "amount": c * v,
        })
    return pd.DataFrame(records)


def _build_engine(
    synthetic_data_df: pd.DataFrame,
    trading_dates: List[date],
    exit_strategy: Optional[TieredExitStrategy] = None,
    initial_capital: float = 100000.0,
    max_positions: int = 1,
) -> StrategyBacktestEngine:
    """创建并初始化回测引擎。"""
    if exit_strategy is None:
        exit_strategy = _make_default_exit_strategy()

    provider = MockDataProvider(synthetic_data_df)
    engine = StrategyBacktestEngine(
        data_provider=provider,
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.001,
        max_positions=max_positions,
        exit_strategy=exit_strategy,
        market_trend_filter={},  # 禁用大盘过滤器
    )
    engine.set_trading_dates(trading_dates)
    engine.set_stock_pool(["000001"])
    engine.set_strategy(MockBuyStrategy())

    # 预填充成交量历史（20 天，避免冷启动期保守假设干扰测试）
    # 正常日均量 = 1,000,000 股
    engine._volume_history["000001"] = deque([1_000_000] * 20, maxlen=20)
    engine._last_volume_update["000001"] = trading_dates[0] - timedelta(days=1)

    return engine


def _make_default_exit_strategy() -> TieredExitStrategy:
    """创建默认参数的动态分级止盈策略。"""
    return TieredExitStrategy(
        stop_loss_pct=0.07,
        time_stop_days=30,
        time_stop_min_return=0.0,
    )


def _run_engine(engine: StrategyBacktestEngine, num_steps: int) -> StrategyBacktestEngine:
    """执行引擎指定步数。"""
    for _ in range(num_steps):
        if not engine._step():
            break
    return engine


# ============================================================
# 测试数据
# ============================================================

BASE_DATE = date(2026, 1, 5)  # 周一

# 3 个交易日：T 买入 → T+1 跌停测试 → T+2 开板/持续
# D0: 正常日，收盘 10.00
# D1: 跌停日，收盘 9.00（-10%）
# D2: 开板日，收盘 9.20（或继续跌停）

# 测试中共用的基础数据（D0 和 D2 固定）
COMMON_DAY_DATA: List[Dict[str, Any]] = [
    {"open": 10.00, "high": 10.50, "low": 9.50, "close": 10.00, "volume": 1_000_000},  # D0: 正常
    {"open": 10.00, "high": 10.20, "low": 9.80, "close": 10.00, "volume": 1_000_000},  # D1: 正常（T+1 挂单日）
]

TRADING_DATES = [
    BASE_DATE,
    BASE_DATE + timedelta(days=1),
    BASE_DATE + timedelta(days=2),
    BASE_DATE + timedelta(days=3),
]


# ============================================================
# Case A: 真封死
# ============================================================


def test_case_a_true_sealed() -> None:
    """
    验证：跌停缩量时被拦截，持仓保留，次日开板后止损。

    D0: 买入 000001 @ 10.00（策略信号触发）
    D1: 收盘 9.00（跌停），成交量 8,000（0.8% 均量）
        → 触发止损 (9.00 ≤ 9.30) → _is_limit_down_sealed → True → 持仓保留
    D2: 收盘 9.20，成交量 1,200,000（120% 均量）
        → 触发止损 (9.20 ≤ 9.30) → _is_limit_down_sealed → False → 以 9.20 清仓
    """
    print("\n" + "=" * 60)
    print("  Case A: 真封死 — 跌停缩量被拦截，次日开板后止损")
    print("=" * 60)

    # 构造数据：D1 跌停且缩量，D2 开板放量
    day_data = COMMON_DAY_DATA + [
        {"open": 9.00, "high": 9.00, "low": 9.00, "close": 9.00, "volume": 8_000},      # D1: 跌停封死（0.8%均量）
        {"open": 9.20, "high": 9.50, "low": 9.10, "close": 9.20, "volume": 1_200_000},  # D2: 开板放量（120%均量）
    ]
    df = _make_synthetic_df(BASE_DATE, day_data)
    engine = _build_engine(df, TRADING_DATES)
    initial_cash = engine.portfolio.cash

    # ---- 执行 4 步（D0 买入 → D1 正常 → D2 封死 → D3 开板卖出） ----
    engine = _run_engine(engine, 4)

    # ---- 断言 ----
    cash_after_d2 = engine.portfolio.cash
    pos = engine.portfolio.get_position("000001")

    # 1. D2 已清仓
    assert pos is None, f"预期 D2 已清仓，但仍有持仓: {pos}"

    # 2. D2 以 9.20 卖出
    #    买入：10.00 * 1.001 = 10.01（含滑点）
    #    买入股数 = 100000 * 0.995 / 10.01 ≈ 9,900 (取整到100)
    #    卖出：9.20（含手续费）
    trades = [t for t in engine.portfolio.trades if t.order_type.name == "SELL"]
    assert len(trades) == 1, f"预期 1 笔卖出，实际 {len(trades)} 笔"
    sell_trade = trades[0]
    assert abs(sell_trade.price - 9.20) < 0.01, f"卖出价格应为 9.20，实际 {sell_trade.price:.2f}"

    # 3. 有 1 次封死判定
    assert engine._sealed_true_count == 1, f"预期 1 次封死判定，实际 {engine._sealed_true_count}"

    # 4. 总检查次数 = 2（D1 + D2 各检查一次）
    assert engine._sealed_check_count == 2, f"预期 2 次封死检查，实际 {engine._sealed_check_count}"

    # 5. 卖出后现金回到账户（约 1,705.90 + 9,800 × 9.19 ≈ 91,767）
    assert cash_after_d2 > 90000, f"卖出后现金应 > 90000，实际 {cash_after_d2:.2f}"

    print(f"  ✓ 卖出价格: {sell_trade.price:.2f}")
    print(f"  ✓ 封死检查: {engine._sealed_check_count} 次, 封死判定: {engine._sealed_true_count} 次")
    print(f"  ✓ 最终现金: {cash_after_d2:.2f}")
    print(f"  ✓ 持仓: {pos}")
    print("  Case A PASSED")


# ============================================================
# Case B: 放量开板（对照组）
# ============================================================


def test_case_b_not_sealed() -> None:
    """
    验证：跌停但放量时不被拦截，当日正常卖出。

    D0: 买入 000001 @ 10.00
    D1: T+1 规则，正常持有
    D2: 收盘 9.00（跌停价），但成交量 300,000（30% 均量 > 15% 阈值）
        → 触发止损 (9.00 ≤ 9.30) → _is_limit_down_sealed → False → 以 9.00 清仓
    D3: 不再持有，无操作
    """
    print("\n" + "=" * 60)
    print("  Case B: 放量开板（对照组）— 跌停放量，当日正常卖出")
    print("=" * 60)

    # 构造数据：D2 跌停但放量（30% 均量 > 15% 阈值）
    day_data = COMMON_DAY_DATA + [
        {"open": 9.00, "high": 9.00, "low": 9.00, "close": 9.00, "volume": 300_000},  # D2: 跌停放量（30%均量）
        {"open": 9.00, "high": 9.00, "low": 9.00, "close": 9.00, "volume": 300_000},  # D3: 无持仓
    ]
    df = _make_synthetic_df(BASE_DATE, day_data)
    engine = _build_engine(df, TRADING_DATES)

    # ---- 执行 3 步（D0 买入 → D1 正常 → D2 放量跌停卖出） ----
    engine = _run_engine(engine, 3)

    # ---- 断言 ----
    pos = engine.portfolio.get_position("000001")

    # 1. D1 已清仓
    assert pos is None, f"预期 D1 已清仓，但仍有持仓: {pos}"

    # 2. D1 以 9.00 卖出
    trades = [t for t in engine.portfolio.trades if t.order_type.name == "SELL"]
    assert len(trades) == 1, f"预期 1 笔卖出，实际 {len(trades)} 笔"
    sell_trade = trades[0]
    assert abs(sell_trade.price - 9.00) < 0.01, f"卖出价格应为 9.00，实际 {sell_trade.price:.2f}"

    # 3. 封死判定为 0
    assert engine._sealed_true_count == 0, f"预期 0 次封死判定，实际 {engine._sealed_true_count}"

    # 4. 总检查次数 = 1（仅 D1 检查一次）
    assert engine._sealed_check_count == 1, f"预期 1 次封死检查，实际 {engine._sealed_check_count}"

    print(f"  ✓ 卖出价格: {sell_trade.price:.2f}")
    print(f"  ✓ 封死检查: {engine._sealed_check_count} 次, 封死判定: {engine._sealed_true_count} 次")
    print(f"  ✓ 持仓: {pos}")
    print("  Case B PASSED")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    test_case_a_true_sealed()
    test_case_b_not_sealed()
    print("\n" + "=" * 60)
    print("  全部测试通过！")
    print("=" * 60)