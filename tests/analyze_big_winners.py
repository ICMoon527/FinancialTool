# -*- coding: utf-8 -*-
"""分析 12 个大赢家样本的止盈止损触发细节。"""

import sys
from pathlib import Path
from datetime import date
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_golden_toad_backtest import (
    track_performance_with_rules,
    track_performance,
    get_nth_trading_day_after,
    TRACK_TRADING_DAYS,
)
from stock_selector.trading_calendar import get_trading_calendar
from src.storage import get_db, StockDaily
from sqlalchemy import select, and_

# 12 个大赢家（从回测结果中提取）
BIG_WINNERS = [
    ("001203", "大中矿业", date(2025, 10, 21), 13.78),
    ("002427", "尤夫股份", date(2025, 3, 24), 4.76),
    ("002490", "山东墨龙", date(2025, 6, 5), 4.02),
    ("002580", "圣阳股份", date(2024, 12, 23), 8.24),
    ("000407", "胜利股份", date(2025, 10, 13), 3.61),
    ("002315", "焦点科技", date(2023, 3, 22), 25.08),
    ("002292", "奥飞娱乐", date(2023, 3, 22), 6.43),
    ("000831", "中国稀土", date(2025, 7, 8), 35.64),
    ("002491", "通鼎互联", date(2026, 3, 3), 12.23),
    ("000969", "安泰科技", date(2025, 8, 29), 14.49),
    ("001225", "和泰机电", date(2025, 12, 25), 54.81),
    ("001339", "智微智能", date(2024, 12, 30), 38.36),
]


def analyze_winner(session, sample: Dict[str, Any], trading_days: List[date]):
    """逐日分析大赢家的交易过程。"""
    entry_date = sample["entry_date"]
    entry_price = sample["entry_price"]
    stock_code = sample["stock_code"]
    stock_name = sample["stock_name"]

    target_date = get_nth_trading_day_after(trading_days, entry_date, TRACK_TRADING_DAYS)
    if target_date is None:
        return

    rows = session.execute(
        select(StockDaily)
        .where(
            and_(
                StockDaily.code == stock_code,
                StockDaily.date > entry_date,
                StockDaily.date <= target_date,
            )
        )
        .order_by(StockDaily.date)
    ).scalars().all()

    if len(rows) == 0:
        return

    # 模拟规则执行，记录关键事件
    stop_loss = 0.88
    breakeven_trigger = 1.10
    partial_tp = 1.20
    trailing_dd = 0.10

    breakeven_activated = False
    partial_sold = False
    partial_sold_price = 0.0
    partial_sold_date = None
    peak_after_partial = 0.0
    exit_date = None
    exit_price = 0.0
    exit_reason = "到期平仓"

    events = []
    max_close = 0.0
    max_close_date = None

    for i, r in enumerate(rows):
        c = float(r.close)
        h = float(r.high)
        l = float(r.low)
        d = r.date if hasattr(r.date, "__str__") else r.date

        gain_pct = (c - entry_price) / entry_price * 100.0

        if c > max_close:
            max_close = c
            max_close_date = d

        if not partial_sold:
            current_stop = entry_price * stop_loss
            if breakeven_activated:
                current_stop = entry_price

            if c >= entry_price * partial_tp:
                partial_sold = True
                partial_sold_price = c
                partial_sold_date = d
                peak_after_partial = c
                breakeven_activated = True
                events.append(f"  {d}  +20%触发止盈50% → 卖出价 {c:.2f} (涨幅 {gain_pct:.1f}%)")
                continue

            if c >= entry_price * breakeven_trigger and not breakeven_activated:
                breakeven_activated = True
                events.append(f"  {d}  +10%触发保本止损 (涨幅 {gain_pct:.1f}%)")

            if l <= current_stop:
                exit_date = d
                exit_price = current_stop
                exit_reason = "保本出局" if breakeven_activated else "止损出局"
                break
        else:
            if c > peak_after_partial:
                peak_after_partial = c
                peak_gain = (peak_after_partial - entry_price) / entry_price * 100.0
                events.append(f"  {d}  新高! 最高价 {c:.2f} (涨幅 {peak_gain:.1f}%)")

            trailing_stop = peak_after_partial * (1 - trailing_dd)
            if l <= trailing_stop:
                exit_date = d
                exit_price = trailing_stop
                exit_reason = "移动止盈"
                break

    # 最终收益率
    if partial_sold:
        if exit_price == 0.0:
            exit_price = float(rows[-1].close)
            exit_date = rows[-1].date if hasattr(rows[-1].date, "__str__") else rows[-1].date
        pct_1 = (partial_sold_price - entry_price) / entry_price * 100.0
        pct_2 = (exit_price - entry_price) / entry_price * 100.0
        total_return = pct_1 * 0.5 + pct_2 * 0.5
    else:
        if exit_price == 0.0:
            exit_price = float(rows[-1].close)
            exit_date = rows[-1].date if hasattr(rows[-1].date, "__str__") else rows[-1].date
        total_return = (exit_price - entry_price) / entry_price * 100.0

    hold_return = (float(rows[-1].close) - entry_price) / entry_price * 100.0
    max_gain = (max_close - entry_price) / entry_price * 100.0

    print(f"\n{'=' * 70}")
    print(f"  {stock_code} {stock_name}  买入: {entry_date}  价格: {entry_price}")
    print(f"  持有到期收益: {hold_return:+.1f}%  最大涨幅: {max_gain:+.1f}% (日期: {max_close_date})")
    print(f"  规则收益: {total_return:+.1f}%  出局原因: {exit_reason}  出局日期: {exit_date}")
    print(f"  差值: {hold_return - total_return:+.1f}% (规则少赚)")

    # 显示关键事件（最多8个）
    if events:
        print(f"  关键事件:")
        for evt in events[-8:]:
            print(evt)

    # 分析拖后腿原因
    if partial_sold:
        half_pct = (partial_sold_price - entry_price) / entry_price * 100.0
        if exit_reason == "移动止盈":
            trailing_pct = (exit_price - entry_price) / entry_price * 100.0
            print(f"  → 50%仓位在 +{half_pct:.1f}% 止盈，50%仓位在 +{trailing_pct:.1f}% 移动止盈出局")
            if max_gain > total_return * 2:
                print(f"  → 问题: 移动止盈回撤10%太紧，截断了 {max_gain - total_return:.0f}% 的潜在涨幅")
        elif exit_reason == "到期平仓":
            print(f"  → 50%仓位在 +{half_pct:.1f}% 止盈，50%仓位到期平仓")


def main():
    db = get_db()
    trading_cal = get_trading_calendar()
    trading_days = trading_cal.get_all_trading_days()

    print("=" * 70)
    print("  大赢家样本深度分析")
    print("=" * 70)

    with db.get_session() as session:
        for code, name, entry_date, entry_price in BIG_WINNERS:
            sample = {
                "stock_code": code,
                "stock_name": name,
                "entry_date": entry_date,
                "entry_price": entry_price,
            }
            analyze_winner(session, sample, trading_days)


if __name__ == "__main__":
    main()