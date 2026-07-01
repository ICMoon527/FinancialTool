# -*- coding: utf-8 -*-
"""
金蛤蟆策略数据库遍历测试脚本。

遍历数据库中的所有标的，对每个标的按150交易日滑动窗口运行金蛤蟆策略，
找到第一个符合条件的窗口后停止并输出结果。
"""

import sys
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stock_selector.strategies.Python.golden_toad_strategy import GoldenToadStrategy
from src.storage import get_db, StockDaily
from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks
from sqlalchemy import select

# 配置日志级别
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WINDOW_SIZE = 150  # 选股时间窗口（交易日）


def print_separator(title: str) -> None:
    """打印分隔标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    """主入口：遍历数据库标的，搜索金蛤蟆上车点。"""
    print_separator("金蛤蟆策略 - 数据库遍历搜索")

    # 初始化策略
    strategy = GoldenToadStrategy()

    # 获取数据库连接
    db = get_db()

    # 获取所有股票代码和名称
    print("正在加载股票池...")
    stock_pairs = get_all_stock_code_name_pairs()
    stock_pairs = filter_st_stocks(stock_pairs)
    stock_codes = [code for code, _ in stock_pairs]
    stock_codes = filter_special_stock_codes(stock_codes)
    code_to_name = {code: name for code, name in stock_pairs}

    print(f"股票池共 {len(stock_codes)} 只标的，开始遍历...")

    total_checked = 0
    total_windows = 0

    with db.get_session() as session:
        for stock_code in stock_codes:
            total_checked += 1

            # 每100只输出一次进度
            if total_checked % 100 == 0:
                print(f"  已检查 {total_checked}/{len(stock_codes)} 只标的，"
                      f"已扫描 {total_windows} 个窗口...")

            stock_name = code_to_name.get(stock_code, "")

            # 查询该股票的所有日线数据，按日期排序
            records = session.execute(
                select(StockDaily)
                .where(StockDaily.code == stock_code)
                .order_by(StockDaily.date)
            ).scalars().all()

            if len(records) < WINDOW_SIZE:
                continue

            # 构建 DataFrame
            df = pd.DataFrame([
                {
                    "date": r.date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "amount": r.amount,
                    "pct_chg": r.pct_chg,
                }
                for r in records
            ])

            # 数据清洗
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df.sort_values("date", ascending=True).reset_index(drop=True)

            # 修复 high/low 异常
            for idx in range(len(df)):
                row = df.iloc[idx]
                prices = [row["open"], row["high"], row["low"], row["close"]]
                if row["low"] > row["high"]:
                    df.at[idx, "high"] = max(prices)
                    df.at[idx, "low"] = min(prices)

            df = df[df["low"] <= df["high"]]

            if len(df) < WINDOW_SIZE:
                continue

            # 滑动窗口遍历
            n = len(df)
            for i in range(n - WINDOW_SIZE + 1):
                total_windows += 1
                window_df = df.iloc[i : i + WINDOW_SIZE].copy().reset_index(drop=True)

                # 确保窗口内数据连续（日期列需存在）
                if "date" not in window_df.columns:
                    continue

                match = strategy.select(stock_code, stock_name, daily_data=window_df)

                if match.matched:
                    # 找到了！输出结果
                    details = match.match_details
                    window_start = window_df["date"].iloc[0]
                    window_end = window_df["date"].iloc[-1]
                    entry_price = float(window_df["close"].iloc[-1])

                    # 从窗口数据中提取各关键日期
                    le = details.get("left_eye", {})
                    re = details.get("right_eye", {})
                    lc = details.get("left_claw", {})
                    rc = details.get("right_claw", {})
                    bp1 = details.get("buy_point_1", {})
                    bp2 = details.get("buy_point_2", {})
                    rsi_check = details.get("rsi_check", {})
                    leg_check = details.get("leg_check", {})

                    left_eye_date = None
                    right_eye_date = None
                    left_claw_date = None
                    right_claw_date = None
                    breakout_date = None

                    if le.get("found") and le["position"] >= 0:
                        left_eye_date = window_df["date"].iloc[le["position"]]
                    if re.get("found") and re["position"] >= 0:
                        right_eye_date = window_df["date"].iloc[re["position"]]
                    if lc.get("found") and lc["position"] >= 0:
                        left_claw_date = window_df["date"].iloc[lc["position"]]
                    if rc.get("found") and rc["low_position"] >= 0:
                        right_claw_date = window_df["date"].iloc[rc["low_position"]]
                    if bp2.get("triggered"):
                        breakout_date = window_end

                    print_separator("找到金蛤蟆上车点！")
                    print(f"  标的名称: {stock_name}")
                    print(f"  标的代码: {stock_code}")
                    print(f"  窗口起始日期: {window_start}")
                    print(f"  窗口结束日期: {window_end}")
                    print(f"  上车价格: {entry_price:.2f}")
                    print(f"  原始得分: {match.raw_score:.1f} / 100")
                    print(f"  必要条件得分: {details.get('essential_score', 0):.0f} / 100")
                    print(f"  加分项得分: {details.get('bonus_score', 0):.0f} / 25")

                    # 铁律检查结果
                    iron_pass = details.get("iron_rules_pass", [])
                    iron_fail = details.get("iron_rules_fail", [])
                    if iron_pass:
                        print(f"\n  [铁律检查 - 全部通过 {len(iron_pass)}/{len(iron_pass) + len(iron_fail)}]")
                        for rule in iron_pass:
                            print(f"    {rule}")

                    # 买点信息
                    if bp1.get("triggered"):
                        print(f"\n  买点: 买点1（右爪缩量低吸）")
                    if bp2.get("triggered"):
                        print(f"  买点: 买点2（放量突破颈线）")

                    # 形态详情（含日期）
                    print(f"\n  [形态详情]")
                    if lc.get("found") and left_claw_date is not None:
                        print(f"  左爪日期: {left_claw_date}  close={lc['close_price']:.2f}  "
                              f"缩量={lc.get('volume_ratio', 0):.2f}  "
                              f"距MA60={lc.get('ma60_dist_pct', 0):.1f}%")
                    if le.get("found") and left_eye_date is not None:
                        print(f"  左眼高点日期: {left_eye_date}  high={le['price']:.2f}  "
                              f"量比={le.get('volume_ratio', 0):.2f}  "
                              f"上升{le.get('uptrend_days', 0)}天/回踩{le.get('pullback_days', 0)}天")
                    if re.get("found") and right_eye_date is not None:
                        print(f"  右眼高点日期: {right_eye_date}  high={re['price']:.2f}  "
                              f"上影线={re.get('upper_shadow_ratio', 0):.1%}  "
                              f"量比={re.get('volume_ratio', 0):.2f}  "
                              f"上升{re.get('uptrend_days', 0)}天/回踩{re.get('pullback_days', 0)}天")
                    if rc.get("found") and right_claw_date is not None:
                        print(f"  右爪低点日期: {right_claw_date}  close={rc.get('close_price', 0):.2f}  "
                              f"缩量={rc.get('volume_ratio', 0):.2f}  "
                              f"悬空={rc.get('above_ma60', False)}  回升={rc.get('recovery_pct', 0):.1f}%")
                    if bp2.get("triggered") and breakout_date is not None:
                        print(f"  突破颈线日期: {breakout_date}")

                    # RSI检查
                    if rsi_check:
                        print(f"\n  [RSI检查] 左眼RSI={rsi_check.get('left_rsi', 0):.1f}  "
                              f"右眼RSI={rsi_check.get('right_rsi', 0):.1f}  "
                              f"差异={rsi_check.get('diff_ratio', 0):.1%}  "
                              f"通过={rsi_check.get('ok', False)}")

                    # 腿长检查
                    if leg_check:
                        print(f"  [腿长检查] 左腿={leg_check.get('left_leg_days', 0)}天  "
                              f"右腿={leg_check.get('right_leg_days', 0)}天")

                    print(f"\n  总扫描: {total_checked} 只标的，{total_windows} 个窗口")
                    print(f"{'=' * 60}")
                    return

    # 未找到
    print_separator("搜索完成")
    print(f"  共扫描 {total_checked} 只标的，{total_windows} 个窗口")
    print(f"  未找到符合条件的金蛤蟆上车点")


if __name__ == "__main__":
    main()