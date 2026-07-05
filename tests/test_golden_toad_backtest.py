# -*- coding: utf-8 -*-
"""
金蛤蟆策略回测脚本。

Phase 1: 在全市场扫描金蛤蟆买入信号，收集 100 个有效样本（去重）。
Phase 2: 对每个样本追踪 2 个月（44 个交易日）后市表现。
Phase 3: 统计涨幅 ≥30% 的胜率。
"""

import sys
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stock_selector.strategies.Python.golden_toad_strategy import GoldenToadStrategy
from stock_selector.trading_calendar import get_trading_calendar, get_trading_days
from src.storage import get_db, StockDaily
from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks
from sqlalchemy import select, and_

# 配置日志级别
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
WINDOW_SIZE = 150  # 选股时间窗口（交易日）
TARGET_SAMPLES = 100  # 目标样本数
DEDUP_TRADING_DAYS = 20  # 去重间隔（交易日）
TRACK_TRADING_DAYS = 44  # 后市追踪周期（交易日 ≈ 2 个月）
GAIN_THRESHOLD = 30.0  # 涨幅阈值（%）
MAX_DATA_ROWS = 1000  # 单只股票最大处理数据行数（取最近N行，避免超长历史卡死）


def print_separator(title: str) -> None:
    """打印分隔标题。"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def count_trading_days_between(
    trading_days_sorted: List[date],
    start_date: date,
    end_date: date,
) -> int:
    """计算两个日期之间的交易日数量（不含两端）。"""
    count = 0
    for d in trading_days_sorted:
        if d > start_date and d < end_date:
            count += 1
    return count


def get_nth_trading_day_after(
    trading_days_sorted: List[date],
    start_date: date,
    n: int,
) -> Optional[date]:
    """获取 start_date 之后第 n 个交易日（含 start_date 本身计为第 0 个）。"""
    started = False
    count = 0
    for d in trading_days_sorted:
        if d >= start_date:
            if not started:
                started = True
            if count == n:
                return d
            count += 1
    return None


def build_daily_dataframe(records) -> pd.DataFrame:
    """从 StockDaily 记录列表构建标准化 DataFrame。"""
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
    return df


# ======================================================================
# Phase 1: 样本采集
# ======================================================================

def collect_samples(
    strategy: GoldenToadStrategy,
    stock_codes: List[str],
    code_to_name: Dict[str, str],
    db,
    trading_days_sorted: List[date],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    遍历全市场标的，滑动窗口扫描，收集 TARGET_SAMPLES 个买入信号。

    Returns:
        (samples, total_checked, total_windows)
    """
    samples: List[Dict[str, Any]] = []
    total_checked = 0
    total_windows = 0
    last_entry_dates: Dict[str, date] = {}  # per-stock 去重记录

    with db.get_session() as session:
        for stock_code in stock_codes:
            total_checked += 1

            # 进度输出
            if total_checked % 50 == 0:
                print(f"  样本采集进度: {total_checked}/{len(stock_codes)} 只标的，"
                      f"已扫描 {total_windows} 个窗口，已收集 {len(samples)}/{TARGET_SAMPLES} 个样本")

            if len(samples) >= TARGET_SAMPLES:
                break

            stock_name = code_to_name.get(stock_code, "")

            # 查询该股票的所有日线数据（取最近 MAX_DATA_ROWS 行）
            records = session.execute(
                select(StockDaily)
                .where(StockDaily.code == stock_code)
                .order_by(StockDaily.date.desc())
                .limit(MAX_DATA_ROWS)
            ).scalars().all()
            records = list(reversed(records))  # 恢复为时间升序

            if len(records) < WINDOW_SIZE:
                continue

            df = build_daily_dataframe(records)
            if len(df) < WINDOW_SIZE:
                continue

            # 滑动窗口遍历
            n = len(df)
            for i in range(n - WINDOW_SIZE + 1):
                total_windows += 1
                window_df = df.iloc[i : i + WINDOW_SIZE].copy().reset_index(drop=True)

                if "date" not in window_df.columns:
                    continue

                match = strategy.select(stock_code, stock_name, daily_data=window_df)

                if not match.matched:
                    continue

                # 计算买入日期和价格
                entry_date = window_df["date"].iloc[-1]
                # 确保 entry_date 是 date 类型
                if hasattr(entry_date, "date") and callable(entry_date.date):
                    entry_date = entry_date.date()
                elif isinstance(entry_date, pd.Timestamp):
                    entry_date = entry_date.date()

                # 去重检查
                if stock_code in last_entry_dates:
                    interval = count_trading_days_between(
                        trading_days_sorted,
                        last_entry_dates[stock_code],
                        entry_date,
                    )
                    if interval <= DEDUP_TRADING_DAYS:
                        # 间隔太短，跳过（滑动窗口重复信号）
                        continue

                last_entry_dates[stock_code] = entry_date

                entry_price = float(window_df["close"].iloc[-1])
                details = match.match_details

                # 确定买点类型
                bp1 = details.get("buy_point_1", {})
                bp2 = details.get("buy_point_2", {})
                if bp1.get("triggered") and bp2.get("triggered"):
                    buy_point = "both"
                elif bp2.get("triggered"):
                    buy_point = "buy_point_2"
                else:
                    buy_point = "buy_point_1"

                sample = {
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "raw_score": match.raw_score,
                    "essential_score": details.get("essential_score", 0),
                    "bonus_score": details.get("bonus_score", 0),
                    "buy_point": buy_point,
                    # 形态关键日期（用于报告）
                    "left_eye_date": _extract_date(window_df, details.get("left_eye", {}), "position"),
                    "right_eye_date": _extract_date(window_df, details.get("right_eye", {}), "position"),
                    "left_claw_date": _extract_date(window_df, details.get("left_claw", {}), "position"),
                    "right_claw_date": _extract_date(window_df, details.get("right_claw", {}), "low_position"),
                }
                samples.append(sample)

                if len(samples) >= TARGET_SAMPLES:
                    break

    return samples, total_checked, total_windows


def _extract_date(df: pd.DataFrame, component: Dict[str, Any], pos_key: str) -> Optional[date]:
    """从形态组件字典中提取日期。"""
    if not component.get("found", False):
        return None
    pos = component.get(pos_key, -1)
    if pos < 0 or pos >= len(df):
        return None
    d = df["date"].iloc[pos]
    if hasattr(d, "date") and callable(d.date):
        return d.date()
    elif isinstance(d, pd.Timestamp):
        return d.date()
    return d


# ======================================================================
# Phase 2: 后市追踪
# ======================================================================

def track_performance(
    session,
    sample: Dict[str, Any],
    trading_days_sorted: List[date],
) -> Dict[str, Any]:
    """
    追踪单个样本的 2 个月后市表现。

    Returns:
        {
            "max_close": float,
            "max_gain_pct": float,
            "max_gain_date": Optional[date],
            "days_tracked": int,
            "reached_30pct": bool,
            "first_30pct_date": Optional[date],
            "close_after_2m": float,
            "invalid": bool,
        }
    """
    entry_date = sample["entry_date"]
    entry_price = sample["entry_price"]
    stock_code = sample["stock_code"]

    # 计算 2 个月后的截止日期（44 个交易日）
    target_date = get_nth_trading_day_after(trading_days_sorted, entry_date, TRACK_TRADING_DAYS)

    if target_date is None:
        return {
            "max_close": 0.0, "max_gain_pct": 0.0, "max_gain_date": None,
            "days_tracked": 0, "reached_30pct": False, "first_30pct_date": None,
            "close_after_2m": 0.0, "invalid": True,
        }

    # 查询后市数据（从买入日次日开始）
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
        return {
            "max_close": 0.0, "max_gain_pct": 0.0, "max_gain_date": None,
            "days_tracked": 0, "reached_30pct": False, "first_30pct_date": None,
            "close_after_2m": 0.0, "invalid": True,
        }

    max_close = 0.0
    max_gain_date = None
    first_30pct_date = None
    reached_30pct = False
    close_after_2m = float(rows[-1].close)

    for r in rows:
        c = float(r.close)
        if c > max_close:
            max_close = c
            max_gain_date = r.date if hasattr(r.date, "__str__") else r.date

        gain_pct = (c - entry_price) / entry_price * 100.0
        if gain_pct >= GAIN_THRESHOLD and not reached_30pct:
            reached_30pct = True
            first_30pct_date = r.date if hasattr(r.date, "__str__") else r.date

    max_gain_pct = (max_close - entry_price) / entry_price * 100.0

    return {
        "max_close": round(max_close, 2),
        "max_gain_pct": round(max_gain_pct, 2),
        "max_gain_date": max_gain_date,
        "days_tracked": len(rows),
        "reached_30pct": reached_30pct,
        "first_30pct_date": first_30pct_date,
        "close_after_2m": round(close_after_2m, 2),
        "invalid": False,
    }


def track_all_samples(
    db,
    samples: List[Dict[str, Any]],
    trading_days_sorted: List[date],
) -> List[Dict[str, Any]]:
    """对所有样本执行后市追踪。"""
    results = []
    invalid_count = 0

    with db.get_session() as session:
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                print(f"  追踪进度: {i + 1}/{len(samples)} 个样本...")

            perf = track_performance(session, sample, trading_days_sorted)
            if perf["invalid"]:
                invalid_count += 1
                continue

            results.append({**sample, **perf})

    if invalid_count > 0:
        print(f"  注意: {invalid_count} 个样本因后市数据不足被排除")

    return results


# ======================================================================
# Phase 3: 胜率统计
# ======================================================================

def print_statistics(results: List[Dict[str, Any]]) -> None:
    """输出胜率统计报告。"""
    total = len(results)
    if total == 0:
        print_separator("胜率统计")
        print("  无有效样本，无法统计")
        return

    reached = sum(1 for r in results if r["reached_30pct"])
    win_rate = reached / total * 100.0

    print_separator("胜率统计")
    print(f"  有效样本: {total} 个")
    print(f"  达标样本（涨幅 ≥{GAIN_THRESHOLD:.0f}%）: {reached} 个")
    print(f"  未达标: {total - reached} 个")
    print(f"  胜率: {win_rate:.2f}% ({reached}/{total})")

    # 涨幅分布
    buckets = {
        "涨幅 ≥ 50%": 0,
        "涨幅 30-50%": 0,
        "涨幅 10-30%": 0,
        "涨幅 0-10%": 0,
        "涨幅 < 0%": 0,
    }
    for r in results:
        pct = r["max_gain_pct"]
        if pct >= 50:
            buckets["涨幅 ≥ 50%"] += 1
        elif pct >= 30:
            buckets["涨幅 30-50%"] += 1
        elif pct >= 10:
            buckets["涨幅 10-30%"] += 1
        elif pct >= 0:
            buckets["涨幅 0-10%"] += 1
        else:
            buckets["涨幅 < 0%"] += 1

    print(f"\n  涨幅分布:")
    for label, count in buckets.items():
        print(f"    {label}: {count} 个 ({count / total * 100:.1f}%)")

    # 平均涨幅 / 中位数
    gains = sorted([r["max_gain_pct"] for r in results])
    avg_gain = sum(gains) / len(gains)
    median_gain = gains[len(gains) // 2]
    print(f"\n  平均最大涨幅: {avg_gain:.2f}%")
    print(f"  中位数最大涨幅: {median_gain:.2f}%")

    # 达标样本中，首次达到30%需要的平均交易日数
    reached_dates = [r for r in results if r["reached_30pct"] and r.get("first_30pct_date")]
    if reached_dates:
        avg_days = sum(r["days_tracked"] for r in reached_dates) / len(reached_dates)
        # 这里需要更精确地计算实际达到30%所需的天数，先粗略估计
        # 实际上我们可以在后市追踪中记录精确天数
        print(f"  达标样本平均追踪天数: {avg_days:.1f}")

    # 样本详情（按涨幅降序排列，显示前 20 和后 5）
    print(f"\n  [样本详情 - 按涨幅降序]")
    print(f"  {'代码':<8} {'名称':<8} {'买入日期':<12} {'买入价':<8} {'最高价':<8} {'最大涨幅':<10} {'2个月后':<10} {'达标':<6}")
    print(f"  {'-' * 70}")

    sorted_results = sorted(results, key=lambda r: r["max_gain_pct"], reverse=True)

    # 显示前 20
    show_count = min(20, len(sorted_results))
    for r in sorted_results[:show_count]:
        entry_date_str = str(r["entry_date"]) if r["entry_date"] else "N/A"
        print(
            f"  {r['stock_code']:<8} {r['stock_name']:<8} {entry_date_str:<12} "
            f"{r['entry_price']:<8.2f} {r['max_close']:<8.2f} {r['max_gain_pct']:<10.2f} "
            f"{r['close_after_2m']:<10.2f} {'是' if r['reached_30pct'] else '否':<6}"
        )

    if len(sorted_results) > 20:
        print(f"  {'...':>8}")
        for r in sorted_results[-5:]:
            entry_date_str = str(r["entry_date"]) if r["entry_date"] else "N/A"
            print(
                f"  {r['stock_code']:<8} {r['stock_name']:<8} {entry_date_str:<12} "
                f"{r['entry_price']:<8.2f} {r['max_close']:<8.2f} {r['max_gain_pct']:<10.2f} "
                f"{r['close_after_2m']:<10.2f} {'是' if r['reached_30pct'] else '否':<6}"
            )


# ======================================================================
# Main
# ======================================================================

def main():
    """主入口：三阶段回测流程。"""
    print_separator("金蛤蟆策略回测 - 样本采集与胜率统计")

    # 初始化
    strategy = GoldenToadStrategy()
    db = get_db()
    trading_cal = get_trading_calendar()
    trading_days_sorted = trading_cal.get_all_trading_days()

    # 加载股票池
    print("正在加载股票池...")
    stock_pairs = get_all_stock_code_name_pairs()
    stock_pairs = filter_st_stocks(stock_pairs)
    stock_codes = [code for code, _ in stock_pairs]
    stock_codes = filter_special_stock_codes(stock_codes)
    code_to_name = {code: name for code, name in stock_pairs}
    print(f"股票池共 {len(stock_codes)} 只标的")

    # ----------------------------------------------------------------
    # Phase 1: 样本采集
    # ----------------------------------------------------------------
    print_separator("Phase 1: 样本采集")
    print(f"目标: 收集 {TARGET_SAMPLES} 个有效买入信号（去重间隔 {DEDUP_TRADING_DAYS} 交易日）")

    t0 = time.time()
    samples, total_checked, total_windows = collect_samples(
        strategy, stock_codes, code_to_name, db, trading_days_sorted
    )
    t1 = time.time()

    print_separator("样本采集完成")
    print(f"  扫描标的: {total_checked} 只")
    print(f"  扫描窗口: {total_windows} 个")
    print(f"  收集样本: {len(samples)} 个")
    print(f"  采样耗时: {t1 - t0:.1f} 秒")

    if len(samples) == 0:
        print("  未找到任何符合条件的买入信号，回测终止")
        return

    # ----------------------------------------------------------------
    # Phase 2: 后市追踪
    # ----------------------------------------------------------------
    print_separator("Phase 2: 后市追踪")
    print(f"追踪周期: {TRACK_TRADING_DAYS} 个交易日（约 2 个月）")
    print(f"涨幅阈值: {GAIN_THRESHOLD:.0f}%")

    t2 = time.time()
    results = track_all_samples(db, samples, trading_days_sorted)
    t3 = time.time()
    print(f"追踪耗时: {t3 - t2:.1f} 秒")

    # ----------------------------------------------------------------
    # Phase 3: 胜率统计
    # ----------------------------------------------------------------
    print_statistics(results)

    print(f"\n  总耗时: {t3 - t0:.1f} 秒")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()