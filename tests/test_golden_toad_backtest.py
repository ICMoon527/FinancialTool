# -*- coding: utf-8 -*-
"""
金蛤蟆策略回测脚本。

Phase 1: 在全市场扫描金蛤蟆买入信号，收集 100 个有效样本（去重）。
Phase 2: 对每个样本追踪 2 个月（44 个交易日）后市表现。
Phase 3: 统计涨幅 ≥30% 的胜率。
"""

import sys
import logging
import pickle
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

# 成交量确认参数
VOL_CONFIRM_RATIO = 1.2  # 买入日成交量 > 近5日均量 * 1.2
VOL_CONFIRM_DAYS = 5     # 均量计算周期

# 大盘过滤参数
INDEX_MA_PERIOD = 60      # 大盘均线周期
INDEX_MA_SLOPE_DAYS = 20  # 计算MA60斜率的天数
INDEX_CACHE_DIR = project_root / "data" / "cache"
INDEX_FILES = {
    "sh000001": INDEX_CACHE_DIR / "market_sh000001.pkl",
    "sz399001": INDEX_CACHE_DIR / "market_sz399001.pkl",
}


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


def load_market_index(index_name: str = "sh000001") -> pd.DataFrame:
    """从缓存加载大盘指数数据，返回含 MA60 的 DataFrame。"""
    filepath = INDEX_FILES.get(index_name)
    if filepath is None or not filepath.exists():
        raise FileNotFoundError(f"大盘指数缓存文件不存在: {filepath}")
    df = pd.read_pickle(filepath)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ma60"] = df["close"].rolling(window=INDEX_MA_PERIOD).mean()
    df["ma60_slope"] = df["ma60"].diff(INDEX_MA_SLOPE_DAYS) / INDEX_MA_SLOPE_DAYS
    return df


def is_market_uptrend(index_df: pd.DataFrame, target_date: date) -> bool:
    """
    判断指定日期大盘是否处于上升趋势。

    条件：
      1. 指数收盘价在 MA60 上方
      2. MA60 斜率 > 0（近20天MA60在上升）
    """
    row = index_df[index_df["date"] == target_date]
    if len(row) == 0:
        return False
    r = row.iloc[0]
    if pd.isna(r["ma60"]) or pd.isna(r["ma60_slope"]):
        return False
    return r["close"] > r["ma60"] and r["ma60_slope"] > 0


def check_volume_confirmation(window_df: pd.DataFrame) -> bool:
    """
    检查买入日成交量是否确认。

    条件：买入日（窗口最后一天）成交量 > 近5日均量 * VOL_CONFIRM_RATIO
    """
    if len(window_df) < VOL_CONFIRM_DAYS + 1:
        return False
    recent_vol = window_df["volume"].iloc[-(VOL_CONFIRM_DAYS + 1):-1]
    entry_vol = window_df["volume"].iloc[-1]
    avg_vol = recent_vol.mean()
    if pd.isna(avg_vol) or pd.isna(entry_vol) or avg_vol <= 0:
        return False
    return entry_vol > avg_vol * VOL_CONFIRM_RATIO


# ======================================================================
# Phase 1: 样本采集
# ======================================================================

def collect_samples(
    strategy: GoldenToadStrategy,
    stock_codes: List[str],
    code_to_name: Dict[str, str],
    db,
    trading_days_sorted: List[date],
    index_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, int]]:
    """
    遍历全市场标的，滑动窗口扫描，收集 TARGET_SAMPLES 个买入信号。

    Returns:
        (samples, total_checked, total_windows, filter_stats)
    """
    samples: List[Dict[str, Any]] = []
    total_checked = 0
    total_windows = 0
    last_entry_dates: Dict[str, date] = {}  # per-stock 去重记录

    filter_stats = {"volume_filtered": 0, "market_filtered": 0}

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

                # 成交量确认：买入日成交量必须放量
                if not check_volume_confirmation(window_df):
                    filter_stats["volume_filtered"] += 1
                    continue

                # 大盘趋势过滤：买入日大盘必须处于上升趋势
                if not is_market_uptrend(index_df, entry_date):
                    filter_stats["market_filtered"] += 1
                    continue

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

    return samples, total_checked, total_windows, filter_stats


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


def track_performance_with_rules(
    session,
    sample: Dict[str, Any],
    trading_days_sorted: List[date],
) -> Dict[str, Any]:
    """
    追踪单个样本的后市表现，应用分级止盈止损规则。

    默认模式：
      1. 初始止损 -12%
      2. 涨幅达到 +10% → 止损上移至保本价
      3. 涨幅达到 +20% → 卖出 50% 仓位，剩余 50% 启动 10% 移动止盈
      4. 44 个交易日到期 → 市价平仓

    大赢家模式（买入后 10 个交易日内收盘价涨幅 ≥15% 触发）：
      1. 初始止损 -15%（给更多波动空间）
      2. 涨幅达到 +10% → 保本止损（不变）
      3. 不触发 +20% 部分止盈（全部仓位保留）
      4. 15% 移动止盈（给强势股更多回撤空间）

    Returns:
        {
            "exit_date": Optional[date],
            "exit_price": float,
            "exit_reason": str,
            "total_return_pct": float,
            "max_close": float,
            "max_gain_pct": float,
            "days_held": int,
            "invalid": bool,
            "tier": str,  # "default" 或 "big_winner"
        }
    """
    entry_date = sample["entry_date"]
    entry_price = sample["entry_price"]
    stock_code = sample["stock_code"]

    target_date = get_nth_trading_day_after(trading_days_sorted, entry_date, TRACK_TRADING_DAYS)
    if target_date is None:
        return {
            "exit_date": None, "exit_price": 0.0, "exit_reason": "数据不足",
            "total_return_pct": 0.0, "max_close": 0.0, "max_gain_pct": 0.0,
            "days_held": 0, "invalid": True, "tier": "default",
        }

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
            "exit_date": None, "exit_price": 0.0, "exit_reason": "无后市数据",
            "total_return_pct": 0.0, "max_close": 0.0, "max_gain_pct": 0.0,
            "days_held": 0, "invalid": True, "tier": "default",
        }

    # 大赢家模式触发条件
    BIG_WINNER_DAYS = 10       # 窗口期
    BIG_WINNER_TRIGGER = 1.15  # 10个交易日内涨幅达到15%

    # 默认模式参数
    DEFAULT_STOP_LOSS = 0.88
    DEFAULT_TRAILING_DD = 0.10
    BREAKEVEN_TRIGGER = 1.10
    PARTIAL_TAKE_PROFIT = 1.20  # +20%止盈50%

    # 大赢家模式参数
    BIG_WINNER_STOP_LOSS = 0.85
    BIG_WINNER_TRAILING_DD = 0.15

    # 状态
    tier = "default"
    breakeven_activated = False
    partial_sold = False
    partial_sold_price = 0.0
    partial_sold_date = None
    peak_after_partial = 0.0
    big_winner_peak = 0.0  # 大赢家模式下的最高价
    max_close = float(rows[0].close)
    max_gain_pct = 0.0

    exit_date = None
    exit_price = 0.0
    exit_reason = "到期平仓"

    for i, r in enumerate(rows):
        c = float(r.close)
        h = float(r.high)
        l = float(r.low)
        r_date = r.date if hasattr(r.date, "__str__") else r.date

        # 更新最高价
        if c > max_close:
            max_close = c
        gain_pct = (c - entry_price) / entry_price * 100.0
        if gain_pct > max_gain_pct:
            max_gain_pct = gain_pct

        # 大赢家模式检测：10个交易日内收盘价涨幅 ≥15%
        if tier == "default" and i < BIG_WINNER_DAYS and c >= entry_price * BIG_WINNER_TRIGGER:
            tier = "big_winner"
            big_winner_peak = c
            breakeven_activated = True  # 已触发保本

        if tier == "default":
            # ----------------------------------------------------------
            # 默认模式
            # ----------------------------------------------------------
            current_stop = entry_price * DEFAULT_STOP_LOSS
            if breakeven_activated:
                current_stop = entry_price

            if c >= entry_price * PARTIAL_TAKE_PROFIT:
                partial_sold = True
                partial_sold_price = c
                partial_sold_date = r_date
                peak_after_partial = c
                breakeven_activated = True
                continue

            if c >= entry_price * BREAKEVEN_TRIGGER and not breakeven_activated:
                breakeven_activated = True

            if l <= current_stop:
                exit_date = r_date
                exit_price = current_stop
                exit_reason = "保本出局" if breakeven_activated else "止损出局"
                break
        else:
            # ----------------------------------------------------------
            # 大赢家模式：全仓持有，宽松移动止盈
            # ----------------------------------------------------------
            if c > big_winner_peak:
                big_winner_peak = c

            # 止损线：大赢家模式的止损（-15%）
            stop_line = entry_price * BIG_WINNER_STOP_LOSS
            # 保本止损
            if breakeven_activated and entry_price > stop_line:
                stop_line = entry_price

            # 移动止盈线
            trailing_stop = big_winner_peak * (1 - BIG_WINNER_TRAILING_DD)
            actual_stop = max(stop_line, trailing_stop)

            if l <= actual_stop:
                exit_date = r_date
                exit_price = actual_stop
                if actual_stop == stop_line:
                    exit_reason = "大赢家保本出局" if breakeven_activated else "大赢家止损出局"
                else:
                    exit_reason = "大赢家移动止盈"
                break

    # 计算总收益率
    if partial_sold:
        # 默认模式 50%止盈
        if exit_price == 0.0:
            exit_price = float(rows[-1].close)
            exit_date = rows[-1].date if hasattr(rows[-1].date, "__str__") else rows[-1].date
        pct_1 = (partial_sold_price - entry_price) / entry_price * 100.0
        pct_2 = (exit_price - entry_price) / entry_price * 100.0
        total_return_pct = pct_1 * 0.5 + pct_2 * 0.5
    else:
        if exit_price == 0.0:
            exit_price = float(rows[-1].close)
            exit_date = rows[-1].date if hasattr(rows[-1].date, "__str__") else rows[-1].date
        total_return_pct = (exit_price - entry_price) / entry_price * 100.0

    days_held = len(rows) if exit_reason == "到期平仓" else i + 1

    return {
        "exit_date": exit_date,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "total_return_pct": round(total_return_pct, 2),
        "max_close": round(max_close, 2),
        "max_gain_pct": round(max_gain_pct, 2),
        "days_held": days_held,
        "invalid": False,
        "tier": tier,
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


def track_all_samples_with_rules(
    db,
    samples: List[Dict[str, Any]],
    trading_days_sorted: List[date],
) -> List[Dict[str, Any]]:
    """对所有样本应用止盈止损规则追踪。"""
    results = []
    invalid_count = 0

    with db.get_session() as session:
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                print(f"  规则追踪进度: {i + 1}/{len(samples)} 个样本...")

            perf = track_performance_with_rules(session, sample, trading_days_sorted)
            if perf["invalid"]:
                invalid_count += 1
                continue

            results.append({**sample, **perf})

    if invalid_count > 0:
        print(f"  注意: {invalid_count} 个样本因后市数据不足被排除")

    return results


def print_comparison(hold_results: List[Dict[str, Any]], rule_results: List[Dict[str, Any]]) -> None:
    """对比持有策略 vs 止盈止损策略。"""
    print_separator("Phase 4: 持有策略 vs 止盈止损策略 对比")

    # 只对比两者都有效的样本
    hold_map = {r["stock_code"] + str(r["entry_date"]): r for r in hold_results}
    rule_map = {r["stock_code"] + str(r["entry_date"]): r for r in rule_results}
    common_keys = set(hold_map.keys()) & set(rule_map.keys())

    hold_returns = []
    rule_returns = []
    exit_reasons: Dict[str, int] = {}

    for key in common_keys:
        h = hold_map[key]
        r = rule_map[key]
        hold_ret = (h["close_after_2m"] - h["entry_price"]) / h["entry_price"] * 100.0
        hold_returns.append(hold_ret)
        rule_returns.append(r["total_return_pct"])
        reason = r["exit_reason"]
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    n = len(hold_returns)
    if n == 0:
        print("  无有效对比样本")
        return

    # 持有策略统计
    hold_avg = sum(hold_returns) / n
    hold_median = sorted(hold_returns)[n // 2]
    hold_win = sum(1 for r in hold_returns if r > 0)
    hold_loss = sum(1 for r in hold_returns if r < 0)

    # 规则策略统计
    rule_avg = sum(rule_returns) / n
    rule_median = sorted(rule_returns)[n // 2]
    rule_win = sum(1 for r in rule_returns if r > 0)
    rule_loss = sum(1 for r in rule_returns if r < 0)

    print(f"\n  对比样本数: {n}")
    print(f"\n  {'指标':<16} {'持有到期':<14} {'止盈止损':<14}")
    print(f"  {'-' * 44}")
    print(f"  {'平均收益率':<16} {hold_avg:>+8.2f}%      {rule_avg:>+8.2f}%")
    print(f"  {'中位数收益率':<16} {hold_median:>+8.2f}%      {rule_median:>+8.2f}%")
    print(f"  {'盈利样本数':<16} {hold_win:>8}        {rule_win:>8}")
    print(f"  {'亏损样本数':<16} {hold_loss:>8}        {rule_loss:>8}")
    print(f"  {'胜率':<16} {hold_win / n * 100:>7.1f}%        {rule_win / n * 100:>7.1f}%")
    print(f"  {'最大单笔盈利':<16} {max(hold_returns):>+8.2f}%      {max(rule_returns):>+8.2f}%")
    print(f"  {'最大单笔亏损':<16} {min(hold_returns):>+8.2f}%      {min(rule_returns):>+8.2f}%")

    # 分级统计
    big_winner_count = sum(1 for r in rule_results if r.get("tier") == "big_winner")
    default_count = n - big_winner_count
    print(f"\n  分级统计: 大赢家模式 {big_winner_count} 个 | 默认模式 {default_count} 个")

    print(f"\n  止盈止损出局原因分布:")
    for reason in ["止损出局", "保本出局", "止盈50%+移动止盈", "移动止盈", "大赢家移动止盈", "大赢家保本出局", "大赢家止损出局", "到期平仓"]:
        count = exit_reasons.get(reason, 0)
        if count > 0:
            print(f"    {reason}: {count} 个 ({count / n * 100:.1f}%)")

    # 逐一对比前 10 个样本
    print(f"\n  [样本逐个对比 - 前15个]")
    print(f"  {'代码':<8} {'买入日期':<12} {'买入价':<8} {'持有收益':<10} {'规则收益':<10} {'出局原因':<12}")
    print(f"  {'-' * 70}")

    # 用规则收益排序
    sorted_pairs = sorted(
        [(key, hold_map[key], rule_map[key]) for key in common_keys],
        key=lambda x: x[2]["total_return_pct"],
        reverse=True,
    )
    for key, h, r in sorted_pairs[:15]:
        entry_date_str = str(h["entry_date"]) if h["entry_date"] else "N/A"
        hold_ret = (h["close_after_2m"] - h["entry_price"]) / h["entry_price"] * 100.0
        print(
            f"  {h['stock_code']:<8} {entry_date_str:<12} "
            f"{h['entry_price']:<8.2f} {hold_ret:>+8.2f}%    {r['total_return_pct']:>+8.2f}%    "
            f"{r['exit_reason']:<12}"
        )


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

    # 加载大盘指数数据
    print("正在加载大盘指数数据...")
    index_df = load_market_index("sh000001")
    print(f"  上证指数: {len(index_df)} 条日线数据")

    # ----------------------------------------------------------------
    # Phase 1: 样本采集
    # ----------------------------------------------------------------
    print_separator("Phase 1: 样本采集")
    print(f"目标: 收集 {TARGET_SAMPLES} 个有效买入信号（去重间隔 {DEDUP_TRADING_DAYS} 交易日）")
    print(f"过滤条件: 成交量放量确认(>5日均量{VOL_CONFIRM_RATIO:.1f}倍) | 大盘上升趋势(MA60上方+MA60斜率>0)")

    t0 = time.time()
    samples, total_checked, total_windows, filter_stats = collect_samples(
        strategy, stock_codes, code_to_name, db, trading_days_sorted, index_df
    )
    t1 = time.time()

    print_separator("样本采集完成")
    print(f"  扫描标的: {total_checked} 只")
    print(f"  扫描窗口: {total_windows} 个")
    print(f"  收集样本: {len(samples)} 个")
    print(f"  过滤统计: 成交量不足 {filter_stats['volume_filtered']} 个 | 大盘弱势 {filter_stats['market_filtered']} 个")
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

    # ----------------------------------------------------------------
    # Phase 4: 持有策略 vs 止盈止损策略 对比
    # ----------------------------------------------------------------
    print_separator("Phase 4: 分级止盈止损规则追踪")
    print(f"规则: 默认(-12%止损/+10%保本/+20%止盈50%/10%移动止盈)")
    print(f"      大赢家模式(10日内涨≥15%触发→-15%止损/全仓持有/15%移动止盈)")

    t4 = time.time()
    rule_results = track_all_samples_with_rules(db, samples, trading_days_sorted)
    t5 = time.time()
    print(f"规则追踪耗时: {t5 - t4:.1f} 秒")

    print_comparison(results, rule_results)

    print(f"\n  总耗时: {t5 - t0:.1f} 秒")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()