# -*- coding: utf-8 -*-
"""
===================================
交易日历模块 (使用本地 trading_calendar.pkl)
===================================

职责：
1. 按市场（A股/港股/美股）判断当日是否为交易日
2. 按市场时区取"今日"日期，避免服务器 UTC 导致日期错误
3. 支持 per-stock 过滤：只分析当日开市市场的股票
4. 根据交易日数计算起始日期

使用项目已有的 stock_selector/trading_calendar.py 模块
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Set

logger = logging.getLogger(__name__)


# 导入项目已有的交易日历模块
import sys
from pathlib import Path

# 确保可以导入 stock_selector 模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stock_selector.trading_calendar import (
    get_trading_calendar,
    is_trading_day as _is_trading_day_cn,
    get_trading_days as _get_trading_days_cn
)


# Market -> IANA timezone for "today"
MARKET_TIMEZONE = {
    "cn": "Asia/Shanghai",
    "hk": "Asia/Hong_Kong",
    "us": "America/New_York",
}


def get_market_for_stock(code: str) -> Optional[str]:
    """
    Infer market region for a stock code.

    Returns:
        'cn' | 'hk' | 'us' | None (None = unrecognized, fail-open: treat as open)
    """
    if not code or not isinstance(code, str):
        return None
    code = (code or "").strip().upper()

    from data_provider import is_us_stock_code, is_us_index_code, is_hk_stock_code

    if is_us_stock_code(code) or is_us_index_code(code):
        return "us"
    if is_hk_stock_code(code):
        return "hk"
    # A-share: 6-digit numeric
    if code.isdigit() and len(code) == 6:
        return "cn"
    return None


def is_market_open(market: str, check_date: date) -> bool:
    """
    Check if the given market is open on the given date.

    For CN market, use local trading_calendar.pkl.
    For other markets, fail-open: returns True.

    Args:
        market: 'cn' | 'hk' | 'us'
        check_date: Date to check

    Returns:
        True if trading day (or fail-open), False otherwise
    """
    if market == "cn":
        # A股使用本地交易日历
        return _is_trading_day_cn(check_date)
    # 其他市场暂时使用 fail-open
    return True


def get_open_markets_today() -> Set[str]:
    """
    Get markets that are open today (by each market's local timezone).

    Returns:
        Set of market keys ('cn', 'hk', 'us') that are trading today
    """
    result: Set[str] = set()
    from zoneinfo import ZoneInfo
    for mkt, tz_name in MARKET_TIMEZONE.items():
        try:
            tz = ZoneInfo(tz_name)
            today = datetime.now(tz).date()
            if is_market_open(mkt, today):
                result.add(mkt)
        except Exception as e:
            logger.warning("get_open_markets_today fail-open for %s: %s", mkt, e)
            result.add(mkt)
    return result


def get_latest_trading_day(market: str, check_date: date) -> date:
    """
    获取离指定日期最近的交易日（先检查当天，再向前查找）

    Args:
        market: 'cn' | 'hk' | 'us'
        check_date: 要检查的日期

    Returns:
        最近的交易日日期
    """
    if market != "cn":
        # 非A股市场暂时返回原日期
        return check_date
    
    # A股使用本地交易日历
    current_date = check_date
    
    # 先检查当天
    if _is_trading_day_cn(current_date):
        return current_date
    
    # 如果当天不是交易日，向前查找最多30天
    current_date = current_date - timedelta(days=1)
    for i in range(30):
        if _is_trading_day_cn(current_date):
            return current_date
        current_date = current_date - timedelta(days=1)
    
    # 如果30天内都没找到，返回原始日期
    logger.warning(f"未在 {check_date} 前30天内找到交易日，使用原始日期")
    return check_date


def get_start_date_by_trading_days(
    end_date: date,
    trading_days: int,
    market: str = "cn"
) -> date:
    """
    根据结束日期和交易日数，计算起始日期（向前推 N 个交易日）

    Args:
        end_date: 结束日期
        trading_days: 需要的交易日数
        market: 市场类型，默认为 'cn'（A股）

    Returns:
        计算得到的起始日期
    """
    if market != "cn":
        # 非A股市场暂时回退到自然日计算
        return end_date - timedelta(days=trading_days * 2)
    
    # A股使用本地交易日历
    calendar = get_trading_calendar()
    all_trading_days = calendar.get_all_trading_days()
    
    # 找到 end_date 在所有交易日中的位置
    try:
        # 先找到小于等于 end_date 的最新交易日
        end_idx = None
        for i, d in enumerate(reversed(all_trading_days)):
            if d <= end_date:
                end_idx = len(all_trading_days) - 1 - i
                break
        
        if end_idx is None:
            # 如果没找到，回退到自然日计算
            logger.warning(f"未找到 {end_date} 之前的交易日，回退到自然日计算")
            return end_date - timedelta(days=trading_days * 2)
        
        # 计算起始索引
        start_idx = max(0, end_idx - trading_days + 1)
        start_date = all_trading_days[start_idx]
        
        logger.info(f"根据 {trading_days} 个交易日计算，起始日期: {start_date}")
        return start_date
        
    except Exception as e:
        logger.warning(f"get_start_date_by_trading_days 出错: {e}，回退到自然日计算")
        # 出错时回退到自然日计算
        return end_date - timedelta(days=trading_days * 2)


def compute_effective_region(
    config_region: str, open_markets: Set[str]
) -> Optional[str]:
    """
    Compute effective market review region given config and open markets.

    Args:
        config_region: From MARKET_REVIEW_REGION ('cn' | 'us' | 'both')
        open_markets: Markets open today

    Returns:
        None: caller uses config default (check disabled)
        '': all relevant markets closed, skip market review
        'cn' | 'us' | 'both': effective subset for today
    """
    if config_region not in ("cn", "us", "both"):
        config_region = "cn"
    if config_region == "cn":
        return "cn" if "cn" in open_markets else ""
    if config_region == "us":
        return "us" if "us" in open_markets else ""
    # both
    parts = []
    if "cn" in open_markets:
        parts.append("cn")
    if "us" in open_markets:
        parts.append("us")
    if not parts:
        return ""
    return "both" if len(parts) == 2 else parts[0]
