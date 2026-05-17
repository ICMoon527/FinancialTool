# -*- coding: utf-8 -*-
"""
历史数据回填脚本：为过去 N 个交易日生成 IntradayDailySummary 记录

使预热机制（indicator warm-up / preheating）产生实质效果。
每只股票每天存一条记录，含最后80根K线（last_klines_json），
次日请求时可加载前日数据预热 MACD/RSI/KDJ/MFI/主力吸筹等指标。

用法:
    python scripts/backfill_daily_summaries.py --days 30
    python scripts/backfill_daily_summaries.py --days 30 --batch 50
    python scripts/backfill_daily_summaries.py --code 000001 --days 10
"""
import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from src.storage import DatabaseManager
from stock_selector.trading_calendar import get_trading_days, get_previous_trading_day


WARMUP_BARS = 80


def _compute_daily_summary_from_klines(klines: list) -> dict:
    """
    从K线数据计算每日终值快照的指标数据

    此函数与 api/v1/endpoints/intraday.py 中的同名函数保持逻辑一致。
    """
    if not klines:
        return {}

    last_k = klines[-1]
    result = {
        "last_price": last_k.get("Close") or 0.0,
        "avg_price": last_k.get("AvgPrice") or 0.0,
        "open_price": klines[0].get("Open") or 0.0,
        "high_price": max((k.get("High") or 0.0 for k in klines), default=0.0),
        "low_price": min((k.get("Low") or 0.0 for k in klines), default=0.0),
        "total_volume": sum(k.get("Volume") or 0.0 for k in klines),
        "total_amount": sum(k.get("Amount") or 0.0 for k in klines),
        "kline_count": len(klines),
        "last_time": last_k.get("timestamp", "")[11:16] if len(last_k.get("timestamp", "")) >= 16 else "",
        "last_klines": klines[-WARMUP_BARS:],
    }
    return result


def get_distinct_codes_from_klines(db_manager: DatabaseManager) -> List[str]:
    """
    从 intraday_kline_1min 表中查询所有有K线数据的股票代码
    """
    with db_manager.get_session() as session:
        from src.storage import IntradayKline1Min
        from sqlalchemy import distinct

        results = (
            session.query(distinct(IntradayKline1Min.code))
            .order_by(IntradayKline1Min.code)
            .all()
        )
        codes = sorted([r[0] for r in results])
        logger.info(f"K线表中找到 {len(codes)} 只股票代码")
        return codes


def backfill_daily_summaries(
    db_manager: DatabaseManager,
    codes: List[str],
    days: int,
    batch_size: int = 50,
) -> dict:
    """
    为最近 days 个交易日回填 IntradayDailySummary 记录

    Args:
        db_manager: 数据库管理器
        codes: 股票代码列表
        days: 回填最近几个交易日
        batch_size: 每批提交的记录数

    Returns:
        {"total": 总数, "skipped": 跳过数, "created": 新建数, "failed": 失败数}
    """
    end_date = get_previous_trading_day()
    start_date = end_date - timedelta(days=days * 2)
    trading_days = get_trading_days(start_date, end_date)
    trading_days = trading_days[-days:] if len(trading_days) > days else trading_days

    logger.info(f"回填范围: {trading_days[0]} ~ {trading_days[-1]}，共 {len(trading_days)} 个交易日")
    logger.info(f"待处理股票: {len(codes)} 只，批量大小: {batch_size}")

    stats = {"total": 0, "skipped": 0, "created": 0, "failed": 0, "refilling": 0}
    pending = []

    for date_obj in trading_days:
        logger.info(f"--- 处理日期: {date_obj} ---")
        for code in codes:
            stats["total"] += 1
            try:
                existing = db_manager.load_daily_summary(code, date_obj)
                if existing:
                    last_klines = existing.get("last_klines", None)
                    if last_klines is not None and len(last_klines) > 0:
                        stats["skipped"] += 1
                        continue
                    if last_klines is not None:
                        stats["refilling"] = stats.get("refilling", 0) + 1

                klines = db_manager.load_intraday_klines(code, date_obj)
                if not klines or len(klines) < 10:
                    stats["skipped"] += 1
                    continue

                indicators = _compute_daily_summary_from_klines(klines)
                ok = db_manager.save_daily_summary(code, date_obj, klines, indicators)
                if ok:
                    stats["created"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning(f"保存失败: {code} {date_obj}")
            except Exception as e:
                stats["failed"] += 1
                logger.warning(f"处理异常 {code} {date_obj}: {e}")

            if stats["total"] % 50 == 0:
                logger.info(
                    f"进度: {stats['total']}/{len(trading_days) * len(codes)} "
                    f"| 新建={stats['created']} 跳过={stats['skipped']} 失败={stats['failed']}"
                )

    logger.info(
        f"回填完成: 总计={stats['total']} 新建={stats['created']} "
        f"跳过={stats['skipped']} 失败={stats['failed']}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="回填 IntradayDailySummary 历史数据")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回填最近几个交易日（默认30）",
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="指定单只股票代码，不指定则回填所有",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
        help="批量进度报告间隔（默认50）",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始历史数据回填")
    logger.info(f"参数: days={args.days}, code={args.code or '全部'}, batch={args.batch}")
    logger.info("=" * 60)

    db_manager = DatabaseManager()

    if args.code:
        codes = [args.code]
    else:
        codes = get_distinct_codes_from_klines(db_manager)

    if not codes:
        logger.warning("没有找到任何股票代码，退出")
        return

    stats = backfill_daily_summaries(db_manager, codes, args.days, args.batch)

    logger.info("=" * 60)
    logger.info(f"结果: 新建={stats['created']} 跳过={stats['skipped']} 失败={stats['failed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()