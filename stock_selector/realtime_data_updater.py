# -*- coding: utf-8 -*-
"""
实时数据更新模块 - 专门为 stock_selector 设计的实时数据更新器（优化版）

核心功能：
1. 使用批量API获取所有股票实时行情数据（腾讯财经接口）
2. 将实时行情数据转换为 StockDaily 格式并批量保存到数据库
3. 完善的异常处理和日志记录
4. 支持批量更新多只股票的实时数据

性能优化：
- 批次大小：300只/批（从100优化）
- 取消批次间休眠
- 批量数据库保存（不再单只保存）
- 批量更新追踪器
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import random
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy import and_, select

from data_provider.realtime_types import (
    RealtimeSource,
    UnifiedRealtimeQuote,
    safe_float,
    safe_int,
)
from src.storage import DatabaseManager, StockDaily

from .data_update_tracker import DataUpdateTracker, get_update_tracker
from .stock_pool import filter_special_stock_codes, get_all_stock_codes

logger = logging.getLogger(__name__)

# 策略2：数据源优先级定义（数字越大优先级越高）
SOURCE_PRIORITY = {
    "TushareFetcher": 4,
    "AkshareFetcher": 3,
    "EfinanceFetcher": 2,
    "RealtimeDataUpdater": 1,  # 实时数据优先级最低
    "tencent": 1,
}

# User-Agent 池，用于随机轮换
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class RealtimeDataUpdater:
    """
    实时数据更新器 - 专门为 stock_selector 设计
    使用批量API提高效率（优化版）
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        update_tracker: Optional[DataUpdateTracker] = None,
        batch_size: int = 300,
    ):
        """
        初始化实时数据更新器

        Args:
            db_manager: 数据库管理器
            update_tracker: 数据更新追踪器
            batch_size: 每批查询的股票数量（默认300只，优化版）
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self.update_tracker = update_tracker or get_update_tracker()
        self.batch_size = batch_size

        self._should_stop = False

        logger.info("RealtimeDataUpdater 初始化成功（批量优化版）")

    def stop(self):
        """
        停止更新
        """
        logger.info("RealtimeDataUpdater 收到停止信号")
        self._should_stop = True

    def _fetch_batch_quotes_tencent(self, stock_codes: List[str]) -> Dict[str, Optional[UnifiedRealtimeQuote]]:
        """
        使用腾讯财经接口批量获取实时行情

        Args:
            stock_codes: 股票代码列表

        Returns:
            股票代码到 UnifiedRealtimeQuote 的字典
        """
        result = {}

        try:
            # 构建股票符号列表（添加市场前缀）
            symbols = []
            for code in stock_codes:
                if code.startswith(("6", "5", "9")):
                    symbols.append(f"sh{code}")
                else:
                    symbols.append(f"sz{code}")

            # 构建URL
            url = f"http://qt.gtimg.cn/q={','.join(symbols)}"
            headers = {
                "Referer": "http://finance.qq.com",
                "User-Agent": random.choice(USER_AGENTS),
            }

            logger.debug(f"[批量查询] 腾讯接口: {len(stock_codes)} 只股票")

            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = "gbk"

            if response.status_code != 200:
                logger.warning(f"[批量查询] 腾讯接口返回状态码 {response.status_code}")
                return result

            # 解析响应
            content = response.text.strip()

            # 腾讯返回格式: v_sh600519="...";v_sz000001="..."
            lines = content.split(";")

            for line in lines:
                line = line.strip()
                if not line or "=" not in line:
                    continue

                # 提取股票代码和数据
                eq_pos = line.find("=")
                if eq_pos == -1:
                    continue

                # 提取代码部分（v_sh600519 -> sh600519 -> 600519）
                code_part = line[:eq_pos].strip()
                if code_part.startswith("v_"):
                    code_part = code_part[2:]

                if code_part.startswith("sh"):
                    stock_code = code_part[2:]
                elif code_part.startswith("sz"):
                    stock_code = code_part[2:]
                else:
                    continue

                # 提取数据部分
                data_start = line.find('"')
                data_end = line.rfind('"')
                if data_start == -1 or data_end == -1:
                    continue

                data_str = line[data_start + 1 : data_end]
                fields = data_str.split("~")

                if len(fields) < 45:
                    logger.debug(f"[批量查询] {stock_code} 数据字段不足: {len(fields)}")
                    continue

                # 构建 UnifiedRealtimeQuote
                quote = UnifiedRealtimeQuote(
                    code=stock_code,
                    name=fields[1] if len(fields) > 1 else "",
                    source=RealtimeSource.TENCENT,
                    price=safe_float(fields[3]),
                    change_pct=safe_float(fields[32]),
                    change_amount=safe_float(fields[31]) if len(fields) > 31 else None,
                    volume=safe_int(fields[6]) * 100 if fields[6] else None,
                    open_price=safe_float(fields[5]),
                    high=safe_float(fields[34]) if len(fields) > 34 else None,
                    low=(
                        safe_float(fields[35].split("/")[0])
                        if len(fields) > 35 and "/" in str(fields[35])
                        else safe_float(fields[35]) if len(fields) > 35 else None
                    ),
                    pre_close=safe_float(fields[4]),
                    turnover_rate=safe_float(fields[38]) if len(fields) > 38 else None,
                    amplitude=safe_float(fields[43]) if len(fields) > 43 else None,
                    volume_ratio=safe_float(fields[49]) if len(fields) > 49 else None,
                    pe_ratio=safe_float(fields[39]) if len(fields) > 39 else None,
                    pb_ratio=safe_float(fields[46]) if len(fields) > 46 else None,
                    circ_mv=safe_float(fields[44]) * 100000000 if len(fields) > 44 and fields[44] else None,
                    total_mv=safe_float(fields[45]) * 100000000 if len(fields) > 45 and fields[45] else None,
                )

                if quote.has_basic_data():
                    result[stock_code] = quote
                    logger.debug(f"[批量查询] 成功解析 {stock_code}: {quote.name}")
                else:
                    result[stock_code] = None

        except Exception as e:
            logger.error(f"[批量查询] 腾讯接口查询失败: {e}", exc_info=True)

        return result

    def _convert_realtime_to_stock_daily(
        self, quote: UnifiedRealtimeQuote, record_date: Optional[date] = None
    ) -> Optional[StockDaily]:
        """
        将 UnifiedRealtimeQuote 转换为 StockDaily 格式

        Args:
            quote: 实时行情数据
            record_date: 记录日期（默认今天）

        Returns:
            StockDaily 对象，转换失败返回 None
        """
        if quote is None:
            logger.warning("实时行情数据为空，无法转换")
            return None

        if not quote.has_basic_data():
            logger.warning(f"股票 {quote.code} 的实时行情缺少基本价格数据，无法转换")
            return None

        if record_date is None:
            record_date = date.today()

        try:
            # 构建 StockDaily 对象
            stock_daily = StockDaily(
                code=quote.code,
                date=record_date,
                open=quote.open_price,
                high=quote.high,
                low=quote.low,
                close=quote.price,
                volume=quote.volume,
                amount=quote.amount,
                pct_chg=quote.change_pct,
                volume_ratio=quote.volume_ratio,
                data_source=quote.source.value if quote.source else "RealtimeDataUpdater",
            )

            logger.debug(f"成功转换 {quote.code} 实时行情到 StockDaily 格式")
            return stock_daily

        except Exception as e:
            logger.error(f"转换 {quote.code} 实时行情失败: {e}")
            return None

    def _is_complete_trading_day_data(self) -> bool:
        """
        判断是否为完整的交易日数据（收盘后）

        Returns:
            True 如果是收盘后（15:30后），False 否则
        """
        from datetime import datetime, time
        now = datetime.now().time()
        market_close = time(15, 30)
        return now >= market_close

    def _batch_save_realtime_data(self, stock_dailies: List[StockDaily], record_date: date) -> Tuple[int, List[str]]:
        """
        批量保存实时数据到数据库

        Args:
            stock_dailies: StockDaily 对象列表
            record_date: 记录日期

        Returns:
            (成功数量, 失败股票代码列表)
        """
        if not stock_dailies:
            return 0, []

        # 策略3：判断是否为完整交易日数据
        is_complete_day = self._is_complete_trading_day_data()
        if not is_complete_day:
            logger.info("当前为盘中时间，只有在没有历史数据时才会写入新记录")

        success_count = 0
        failed_codes = []

        try:
            # 先收集所有股票代码
            codes_to_update = [sd.code for sd in stock_dailies]

            with self.db_manager.session_scope() as session:
                # 1. 查询所有已存在的记录
                existing_records = (
                    session.execute(
                        select(StockDaily).where(
                            and_(
                                StockDaily.code.in_(codes_to_update),
                                StockDaily.date == record_date,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )

                # 2. 构建已有记录的字典
                existing_dict = {rec.code: rec for rec in existing_records}

                # 3. 分别更新和插入
                new_records = []
                for stock_daily in stock_dailies:
                    if stock_daily.code in existing_dict:
                        # 策略3：如果是盘中时间且已有历史数据，跳过更新
                        if not is_complete_day:
                            logger.debug(f"盘中时间，保留历史数据，跳过更新: {stock_daily.code}")
                            continue
                        
                        # 策略2：检查数据源优先级
                        existing = existing_dict[stock_daily.code]
                        existing_priority = SOURCE_PRIORITY.get(existing.data_source, 0)
                        new_priority = SOURCE_PRIORITY.get(stock_daily.data_source, 0)
                        
                        if new_priority < existing_priority:
                            logger.debug(f"数据源优先级不足，跳过更新: {stock_daily.code} "
                                       f"(现有: {existing.data_source}={existing_priority}, "
                                       f"新: {stock_daily.data_source}={new_priority})")
                            continue
                        
                        # 更新现有记录（仅在收盘后且优先级足够时执行）
                        if stock_daily.open is not None:
                            existing.open = stock_daily.open
                        if stock_daily.high is not None:
                            existing.high = stock_daily.high
                        if stock_daily.low is not None:
                            existing.low = stock_daily.low
                        if stock_daily.close is not None:
                            existing.close = stock_daily.close
                        if stock_daily.volume is not None:
                            existing.volume = stock_daily.volume
                        if stock_daily.amount is not None:
                            existing.amount = stock_daily.amount
                        if stock_daily.pct_chg is not None:
                            existing.pct_chg = stock_daily.pct_chg
                        if stock_daily.volume_ratio is not None:
                            existing.volume_ratio = stock_daily.volume_ratio
                        existing.data_source = stock_daily.data_source
                        existing.updated_at = datetime.now()
                        success_count += 1
                    else:
                        # 新记录，添加到批量插入列表（盘中或收盘后都可以插入）
                        new_records.append(stock_daily)

                # 4. 批量插入新记录
                if new_records:
                    session.add_all(new_records)
                    success_count += len(new_records)
                    logger.debug(f"批量插入 {len(new_records)} 条新记录")

        except Exception as e:
            logger.error(f"批量保存实时数据失败: {e}", exc_info=True)
            # 如果批量保存失败，回退到单只保存
            logger.info("回退到单只保存模式")
            for stock_daily in stock_dailies:
                try:
                    with self.db_manager.session_scope() as session:
                        existing = session.execute(
                            select(StockDaily).where(
                                and_(
                                    StockDaily.code == stock_daily.code,
                                    StockDaily.date == record_date,
                                )
                            )
                        ).scalar_one_or_none()

                        if existing:
                            # 策略3：如果是盘中时间且已有历史数据，跳过更新
                            if not is_complete_day:
                                logger.debug(f"盘中时间，保留历史数据，跳过更新: {stock_daily.code}")
                                continue
                            
                            # 策略2：检查数据源优先级
                            existing_priority = SOURCE_PRIORITY.get(existing.data_source, 0)
                            new_priority = SOURCE_PRIORITY.get(stock_daily.data_source, 0)
                            
                            if new_priority < existing_priority:
                                logger.debug(f"数据源优先级不足，跳过更新: {stock_daily.code} "
                                           f"(现有: {existing.data_source}={existing_priority}, "
                                           f"新: {stock_daily.data_source}={new_priority})")
                                continue
                            
                            if stock_daily.open is not None:
                                existing.open = stock_daily.open
                            if stock_daily.high is not None:
                                existing.high = stock_daily.high
                            if stock_daily.low is not None:
                                existing.low = stock_daily.low
                            if stock_daily.close is not None:
                                existing.close = stock_daily.close
                            if stock_daily.volume is not None:
                                existing.volume = stock_daily.volume
                            if stock_daily.amount is not None:
                                existing.amount = stock_daily.amount
                            if stock_daily.pct_chg is not None:
                                existing.pct_chg = stock_daily.pct_chg
                            if stock_daily.volume_ratio is not None:
                                existing.volume_ratio = stock_daily.volume_ratio
                            existing.data_source = stock_daily.data_source
                            existing.updated_at = datetime.now()
                        else:
                            session.add(stock_daily)
                    success_count += 1
                except Exception as e2:
                    logger.error(f"保存 {stock_daily.code} 失败: {e2}")
                    failed_codes.append(stock_daily.code)

        return success_count, failed_codes

    def update_realtime_data(
        self,
        stock_codes: Optional[List[str]] = None,
        record_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        批量更新股票实时数据（优化版）

        性能优化：
        - 批次大小：300只/批
        - 取消批次间休眠
        - 批量数据库保存
        - 批量更新追踪器

        Args:
            stock_codes: 股票代码列表（默认所有股票）
            record_date: 记录日期（默认今天）

        Returns:
            更新统计信息
        """
        if record_date is None:
            record_date = date.today()

        # 策略1：检查是否为交易日，非交易日直接跳过
        from .trading_calendar import is_trading_day
        if not is_trading_day(record_date):
            logger.warning(f"{record_date} 不是交易日，跳过实时数据更新")
            return {
                "total_stocks": 0,
                "stocks_success": 0,
                "stocks_failed": 0,
                "failed_stocks": [],
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "batches": 0,
                "skipped": True,
                "reason": "非交易日"
            }

        if stock_codes is None:
            stock_codes = get_all_stock_codes()

        # 过滤特定板块的股票代码
        original_count = len(stock_codes)
        stock_codes = filter_special_stock_codes(stock_codes)
        filtered_count = original_count - len(stock_codes)

        logger.info(f"开始批量更新实时数据：{len(stock_codes)} 只股票，日期 {record_date}")
        if filtered_count > 0:
            logger.info(f"（已过滤 {filtered_count} 只北交所股票）")

        stats = {
            "total_stocks": len(stock_codes),
            "stocks_success": 0,
            "stocks_failed": 0,
            "failed_stocks": [],
            "start_time": datetime.now(),
            "end_time": None,
            "batches": 0,
        }

        print("\n" + "=" * 80)
        print(f"开始批量更新实时数据：{len(stock_codes)} 只股票（优化版）")
        if filtered_count > 0:
            print(f"（已过滤 {filtered_count} 只北交所股票）")
        print(f"日期：{record_date}")
        print(f"批量大小：{self.batch_size} 只/批")
        print("性能优化：批量保存 + 取消休眠")
        print("=" * 80 + "\n")

        # 用于批量更新追踪器的成功股票列表
        all_success_codes = []

        try:
            from tqdm import tqdm

            # 分批处理
            total_batches = (len(stock_codes) + self.batch_size - 1) // self.batch_size
            pbar = tqdm(range(total_batches), desc="批量更新进度", unit="batch")

            for batch_idx in pbar:
                # 检查停止标志
                if self._should_stop:
                    logger.info("更新被用户终止")
                    break

                # 计算当前批次的股票
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(stock_codes))
                batch_stocks = stock_codes[start_idx:end_idx]

                stats["batches"] += 1

                logger.debug(f"处理第 {batch_idx + 1}/{total_batches} 批：{len(batch_stocks)} 只股票")

                # 批量获取实时行情
                batch_success_count = 0
                batch_failed_count = 0
                batch_stock_dailies = []

                try:
                    batch_quotes = self._fetch_batch_quotes_tencent(batch_stocks)

                    # 转换为 StockDaily 格式
                    for stock_code in batch_stocks:
                        quote = batch_quotes.get(stock_code)

                        if quote is not None and quote.has_basic_data():
                            stock_daily = self._convert_realtime_to_stock_daily(quote, record_date)
                            if stock_daily is not None:
                                batch_stock_dailies.append(stock_daily)
                                all_success_codes.append(stock_code)
                            else:
                                batch_failed_count += 1
                                stats["failed_stocks"].append({"code": stock_code, "error": "转换失败"})
                        else:
                            batch_failed_count += 1
                            stats["failed_stocks"].append({"code": stock_code, "error": "获取数据失败"})

                    # 批量保存到数据库
                    if batch_stock_dailies:
                        save_success, save_failed = self._batch_save_realtime_data(batch_stock_dailies, record_date)
                        batch_success_count = save_success
                        batch_failed_count += len(save_failed)
                        for code in save_failed:
                            stats["failed_stocks"].append({"code": code, "error": "保存失败"})

                    stats["stocks_success"] += batch_success_count
                    stats["stocks_failed"] += batch_failed_count

                except Exception as e:
                    logger.error(f"处理第 {batch_idx + 1} 批时出错: {e}", exc_info=True)
                    # 这批全部标记为失败
                    for stock_code in batch_stocks:
                        stats["stocks_failed"] += 1
                        stats["failed_stocks"].append({"code": stock_code, "error": f"批量处理失败: {e}"})

                # 更新进度条描述
                pbar.set_description(f"批量更新进度 | 成功: {stats['stocks_success']}, 失败: {stats['stocks_failed']}")

                # 注意：已取消批次间休眠！

        except KeyboardInterrupt:
            print("\n\n更新被用户中断")
        except Exception as e:
            print(f"\n\n更新过程中发生错误：{e}")
            logger.error(f"批量更新实时数据时发生错误: {e}", exc_info=True)

        # 批量更新追踪器（优化版，一次数据库操作）
        if all_success_codes:
            logger.info(f"批量更新追踪器：{len(all_success_codes)} 只股票")
            self.update_tracker.update_records_batch(
                all_success_codes,
                data_start_date=record_date,
                data_end_date=record_date,
            )

        stats["end_time"] = datetime.now()
        duration = (stats["end_time"] - stats["start_time"]).total_seconds()

        print("\n" + "=" * 80)
        print("实时数据批量更新完成！（优化版）")
        print("=" * 80)
        print(f"总股票数：{stats['total_stocks']}")
        print(f"成功：{stats['stocks_success']}")
        print(f"失败：{stats['stocks_failed']}")
        print(f"总批次数：{stats['batches']}")
        print(f"总耗时：{duration:.2f} 秒")
        print(f"平均速度：{stats['total_stocks'] / duration:.1f} 只/秒")

        if stats["failed_stocks"]:
            print(f"\n失败股票 ({len(stats['failed_stocks'])}):")
            for i, failed in enumerate(stats["failed_stocks"][:20], 1):
                print(f"  {i}. {failed['code']}: {failed['error']}")
            if len(stats["failed_stocks"]) > 20:
                print(f"  ... 还有 {len(stats['failed_stocks']) - 20} 只")

        print("=" * 80 + "\n")

        return stats


_realtime_updater_instance: Optional[RealtimeDataUpdater] = None


def get_realtime_updater(batch_size: int = 300) -> RealtimeDataUpdater:
    """
    获取全局实时数据更新器实例（优化版）

    Args:
        batch_size: 每批查询的股票数量（默认300只，优化版）

    Returns:
        RealtimeDataUpdater 实例
    """
    global _realtime_updater_instance
    if _realtime_updater_instance is None:
        _realtime_updater_instance = RealtimeDataUpdater(batch_size=batch_size)
    return _realtime_updater_instance
