# -*- coding: utf-8 -*-
"""
Stock Data Update Tracker - 数据更新状态追踪器

提供功能：
1. 记录每只股票的最后更新日期
2. 智能判断需要更新的日期范围（增量更新 vs 全量更新）
3. 支持按交易日批量获取多只股票数据
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import Boolean, Column, Date, DateTime, Integer, String, and_, func, select
from sqlalchemy.orm import declarative_base

from src.storage import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class StockUpdateRecord(Base):
    """
    股票数据更新记录表

    记录每只股票的最后更新日期和状态
    """

    __tablename__ = "stock_update_records"

    # 股票代码
    code = Column(String(10), primary_key=True, index=True)

    # 最后更新的最新日期
    last_updated_date = Column(Date, index=True, nullable=True)

    # 数据库中最早的数据日期
    first_data_date = Column(Date, index=True, nullable=True)

    # 更新次数统计
    update_count = Column(Integer, default=0)

    # 最后更新时间
    last_updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 是否活跃（用于标记退市股票等）
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<StockUpdateRecord(code={self.code}, last_updated={self.last_updated_date})>"


class DataUpdateTracker:
    """
    数据更新追踪器

    核心功能：
    1. 追踪每只股票的最后更新日期
    2. 智能判断更新策略：增量更新 vs 全量更新
    3. 支持批量按交易日获取多只股票数据
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        初始化数据更新追踪器

        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self._init_table()

    def _init_table(self):
        """初始化数据库表"""
        try:
            StockUpdateRecord.__table__.create(self.db_manager._engine, checkfirst=True)
            logger.info("Stock update records table initialized")
        except Exception as e:
            logger.warning(f"Failed to create stock update records table: {e}")

    def get_update_record(self, stock_code: str) -> Optional[StockUpdateRecord]:
        """
        获取股票的更新记录

        Args:
            stock_code: 股票代码

        Returns:
            更新记录或None
        """
        try:
            with self.db_manager.get_session() as session:
                stmt = select(StockUpdateRecord).where(StockUpdateRecord.code == stock_code)
                return session.execute(stmt).scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get update record for {stock_code}: {e}")
            return None

    def determine_update_range(
        self, stock_code: str, target_start_date: date, target_end_date: date
    ) -> Tuple[date, date, bool]:
        """
        智能判断需要更新的日期范围

        Args:
            stock_code: 股票代码
            target_start_date: 目标开始日期
            target_end_date: 目标结束日期

        Returns:
            Tuple of (actual_start_date, actual_end_date, is_incremental)
            - actual_start_date: 实际需要更新的开始日期
            - actual_end_date: 实际需要更新的结束日期
            - is_incremental: 是否为增量更新
        """
        record = self.get_update_record(stock_code)

        if record is None or record.last_updated_date is None:
            # 没有更新记录，全量更新
            logger.debug(f"{stock_code}: No update record, full update required")
            return target_start_date, target_end_date, False

        # 有更新记录，判断是否需要增量更新
        last_updated = record.last_updated_date

        if target_end_date <= last_updated:
            # 目标结束日期在最后更新日期之前，无需更新
            logger.debug(f"{stock_code}: Already up to date (last: {last_updated}, target: {target_end_date})")
            return None, None, True

        # 增量更新：从最后更新日期的下一天开始
        actual_start = last_updated + timedelta(days=1)
        actual_start = max(actual_start, target_start_date)

        logger.debug(f"{stock_code}: Incremental update from {actual_start} to {target_end_date}")
        return actual_start, target_end_date, True

    def update_record(
        self, stock_code: str, data_start_date: Optional[date] = None, data_end_date: Optional[date] = None
    ):
        """
        更新股票的更新记录

        Args:
            stock_code: 股票代码
            data_start_date: 数据的开始日期（可选）
            data_end_date: 数据的结束日期（可选）
        """
        try:
            with self.db_manager.get_session() as session:
                stmt = select(StockUpdateRecord).where(StockUpdateRecord.code == stock_code)
                record = session.execute(stmt).scalar_one_or_none()

                if record is None:
                    record = StockUpdateRecord(code=stock_code)
                    session.add(record)

                # 更新数据日期范围
                if data_start_date is not None:
                    if record.first_data_date is None or data_start_date < record.first_data_date:
                        record.first_data_date = data_start_date

                if data_end_date is not None:
                    if record.last_updated_date is None or data_end_date > record.last_updated_date:
                        record.last_updated_date = data_end_date

                if record.update_count is None:
                    record.update_count = 0
                record.update_count += 1
                record.last_updated_at = datetime.now()

                session.commit()
                logger.debug(f"Updated record for {stock_code}: last={record.last_updated_date}")
        except Exception as e:
            logger.error(f"Failed to update record for {stock_code}: {e}")
            if "session" in locals():
                session.rollback()

    def get_stocks_needing_update(self, stock_codes: List[str], target_date: date) -> Tuple[List[str], List[str]]:
        """
        获取需要更新的股票列表

        Args:
            stock_codes: 股票代码列表
            target_date: 目标更新日期

        Returns:
            Tuple of (stocks_needing_update, stocks_up_to_date)
        """
        needs_update = []
        up_to_date = []

        for code in stock_codes:
            record = self.get_update_record(code)
            if record is None or record.last_updated_date is None or record.last_updated_date < target_date:
                needs_update.append(code)
            else:
                up_to_date.append(code)

        logger.info(f"Stock update check: {len(needs_update)} need update, {len(up_to_date)} up to date")
        return needs_update, up_to_date

    def get_trading_days_between(self, start_date: date, end_date: date) -> List[date]:
        """
        获取两个日期之间的所有交易日（从数据库现有数据推断）

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        from src.storage import StockDaily

        try:
            with self.db_manager.get_session() as session:
                stmt = (
                    select(StockDaily.date)
                    .distinct()
                    .where(and_(StockDaily.date >= start_date, StockDaily.date <= end_date))
                    .order_by(StockDaily.date)
                )
                dates = session.execute(stmt).scalars().all()
                return list(dates)
        except Exception as e:
            logger.error(f"Failed to get trading days: {e}")
            return []

    def update_records_batch(
        self, stock_codes: List[str], data_start_date: Optional[date] = None, data_end_date: Optional[date] = None
    ):
        """
        批量更新股票的更新记录（性能优化版）

        Args:
            stock_codes: 股票代码列表
            data_start_date: 数据的开始日期（可选）
            data_end_date: 数据的结束日期（可选）
        """
        if not stock_codes:
            return

        try:
            with self.db_manager.get_session() as session:
                # 1. 查询所有已存在的记录
                stmt = select(StockUpdateRecord).where(StockUpdateRecord.code.in_(stock_codes))
                existing_records = session.execute(stmt).scalars().all()
                existing_dict = {rec.code: rec for rec in existing_records}

                # 2. 分别处理更新和插入
                new_records = []
                for code in stock_codes:
                    if code in existing_dict:
                        # 更新现有记录
                        record = existing_dict[code]
                        if data_start_date is not None:
                            if record.first_data_date is None or data_start_date < record.first_data_date:
                                record.first_data_date = data_start_date

                        if data_end_date is not None:
                            if record.last_updated_date is None or data_end_date > record.last_updated_date:
                                record.last_updated_date = data_end_date

                        if record.update_count is None:
                            record.update_count = 0
                        record.update_count += 1
                        record.last_updated_at = datetime.now()
                    else:
                        # 创建新记录
                        record = StockUpdateRecord(code=code)
                        if data_start_date is not None:
                            record.first_data_date = data_start_date
                        if data_end_date is not None:
                            record.last_updated_date = data_end_date
                        record.update_count = 1
                        new_records.append(record)

                # 3. 批量插入新记录
                if new_records:
                    session.add_all(new_records)

                # 4. 提交事务
                session.commit()
                logger.debug(f"Batch updated {len(stock_codes)} records")

        except Exception as e:
            logger.error(f"Failed to batch update records: {e}")
            if "session" in locals():
                session.rollback()

    def mark_inactive(self, stock_code: str):
        """标记股票为不活跃（如退市）"""
        try:
            with self.db_manager.get_session() as session:
                stmt = select(StockUpdateRecord).where(StockUpdateRecord.code == stock_code)
                record = session.execute(stmt).scalar_one_or_none()

                if record:
                    record.is_active = False
                    session.commit()
                    logger.info(f"Marked {stock_code} as inactive")
        except Exception as e:
            logger.error(f"Failed to mark {stock_code} as inactive: {e}")

    def repair_invalid_dates(self, stock_codes: Optional[List[str]] = None) -> int:
        """
        修复 tracker 中虚高的 last_updated_date，以及冷启动时从 StockDaily 自动创建 tracker 记录。

        1. 冷启动：如果 stock_update_records 表为空但 StockDaily 有数据，
           从 StockDaily 批量创建 tracker 记录
        2. 修复：对比 StockUpdateRecord.last_updated_date 与 StockDaily 中的实际 MAX(date)，
           若 tracker 记录日期大于实际数据日期，修正为实际值

        Args:
            stock_codes: 待修复的股票代码列表，为 None 时修复全部

        Returns:
            修复/创建的记录数
        """
        from src.storage import StockDaily

        repaired_count = 0
        try:
            with self.db_manager.get_session() as session:
                # --- 冷启动：检查 tracker 表是否为空 ---
                count_stmt = select(func.count()).select_from(StockUpdateRecord)
                tracker_count = session.execute(count_stmt).scalar()

                if tracker_count == 0:
                    # tracker 表为空，从 StockDaily 批量创建记录
                    logger.info("[冷启动] stock_update_records 为空，从 StockDaily 创建 tracker 记录...")

                    # 查询所有有数据的股票及其最新日期
                    if stock_codes:
                        cold_start_stmt = (
                            select(StockDaily.code, func.max(StockDaily.date).label("max_date"))
                            .where(StockDaily.code.in_(stock_codes))
                            .group_by(StockDaily.code)
                        )
                    else:
                        cold_start_stmt = (
                            select(StockDaily.code, func.max(StockDaily.date).label("max_date"))
                            .group_by(StockDaily.code)
                        )
                    cold_start_results = session.execute(cold_start_stmt).all()

                    new_records = []
                    for code, max_date_val in cold_start_results:
                        if max_date_val is not None:
                            record = StockUpdateRecord(
                                code=code,
                                last_updated_date=max_date_val,
                                first_data_date=None,  # 首次数据日期暂不追溯
                                update_count=1,
                            )
                            new_records.append(record)

                    if new_records:
                        session.add_all(new_records)
                        session.commit()
                        repaired_count = len(new_records)
                        logger.info(f"[冷启动] 从 StockDaily 创建了 {repaired_count} 条 tracker 记录")
                    else:
                        logger.info("[冷启动] StockDaily 中无数据，跳过 tracker 创建")

                    return repaired_count

                # --- 常规修复：修正虚高的 last_updated_date ---
                stmt = select(StockUpdateRecord).where(StockUpdateRecord.last_updated_date.isnot(None))
                if stock_codes:
                    stmt = stmt.where(StockUpdateRecord.code.in_(stock_codes))
                records = session.execute(stmt).scalars().all()

                for record in records:
                    max_date_stmt = (
                        select(StockDaily.date)
                        .where(StockDaily.code == record.code)
                        .order_by(StockDaily.date.desc())
                        .limit(1)
                    )
                    max_date_result = session.execute(max_date_stmt).scalar_one_or_none()

                    if max_date_result is None:
                        continue

                    actual_max_date = max_date_result if isinstance(max_date_result, date) else max_date_result

                    if record.last_updated_date > actual_max_date:
                        old_date = record.last_updated_date
                        record.last_updated_date = actual_max_date
                        repaired_count += 1
                        logger.debug(
                            f"修复 {record.code}: last_updated {old_date} -> {actual_max_date}"
                        )

                if repaired_count > 0:
                    session.commit()
                    logger.info(f"修复了 {repaired_count} 条 tracker 记录")

        except Exception as e:
            logger.error(f"修复 tracker 记录失败: {e}")
            if "session" in locals():
                session.rollback()

        return repaired_count


def get_update_tracker() -> DataUpdateTracker:
    """获取全局数据更新追踪器实例"""
    if not hasattr(get_update_tracker, "_instance"):
        get_update_tracker._instance = DataUpdateTracker()
    return get_update_tracker._instance
