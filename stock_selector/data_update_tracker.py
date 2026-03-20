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
from typing import Optional, List, Tuple, Dict, Set
from pathlib import Path

from sqlalchemy import Column, String, Date, DateTime, Integer, Boolean
from sqlalchemy import select, and_, func
from sqlalchemy.orm import declarative_base

from src.storage import DatabaseManager

logger = logging.getLogger(__name__)

Base = declarative_base()


class StockUpdateRecord(Base):
    """
    股票数据更新记录表
    
    记录每只股票的最后更新日期和状态
    """
    __tablename__ = 'stock_update_records'
    
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
        self,
        stock_code: str,
        target_start_date: date,
        target_end_date: date
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
        self,
        stock_code: str,
        data_start_date: Optional[date] = None,
        data_end_date: Optional[date] = None
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
                
                record.update_count += 1
                record.last_updated_at = datetime.now()
                
                session.commit()
                logger.debug(f"Updated record for {stock_code}: last={record.last_updated_date}")
        except Exception as e:
            logger.error(f"Failed to update record for {stock_code}: {e}")
            if 'session' in locals():
                session.rollback()
    
    def get_stocks_needing_update(
        self,
        stock_codes: List[str],
        target_date: date
    ) -> Tuple[List[str], List[str]]:
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
    
    def get_trading_days_between(
        self,
        start_date: date,
        end_date: date
    ) -> List[date]:
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
                    .where(and_(
                        StockDaily.date >= start_date,
                        StockDaily.date <= end_date
                    ))
                    .order_by(StockDaily.date)
                )
                dates = session.execute(stmt).scalars().all()
                return list(dates)
        except Exception as e:
            logger.error(f"Failed to get trading days: {e}")
            return []
    
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


def get_update_tracker() -> DataUpdateTracker:
    """获取全局数据更新追踪器实例"""
    if not hasattr(get_update_tracker, '_instance'):
        get_update_tracker._instance = DataUpdateTracker()
    return get_update_tracker._instance
