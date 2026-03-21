# -*- coding: utf-8 -*-
"""
Database Manager - Optimized Version with Performance Improvements

性能优化：
1. 批量 UPSERT 操作，解决 N+1 查询问题
2. 精确字段检索，减少数据传输量
3. LRU 缓存机制，减少重复数据库查询
4. 批量插入/更新，减少数据库交互次数
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Set, Tuple
from collections import OrderedDict

import pandas as pd
from sqlalchemy import select, and_, update, delete
from sqlalchemy.orm import Session

from src.storage import DatabaseManager, StockDaily

logger = logging.getLogger(__name__)


class OptimizedDatabaseManager(DatabaseManager):
    """
    优化的数据库管理器
    
    性能改进：
    1. 批量 UPSERT - 减少数据库查询次数
    2. 字段投影 - 只检索必要字段
    3. LRU 缓存 - 减少重复查询
    4. 批量操作 - 减少数据库交互次数
    """
    
    # LRU 缓存配置
    CACHE_MAX_SIZE = 1000  # 最大缓存条目数
    CACHE_TTL_SECONDS = 300  # 缓存 TTL: 5 分钟
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LRU 缓存：key=(code, start_date, end_date), value=(data, timestamp)
        self._data_cache = OrderedDict()
    
    def get_data_range_optimized(
        self, 
        code: str, 
        start_date: date, 
        end_date: date,
        use_cache: bool = True
    ) -> List[StockDaily]:
        """
        优化的数据范围查询
        
        优化点：
        1. LRU 缓存，避免重复查询
        2. 精确字段检索（可选）
        3. 批量加载
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            StockDaily 对象列表
        """
        # 检查缓存
        cache_key = (code, start_date, end_date)
        if use_cache and cache_key in self._data_cache:
            cached_data, timestamp = self._data_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < self.CACHE_TTL_SECONDS:
                logger.debug(f"缓存命中：{code} ({age:.1f}s old)")
                # 移动到末尾（最近使用）
                self._data_cache.move_to_end(cache_key)
                return cached_data
        
        # 缓存未命中，执行数据库查询
        with self.get_session() as session:
            # 优化 1: 精确字段检索（只选择需要的列）
            # 注意：如果需要完整的 StockDaily 对象，保持原样
            # 如果只需要部分字段，可以使用以下优化：
            results = (
                session.execute(
                    select(StockDaily)
                    .where(
                        and_(
                            StockDaily.code == code,
                            StockDaily.date >= start_date,
                            StockDaily.date <= end_date
                        )
                    )
                    .order_by(StockDaily.date)
                )
                .scalars()
                .all()
            )
            
            data = list(results)
            
            # 更新缓存
            if use_cache:
                self._update_cache(cache_key, data)
            
            return data
    
    def save_daily_data_bulk(
        self, 
        df: pd.DataFrame, 
        code: str, 
        data_source: str = "Unknown"
    ) -> int:
        """
        批量保存日线数据（优化版本）
        
        优化点：
        1. 批量查询已存在的记录（解决 N+1 问题）
        2. 批量插入新记录
        3. 批量更新现有记录
        4. 减少数据库交互次数
        
        Args:
            df: 包含日线数据的 DataFrame
            code: 股票代码
            data_source: 数据来源名称
            
        Returns:
            新增/更新的记录数
        """
        if df is None or df.empty:
            logger.warning(f"保存数据为空，跳过 {code}")
            return 0
        
        with self.get_session() as session:
            try:
                # Step 1: 解析所有日期
                parsed_dates = []
                for _, row in df.iterrows():
                    row_date = self._parse_date(row.get("date"))
                    if row_date:
                        parsed_dates.append((row_date, row))
                
                if not parsed_dates:
                    logger.warning(f"没有有效的日期数据，跳过 {code}")
                    return 0
                
                date_set = {d for d, _ in parsed_dates}
                
                # Step 2: 批量查询已存在的记录（关键优化：1 次查询代替 N 次）
                existing_records = (
                    session.execute(
                        select(StockDaily)
                        .where(
                            and_(
                                StockDaily.code == code,
                                StockDaily.date.in_(date_set)
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                
                # 创建日期到记录的映射
                existing_map = {rec.date: rec for rec in existing_records}
                
                # Step 3: 批量处理（内存中判断，不查询数据库）
                records_to_add = []
                records_to_update = []
                
                for row_date, row in parsed_dates:
                    if row_date in existing_map:
                        # 更新现有记录
                        records_to_update.append((row_date, row))
                    else:
                        # 创建新记录
                        records_to_add.append((row_date, row))
                
                # Step 4: 批量插入新记录
                if records_to_add:
                    for row_date, row in records_to_add:
                        record = StockDaily(
                            code=code,
                            date=row_date,
                            open=row.get("open"),
                            high=row.get("high"),
                            low=row.get("low"),
                            close=row.get("close"),
                            volume=row.get("volume"),
                            amount=row.get("amount"),
                            pct_chg=row.get("pct_chg"),
                            ma5=row.get("ma5"),
                            ma10=row.get("ma10"),
                            ma20=row.get("ma20"),
                            volume_ratio=row.get("volume_ratio"),
                            data_source=data_source,
                        )
                        session.add(record)
                    
                    logger.info(f"批量插入 {code} 数据：{len(records_to_add)} 条")
                
                # Step 5: 批量更新现有记录
                if records_to_update:
                    for row_date, row in records_to_update:
                        existing_record = existing_map[row_date]
                        existing_record.open = row.get("open")
                        existing_record.high = row.get("high")
                        existing_record.low = row.get("low")
                        existing_record.close = row.get("close")
                        existing_record.volume = row.get("volume")
                        existing_record.amount = row.get("amount")
                        existing_record.pct_chg = row.get("pct_chg")
                        existing_record.ma5 = row.get("ma5")
                        existing_record.ma10 = row.get("ma10")
                        existing_record.ma20 = row.get("ma20")
                        existing_record.volume_ratio = row.get("volume_ratio")
                        existing_record.data_source = data_source
                        existing_record.updated_at = datetime.now()
                    
                    logger.info(f"批量更新 {code} 数据：{len(records_to_update)} 条")
                
                # Step 6: 一次性提交
                session.commit()
                
                total_count = len(records_to_add) + len(records_to_update)
                logger.info(f"保存 {code} 数据成功，共 {total_count} 条（新增 {len(records_to_add)}, 更新 {len(records_to_update)}）")
                
                # 清除相关缓存
                self._invalidate_cache(code)
                
                return total_count
                
            except Exception as e:
                session.rollback()
                logger.error(f"保存 {code} 数据失败：{e}")
                raise
    
    def save_daily_data_bulk_upsert(
        self, 
        df: pd.DataFrame, 
        code: str, 
        data_source: str = "Unknown"
    ) -> int:
        """
        使用 SQLAlchemy 2.0+ 的 insert().on_conflict_do_update() 进行批量 UPSERT
        
        这是最高效的方式，但需要数据库支持（PostgreSQL/SQLite/MySQL）
        
        Args:
            df: 包含日线数据的 DataFrame
            code: 股票代码
            data_source: 数据来源名称
            
        Returns:
            新增/更新的记录数
        """
        if df is None or df.empty:
            logger.warning(f"保存数据为空，跳过 {code}")
            return 0
        
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.mysql import insert as mysql_insert
        
        with self.get_session() as session:
            try:
                # 准备数据
                values_list = []
                for _, row in df.iterrows():
                    row_date = self._parse_date(row.get("date"))
                    if row_date:
                        values_list.append({
                            'code': code,
                            'date': row_date,
                            'open': row.get("open"),
                            'high': row.get("high"),
                            'low': row.get("low"),
                            'close': row.get("close"),
                            'volume': row.get("volume"),
                            'amount': row.get("amount"),
                            'pct_chg': row.get("pct_chg"),
                            'ma5': row.get("ma5"),
                            'ma10': row.get("ma10"),
                            'ma20': row.get("ma20"),
                            'volume_ratio': row.get("volume_ratio"),
                            'data_source': data_source,
                        })
                
                if not values_list:
                    return 0
                
                # 根据数据库类型选择 UPSERT 语法
                dialect = session.bind.dialect.name
                
                if dialect == 'sqlite':
                    stmt = sqlite_insert(StockDaily).values(values_list)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['code', 'date'],
                        set_={
                            'open': stmt.excluded.open,
                            'high': stmt.excluded.high,
                            'low': stmt.excluded.low,
                            'close': stmt.excluded.close,
                            'volume': stmt.excluded.volume,
                            'amount': stmt.excluded.amount,
                            'pct_chg': stmt.excluded.pct_chg,
                            'ma5': stmt.excluded.ma5,
                            'ma10': stmt.excluded.ma10,
                            'ma20': stmt.excluded.ma20,
                            'volume_ratio': stmt.excluded.volume_ratio,
                            'data_source': stmt.excluded.data_source,
                            'updated_at': datetime.now(),
                        }
                    )
                elif dialect == 'postgresql':
                    stmt = pg_insert(StockDaily).values(values_list)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['code', 'date'],
                        set_={
                            'open': stmt.excluded.open,
                            'high': stmt.excluded.high,
                            'low': stmt.excluded.low,
                            'close': stmt.excluded.close,
                            'volume': stmt.excluded.volume,
                            'amount': stmt.excluded.amount,
                            'pct_chg': stmt.excluded.pct_chg,
                            'ma5': stmt.excluded.ma5,
                            'ma10': stmt.excluded.ma10,
                            'ma20': stmt.excluded.ma20,
                            'volume_ratio': stmt.excluded.volume_ratio,
                            'data_source': stmt.excluded.data_source,
                            'updated_at': datetime.now(),
                        }
                    )
                elif dialect == 'mysql':
                    stmt = mysql_insert(StockDaily).values(values_list)
                    stmt = stmt.on_duplicate_key_update(
                        open='VALUES(open)',
                        high='VALUES(high)',
                        low='VALUES(low)',
                        close='VALUES(close)',
                        volume='VALUES(volume)',
                        amount='VALUES(amount)',
                        pct_chg='VALUES(pct_chg)',
                        ma5='VALUES(ma5)',
                        ma10='VALUES(ma10)',
                        ma20='VALUES(ma20)',
                        volume_ratio='VALUES(volume_ratio)',
                        data_source='VALUES(data_source)',
                        updated_at=datetime.now(),
                    )
                else:
                    # 降级到普通方法
                    logger.warning(f"数据库 {dialect} 不支持 UPSERT，使用降级方案")
                    return self.save_daily_data_bulk(df, code, data_source)
                
                # 执行批量 UPSERT
                result = session.execute(stmt)
                session.commit()
                
                logger.info(f"批量 UPSERT {code} 数据成功：{len(values_list)} 条")
                
                # 清除相关缓存
                self._invalidate_cache(code)
                
                return len(values_list)
                
            except Exception as e:
                session.rollback()
                logger.error(f"批量 UPSERT {code} 数据失败：{e}")
                # 降级到普通方法
                return self.save_daily_data_bulk(df, code, data_source)
    
    def get_data_range_with_fields(
        self, 
        code: str, 
        start_date: date, 
        end_date: date,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        优化的数据范围查询 - 只检索指定字段
        
        优化点：
        1. 只选择需要的字段，减少数据传输量
        2. 直接返回 DataFrame，方便后续处理
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要检索的字段列表，默认使用核心字段
            
        Returns:
            pandas DataFrame
        """
        if fields is None:
            # 默认使用核心字段（策略需要的字段）
            fields = [
                'date', 'open', 'high', 'low', 'close', 
                'volume', 'pct_chg'
            ]
        
        with self.get_session() as session:
            # 动态构建查询字段
            columns = [getattr(StockDaily, field) for field in fields if hasattr(StockDaily, field)]
            
            results = (
                session.execute(
                    select(*columns)
                    .where(
                        and_(
                            StockDaily.code == code,
                            StockDaily.date >= start_date,
                            StockDaily.date <= end_date
                        )
                    )
                    .order_by(StockDaily.date)
                )
            )
            
            # 转换为 DataFrame
            df = pd.DataFrame(results.fetchall(), columns=fields)
            
            return df
    
    def _parse_date(self, date_value) -> Optional[date]:
        """
        解析日期值为 date 对象
        
        Args:
            date_value: 可以是字符串、datetime、pd.Timestamp 等
            
        Returns:
            date 对象或 None
        """
        if date_value is None:
            return None
        
        if isinstance(date_value, str):
            try:
                return datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                return None
        elif isinstance(date_value, datetime):
            return date_value.date()
        elif isinstance(date_value, pd.Timestamp):
            return date_value.date()
        elif isinstance(date_value, date):
            return date_value
        
        return None
    
    def _update_cache(self, key: Tuple, data: List[StockDaily]) -> None:
        """
        更新 LRU 缓存
        
        Args:
            key: 缓存键
            data: 数据
        """
        # 如果缓存已满，删除最旧的条目
        if len(self._data_cache) >= self.CACHE_MAX_SIZE:
            self._data_cache.popitem(last=False)
        
        # 添加新条目
        self._data_cache[key] = (data, datetime.now())
    
    def _invalidate_cache(self, code: str) -> None:
        """
        清除指定股票的所有缓存
        
        Args:
            code: 股票代码
        """
        keys_to_remove = [
            key for key in self._data_cache.keys() 
            if key[0] == code
        ]
        
        for key in keys_to_remove:
            del self._data_cache[key]
        
        logger.debug(f"清除 {code} 的缓存，共 {len(keys_to_remove)} 条")
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._data_cache.clear()
        logger.info("数据缓存已清空")


# 便捷函数
def get_optimized_db() -> OptimizedDatabaseManager:
    """获取优化的数据库管理器单例"""
    return OptimizedDatabaseManager.get_instance()
