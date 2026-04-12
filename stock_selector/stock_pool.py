# -*- coding: utf-8 -*-
"""
Stock Pool Module - Manage stock pool for stock selector.

Provides functionality to:
1. Fetch complete stock list (Shanghai A + Shenzhen A)
2. Cache stock list in database
3. Provide stock list to stock selector
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import Column, Date, String, DateTime, select, and_, desc
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class StockPoolItem(Base):
    """
    Stock Pool Item Model
    
    Stores stock information for the complete stock pool.
    """
    __tablename__ = 'stock_pool'
    
    # Stock code (e.g., 600519, 000001)
    code = Column(String(10), primary_key=True, index=True)
    
    # Stock name
    name = Column(String(50), nullable=False)
    
    # Market (SH/SZ)
    market = Column(String(2), index=True)
    
    # List date
    list_date = Column(Date)
    
    # Last update time
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<StockPoolItem(code={self.code}, name={self.name}, market={self.market})>"


class StockPoolManager:
    """
    Stock Pool Manager
    
    Manages the complete stock pool (Shanghai A + Shenzhen A).
    """
    
    def __init__(self, db_manager=None, data_fetcher_manager=None):
        """
        Initialize stock pool manager.
        
        Args:
            db_manager: Database manager instance
            data_fetcher_manager: Data fetcher manager instance
        """
        self.db_manager = db_manager
        self.data_fetcher_manager = data_fetcher_manager
        self._init_dependencies()
    
    def _init_dependencies(self):
        """Initialize dependencies if not provided."""
        if self.db_manager is None:
            from src.storage import get_db
            self.db_manager = get_db()
        
        if self.data_fetcher_manager is None:
            from data_provider import DataFetcherManager
            self.data_fetcher_manager = DataFetcherManager()
        
        # Create table if not exists
        self._create_table()
    
    def _create_table(self):
        """Create stock pool table if not exists."""
        try:
            StockPoolItem.__table__.create(self.db_manager._engine, checkfirst=True)
            logger.info("Stock pool table initialized")
        except Exception as e:
            logger.warning(f"Failed to create stock pool table: {e}")
    
    def is_cache_valid(self) -> bool:
        """
        Check if the stock pool data exists in database.
        
        Returns:
            True if stock pool data exists, False otherwise
        """
        try:
            with self.db_manager.get_session() as session:
                # Check if there are any stock items in the pool
                result = session.execute(
                    select(StockPoolItem)
                    .limit(1)
                ).scalar_one_or_none()
                
                return result is not None
        except Exception as e:
            logger.warning(f"Failed to check stock pool existence: {e}")
            return False
    
    def get_stock_list(self, force_refresh: bool = False) -> List[str]:
        """
        Get the complete stock list.
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            List of stock codes
        """
        if not force_refresh and self.is_cache_valid():
            logger.info("Using cached stock pool")
            return self._get_cached_stock_list()
        
        logger.info("Fetching stock list from data sources...")
        stock_list = self._fetch_stock_list_from_sources()
        
        if stock_list:
            self._save_stock_list_to_cache(stock_list)
            logger.info(f"Fetched and cached {len(stock_list)} stocks")
        
        return [code for code, _, _ in stock_list]
    
    def _get_cached_stock_list(self) -> List[str]:
        """Get stock list from cache."""
        try:
            with self.db_manager.get_session() as session:
                results = session.execute(
                    select(StockPoolItem.code)
                    .order_by(StockPoolItem.code)
                ).scalars().all()
                
                return list(results)
        except Exception as e:
            logger.error(f"Failed to get cached stock list: {e}")
            return []
    
    def _fetch_stock_list_from_sources(self) -> List[tuple]:
        """
        Fetch stock list from data sources.
        
        Returns:
            List of tuples (code, name, market)
        """
        # Try Tushare first (most reliable)
        stock_list = self._fetch_from_tushare()
        if stock_list:
            return stock_list
        
        # Try Baostock
        stock_list = self._fetch_from_baostock()
        if stock_list:
            return stock_list
        
        # Try Pytdx
        stock_list = self._fetch_from_pytdx()
        if stock_list:
            return stock_list
        
        logger.error("All data sources failed to fetch stock list")
        return []
    
    def _fetch_from_tushare(self) -> List[tuple]:
        """Fetch stock list from Tushare."""
        try:
            for fetcher in self.data_fetcher_manager._fetchers:
                if fetcher.name == "TushareFetcher" and hasattr(fetcher, 'get_stock_list'):
                    df = fetcher.get_stock_list()
                    if df is not None and not df.empty:
                        stock_list = []
                        for _, row in df.iterrows():
                            code = row.get('code')
                            name = row.get('name')
                            market = row.get('market', '')
                            if code and name:
                                stock_list.append((code, name, market))
                        logger.info(f"Fetched {len(stock_list)} stocks from Tushare")
                        return stock_list
        except Exception as e:
            logger.warning(f"Failed to fetch from Tushare: {e}")
        return []
    
    def _fetch_from_baostock(self) -> List[tuple]:
        """Fetch stock list from Baostock."""
        try:
            for fetcher in self.data_fetcher_manager._fetchers:
                if fetcher.name == "BaostockFetcher" and hasattr(fetcher, 'get_stock_list'):
                    df = fetcher.get_stock_list()
                    if df is not None and not df.empty:
                        stock_list = []
                        for _, row in df.iterrows():
                            code = row.get('code')
                            name = row.get('name')
                            if code and name:
                                # Determine market
                                market = 'SH' if code.startswith('6') else 'SZ'
                                stock_list.append((code, name, market))
                        logger.info(f"Fetched {len(stock_list)} stocks from Baostock")
                        return stock_list
        except Exception as e:
            logger.warning(f"Failed to fetch from Baostock: {e}")
        return []
    
    def _fetch_from_pytdx(self) -> List[tuple]:
        """Fetch stock list from Pytdx."""
        try:
            for fetcher in self.data_fetcher_manager._fetchers:
                if fetcher.name == "PytdxFetcher":
                    # Check if we can access the stock list cache
                    if hasattr(fetcher, '_stock_list_cache'):
                        # Try to trigger cache loading
                        if hasattr(fetcher, 'get_stock_name'):
                            fetcher.get_stock_name('600519')
                        
                        if fetcher._stock_list_cache:
                            stock_list = []
                            for code, name in fetcher._stock_list_cache.items():
                                market = 'SH' if code.startswith('6') else 'SZ'
                                stock_list.append((code, name, market))
                            logger.info(f"Fetched {len(stock_list)} stocks from Pytdx")
                            return stock_list
        except Exception as e:
            logger.warning(f"Failed to fetch from Pytdx: {e}")
        return []
    
    def _save_stock_list_to_cache(self, stock_list: List[tuple]):
        """
        Save stock list to database.
        
        Args:
            stock_list: List of tuples (code, name, market)
        """
        try:
            with self.db_manager.get_session() as session:
                # Delete old data
                session.query(StockPoolItem).delete()
                
                # Insert new stock list
                for code, name, market in stock_list:
                    item = StockPoolItem(
                        code=code,
                        name=name,
                        market=market
                    )
                    session.add(item)
                
                session.commit()
                logger.info(f"Saved {len(stock_list)} stocks to database")
        except Exception as e:
            logger.error(f"Failed to save stock list to database: {e}")
            if 'session' in locals():
                session.rollback()


# Global stock pool manager instance
_stock_pool_manager: Optional[StockPoolManager] = None


def get_stock_pool_manager() -> StockPoolManager:
    """Get the global stock pool manager instance."""
    global _stock_pool_manager
    if _stock_pool_manager is None:
        _stock_pool_manager = StockPoolManager()
    return _stock_pool_manager


def filter_beijing_stock_exchange(stock_codes: List[str]) -> List[str]:
    """
    过滤北交所的股票代码（8开头和92开头）
    
    Args:
        stock_codes: 股票代码列表
        
    Returns:
        过滤后的股票代码列表
    """
    original_count = len(stock_codes)
    filtered_codes = [
        code for code in stock_codes 
        if not (code.startswith('8') or code.startswith('92'))
    ]
    filtered_count = original_count - len(filtered_codes)
    if filtered_count > 0:
        logger.info(f"过滤掉 {filtered_count} 只北交所股票（8开头和92开头）")
    return filtered_codes


def filter_special_stock_codes(stock_codes: List[str]) -> List[str]:
    """
    过滤特定开头的股票代码（科创板、创业板等）
    
    过滤规则：
    - 688开头：科创板
    - 689开头：科创板
    - 300开头：创业板
    - 301开头：创业板
    - 43开头：北交所老股
    - 83开头：北交所
    - 87开头：北交所
    - 92开头：北交所
    
    Args:
        stock_codes: 股票代码列表
        
    Returns:
        过滤后的股票代码列表
    """
    original_count = len(stock_codes)
    filtered_prefixes = ['688', '689', '300', '301', '302', '43', '83', '87', '92']
    
    filtered_codes = [
        code for code in stock_codes 
        if not any(code.startswith(prefix) for prefix in filtered_prefixes)
    ]
    
    filtered_count = original_count - len(filtered_codes)
    if filtered_count > 0:
        logger.info(f"过滤掉 {filtered_count} 只特定板块股票（{', '.join(filtered_prefixes)}开头）")
    return filtered_codes


def is_st_stock(stock_name: Optional[str]) -> bool:
    """
    判断股票是否为ST股票（特别处理股票）
    
    Args:
        stock_name: 股票名称
        
    Returns:
        如果是ST股票返回True，否则返回False
    """
    if not stock_name:
        return False
    
    stock_name_upper = stock_name.upper()
    
    return any(keyword in stock_name_upper for keyword in ['ST', '*ST', 'SST', 'S*ST'])


def filter_st_stocks(stock_code_name_pairs: List[tuple]) -> List[tuple]:
    """
    过滤ST股票
    
    Args:
        stock_code_name_pairs: 股票代码和名称的元组列表 [(code, name), ...]
        
    Returns:
        过滤后的股票代码和名称列表
    """
    original_count = len(stock_code_name_pairs)
    filtered_pairs = [
        (code, name) for code, name in stock_code_name_pairs 
        if not is_st_stock(name)
    ]
    
    filtered_count = original_count - len(filtered_pairs)
    if filtered_count > 0:
        logger.info(f"过滤掉 {filtered_count} 只ST股票")
    
    return filtered_pairs


def get_all_stock_codes(force_refresh: bool = False) -> List[str]:
    """
    Get all stock codes from the stock pool.
    
    Args:
        force_refresh: Force refresh even if cache is valid
        
    Returns:
        List of all stock codes (including those without data, validation will be done separately)
    """
    manager = get_stock_pool_manager()
    return manager.get_stock_list(force_refresh=force_refresh)


def get_all_stock_code_name_pairs(force_refresh: bool = False) -> List[tuple]:
    """
    获取所有股票代码和名称的配对列表
    
    Args:
        force_refresh: 是否强制刷新
        
    Returns:
        股票代码和名称的元组列表 [(code, name), ...]
    """
    manager = get_stock_pool_manager()
    if not force_refresh and manager.is_cache_valid():
        try:
            with manager.db_manager.get_session() as session:
                results = session.execute(
                    select(StockPoolItem.code, StockPoolItem.name)
                    .order_by(StockPoolItem.code)
                ).all()
                
                return [(row.code, row.name) for row in results]
        except Exception as e:
            logger.error(f"Failed to get cached stock code name pairs: {e}")
            return []
    
    logger.info("Fetching stock list from data sources...")
    stock_list = manager._fetch_stock_list_from_sources()
    
    if stock_list:
        manager._save_stock_list_to_cache(stock_list)
        logger.info(f"Fetched and cached {len(stock_list)} stocks")
    
    return [(code, name) for code, name, _ in stock_list]
