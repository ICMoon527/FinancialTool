# -*- coding: utf-8 -*-
"""
板块数据管理器 - 负责板块数据的获取、缓存和分析
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class SectorInfo:
    """板块信息"""
    name: str
    change_pct: float
    stock_count: int = 0
    limit_up_count: int = 0
    leading_stocks: List[str] = field(default_factory=list)


class SectorManager:
    """
    板块数据管理器
    
    负责：
    1. 获取全市场板块列表
    2. 获取股票所属板块
    3. 板块数据缓存（内存 + 数据库持久化）
    4. 板块热点分析
    """

    def __init__(self, data_manager=None, db_manager=None, config=None):
        """
        初始化板块管理器
        
        Args:
            data_manager: DataFetcherManager 实例
            db_manager: DatabaseManager 实例
            config: StockSelectorConfig 实例
        """
        self._data_manager = data_manager
        self._db_manager = db_manager or DatabaseManager.get_instance()
        self._config = config
        self._sector_cache: Dict[str, List[Dict]] = {}
        self._sector_cache_timestamp: float = 0
        self._sector_cache_ttl: int = 1800  # 30分钟缓存
        self._stock_sector_map: Dict[str, List[str]] = {}  # 股票代码 -> 板块列表
        self._stock_sector_map_timestamp: float = 0

    def _is_cache_valid(self, timestamp: float) -> bool:
        """检查缓存是否有效"""
        return time.time() - timestamp < self._sector_cache_ttl

    def get_sector_rankings(self, n: int = 5, force_refresh: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        获取板块涨跌榜
        
        Args:
            n: 返回前n个
            force_refresh: 是否强制刷新
            
        Returns:
            Tuple: (领涨板块列表, 领跌板块列表)
        """
        # 检查缓存
        if not force_refresh and self._is_cache_valid(self._sector_cache_timestamp):
            logger.debug("[板块] 使用缓存的板块数据")
            top = self._sector_cache.get('top', [])[:n]
            bottom = self._sector_cache.get('bottom', [])[:n]
            return top, bottom

        # 从数据源获取
        if self._data_manager:
            try:
                logger.info("[板块] 从数据源获取板块排行...")
                top, bottom = self._data_manager.get_sector_rankings(n)
                
                # 更新缓存
                self._sector_cache = {'top': top, 'bottom': bottom}
                self._sector_cache_timestamp = time.time()
                
                logger.info(f"[板块] 获取成功，领涨: {[s['name'] for s in top]}, 领跌: {[s['name'] for s in bottom]}")
                return top, bottom
            except Exception as e:
                logger.error(f"[板块] 获取板块排行失败: {e}")
        
        return [], []

    def get_stock_sectors(self, stock_code: str) -> List[str]:
        """
        获取股票所属板块
        
        Args:
            stock_code: 股票代码
            
        Returns:
            板块名称列表
        """
        # 检查内存缓存
        if stock_code in self._stock_sector_map and self._is_cache_valid(self._stock_sector_map_timestamp):
            logger.debug(f"[板块] 使用内存缓存的股票 {stock_code} 板块数据")
            return self._stock_sector_map[stock_code]

        # 检查是否需要强制更新
        force_update = False
        if self._config:
            force_update = self._config.update_sector_data

        # 先尝试从数据库获取，除非强制更新
        if not force_update:
            try:
                db_sectors = self._db_manager.get_stock_sectors(stock_code)
                if db_sectors:
                    logger.debug(f"[板块] 使用数据库存储的股票 {stock_code} 板块数据")
                    # 更新内存缓存
                    self._stock_sector_map[stock_code] = db_sectors
                    self._stock_sector_map_timestamp = time.time()
                    return db_sectors
            except Exception as e:
                logger.error(f"[板块] 从数据库获取股票 {stock_code} 板块数据失败: {e}")

        # 从数据源获取
        if self._data_manager:
            try:
                logger.info(f"[板块] 从数据源获取股票 {stock_code} 的所属板块...")
                sectors = self._data_manager.get_stock_sectors(stock_code)
                
                if sectors:
                    # 更新内存缓存
                    self._stock_sector_map[stock_code] = sectors
                    self._stock_sector_map_timestamp = time.time()
                    
                    # 保存到数据库
                    try:
                        self._db_manager.save_stock_sectors(stock_code, sectors, "sector_manager")
                        logger.info(f"[板块] 股票 {stock_code} 板块数据已保存到数据库")
                    except Exception as e:
                        logger.error(f"[板块] 保存股票 {stock_code} 板块数据到数据库失败: {e}")
                    
                    logger.info(f"[板块] 股票 {stock_code} 所属板块: {sectors}")
                    return sectors
            except Exception as e:
                logger.error(f"[板块] 获取股票 {stock_code} 所属板块失败: {e}")
        
        return []

    def is_hot_sector(self, sector_name: str, threshold_pct: float = 2.0) -> bool:
        """
        判断板块是否为热点板块
        
        Args:
            sector_name: 板块名称
            threshold_pct: 涨幅阈值，默认2%
            
        Returns:
            bool: 是否为热点板块
        """
        try:
            top, _ = self.get_sector_rankings(n=20)
            
            for sector in top:
                if sector.get('name') == sector_name:
                    change_pct = sector.get('change_pct', 0)
                    return change_pct >= threshold_pct
            
            return False
        except Exception as e:
            logger.error(f"[板块] 判断热点板块失败: {e}")
            return False

    def get_sector_change_pct(self, sector_name: str) -> Optional[float]:
        """
        获取板块涨跌幅
        
        Args:
            sector_name: 板块名称
            
        Returns:
            板块涨跌幅，找不到返回 None
        """
        try:
            top, bottom = self.get_sector_rankings(n=50)
            all_sectors = top + bottom
            
            for sector in all_sectors:
                if sector.get('name') == sector_name:
                    return sector.get('change_pct')
            
            return None
        except Exception as e:
            logger.error(f"[板块] 获取板块涨跌幅失败: {e}")
            return None

    def is_stock_in_hot_sector(self, stock_code: str, threshold_pct: float = 2.0) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        判断股票是否在热点板块中
        
        Args:
            stock_code: 股票代码
            threshold_pct: 涨幅阈值
            
        Returns:
            Tuple: (是否在热点板块, 板块名称, 板块涨跌幅)
        """
        sectors = self.get_stock_sectors(stock_code)
        
        if not sectors:
            return False, None, None
        
        for sector_name in sectors:
            if self.is_hot_sector(sector_name, threshold_pct):
                change_pct = self.get_sector_change_pct(sector_name)
                return True, sector_name, change_pct
        
        return False, None, None

    def clear_cache(self) -> None:
        """清除缓存"""
        self._sector_cache = {}
        self._sector_cache_timestamp = 0
        self._stock_sector_map = {}
        self._stock_sector_map_timestamp = 0
        logger.info("[板块] 缓存已清除")
