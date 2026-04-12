# -*- coding: utf-8 -*-
"""
板块数据管理器 - 负责板块数据的获取、缓存和分析
"""

import logging
import time
from datetime import date
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
        self._heat_cache: Optional[Tuple[Dict[str, float], List[str]]] = None
        self._heat_cache_timestamp: float = 0
        self._heat_cache_ttl: int = 1800  # 30分钟

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

        # 优先从数据库获取最新的板块数据（如果不强制刷新）
        if not force_refresh and self._db_manager:
            try:
                logger.debug("[板块] 尝试从数据库获取板块数据...")
                from datetime import date
                current_date = date.today()
                # 从数据库获取今天的所有板块数据
                all_sectors = self._db_manager.get_all_sectors()
                if all_sectors:
                    # 从数据库获取每个板块的最新数据
                    sector_list = []
                    for sector_name in all_sectors:
                        sector_daily = self._db_manager.get_sector_daily(sector_name, current_date)
                        if sector_daily:
                            sector_list.append({
                                'name': sector_daily.name,
                                'change_pct': sector_daily.change_pct or 0.0,
                                'stock_count': sector_daily.stock_count or 0,
                                'limit_up_count': sector_daily.limit_up_count or 0,
                            })
                    if sector_list:
                        # 按涨跌幅排序
                        sorted_sectors = sorted(sector_list, key=lambda x: x['change_pct'], reverse=True)
                        top = sorted_sectors[:n]
                        bottom = sorted_sectors[-n:][::-1]  # 反转领跌板块
                        
                        # 更新缓存
                        self._sector_cache = {'top': top, 'bottom': bottom}
                        self._sector_cache_timestamp = time.time()
                        
                        logger.debug(f"[板块] 从数据库获取成功，领涨: {[s['name'] for s in top]}, 领跌: {[s['name'] for s in bottom]}")
                        return top, bottom
            except Exception as e:
                logger.debug(f"[板块] 从数据库获取板块数据失败: {e}，将尝试从数据源获取")

        # 从数据源获取（当强制刷新或数据库没有数据时）
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

    def get_sector_history(self, name: str, end_date: date, days: int = 15) -> List[Dict]:
        """
        获取指定板块近N天的历史数据

        Args:
            name: 板块名称
            end_date: 结束日期
            days: 获取天数，默认15天

        Returns:
            板块历史数据列表，每个元素为字典格式
        """
        if self._db_manager:
            try:
                history = self._db_manager.get_sector_history(name, end_date, days)
                return [data.to_dict() for data in history] if history else []
            except Exception as e:
                logger.error(f"[板块] 获取板块 {name} 历史数据失败: {e}")
        return []

    def get_sector_avg_change_pct(self, name: str, end_date: date, days: int = 15) -> Optional[float]:
        """
        计算指定板块近N天的平均涨跌幅

        Args:
            name: 板块名称
            end_date: 结束日期
            days: 统计天数，默认15天

        Returns:
            平均涨跌幅（%），如果没有数据则返回 None
        """
        if self._db_manager:
            try:
                return self._db_manager.get_sector_avg_change_pct(name, end_date, days)
            except Exception as e:
                logger.error(f"[板块] 计算板块 {name} 平均涨跌幅失败: {e}")
        return None

    def calculate_sector_momentum_score(self, name: str, current_date: date) -> float:
        """
        计算板块动量分数（0-100分）

        动量因子（60%）：
        - 当前涨跌幅（30%）
        - 近15天平均涨跌幅（30%）

        Args:
            name: 板块名称
            current_date: 当前日期

        Returns:
            动量分数 0-100
        """
        momentum_score = 0.0
        
        current_change = self.get_sector_change_pct(name)
        avg_change = self.get_sector_avg_change_pct(name, current_date, 15)
        
        if current_change is not None and avg_change is not None:
            current_score = min(max(current_change * 10, 0), 100)
            avg_score = min(max(avg_change * 10, 0), 100)
            momentum_score = current_score * 0.5 + avg_score * 0.5
        elif current_change is not None:
            current_score = min(max(current_change * 10, 0), 100)
            momentum_score = current_score * 1.0
        elif avg_change is not None:
            avg_score = min(max(avg_change * 10, 0), 100)
            momentum_score = avg_score * 1.0
        
        return min(max(momentum_score, 0.0), 100.0)

    def calculate_sector_sentiment_score(self, name: str) -> float:
        """
        计算板块景气度分数（0-100分）

        景气度因子（40%）：
        - 是否为热点板块（20%）
        - 板块内涨停股票占比（20%）

        Args:
            name: 板块名称

        Returns:
            景气度分数 0-100
        """
        sentiment_score = 0.0
        
        is_hot = self.is_hot_sector(name, threshold_pct=2.0)
        hot_score = 100.0 if is_hot else 0.0
        
        has_limit_up_data = False
        ratio_score = 0.0
        found_sector = False
        top, _ = self.get_sector_rankings(n=50)
        for sector in top:
            if sector.get('name') == name:
                found_sector = True
                stock_count = sector.get('stock_count', 0)
                limit_up_count = sector.get('limit_up_count', 0)
                if stock_count > 0:
                    limit_up_ratio = limit_up_count / stock_count
                    ratio_score = min(limit_up_ratio * 100, 100.0)
                    has_limit_up_data = True
                break
        
        if found_sector and has_limit_up_data:
            sentiment_score = hot_score * 0.5 + ratio_score * 0.5
        elif has_limit_up_data:
            sentiment_score = ratio_score * 1.0
        else:
            sentiment_score = hot_score * 1.0
        
        return min(max(sentiment_score, 0.0), 100.0)

    def calculate_sector_score(self, name: str, current_date: date) -> float:
        """
        计算板块综合分数（0-100分）

        评分规则：
        - tier_1: 20分
        - tier_2: 10分
        - tier_3: 0分

        Args:
            name: 板块名称
            current_date: 当前日期

        Returns:
            板块综合分数 0-100
        """
        tier = self.get_sector_tier(name)
        
        if tier == "tier_1":
            score = 20.0
        elif tier == "tier_2":
            score = 10.0
        else:
            score = 0.0
        
        return min(max(score, 0.0), 100.0)

    def get_sector_tier(self, sector_name: str) -> str:
        """
        获取板块梯队
        
        Args:
            sector_name: 板块名称
            
        Returns:
            str: 梯队级别 ("tier_1", "tier_2", "tier_3")
        """
        heat_dict, sorted_sectors = self.get_all_sectors_heat()
        
        if not sorted_sectors:
            return "tier_3"
        
        try:
            index = sorted_sectors.index(sector_name)
            total = len(sorted_sectors)
            if index < total * 0.05:
                return "tier_1"
            elif index < total * 0.20:
                return "tier_2"
            else:
                return "tier_3"
        except ValueError:
            return "tier_3"
    
    def clear_heat_cache(self) -> None:
        """清除热度缓存"""
        self._heat_cache = None
        self._heat_cache_timestamp = 0
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._sector_cache = {}
        self._sector_cache_timestamp = 0
        self._stock_sector_map = {}
        self._stock_sector_map_timestamp = 0
        self.clear_heat_cache()
        logger.info("[板块] 缓存已清除")

    def get_all_sectors_heat(self, end_date: Optional[date] = None) -> Tuple[Dict[str, float], List[str]]:
        """
        获取所有板块的热度排名

        Args:
            end_date: 结束日期，默认为今天

        Returns:
            Tuple: (板块热度字典, 按热度排序的板块名称列表)
        """
        if end_date is None:
            end_date = date.today()
        
        if time.time() - self._heat_cache_timestamp < self._heat_cache_ttl and self._heat_cache is not None:
            logger.debug("[板块] 使用缓存的热度数据")
            return self._heat_cache
        
        if not self._db_manager:
            logger.error("[板块] 数据库管理器未初始化")
            return {}, []
        
        try:
            logger.debug("[板块] 获取所有板块热度...")
            
            all_sectors = self._db_manager.get_all_sectors()
            if not all_sectors:
                logger.warning("[板块] 没有找到任何板块数据")
                return {}, []
            
            heat_dict = {}
            for sector_name in all_sectors:
                avg_change = self.get_sector_avg_change_pct(sector_name, end_date, days=15)
                if avg_change is not None:
                    heat_dict[sector_name] = avg_change
            
            sorted_sectors = sorted(heat_dict.items(), key=lambda x: x[1], reverse=True)
            heat_dict_sorted = {k: v for k, v in sorted_sectors}
            sector_names_sorted = [k for k, v in sorted_sectors]
            
            self._heat_cache = (heat_dict_sorted, sector_names_sorted)
            self._heat_cache_timestamp = time.time()
            
            logger.debug(f"[板块] 获取成功，共 {len(heat_dict)} 个板块有热度数据")
            return heat_dict_sorted, sector_names_sorted
            
        except Exception as e:
            logger.error(f"[板块] 获取所有板块热度失败: {e}")
            return {}, []
