
# -*- coding: utf-8 -*-
"""
共享的大盘数据缓存管理器
"""

import logging
from typing import Optional
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import os

logger = logging.getLogger(__name__)


class MarketDataCache:
    """大盘数据缓存管理器"""

    _CACHE_DIR = None
    _CACHE_VALID_HOURS = 24 * 7  # 默认缓存有效期为7天（168小时）
    _force_update = False  # 强制更新标志

    @classmethod
    def set_force_update(cls, force_update: bool):
        """设置是否强制更新缓存"""
        cls._force_update = force_update
        logger.info(f"市场数据缓存强制更新模式: {force_update}")

    @classmethod
    def _get_cache_dir(cls) -> Path:
        """获取缓存目录路径"""
        if cls._CACHE_DIR is None:
            project_root = Path(__file__).parent.parent
            cls._CACHE_DIR = project_root / "data" / "cache"
            cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls._CACHE_DIR

    @classmethod
    def _get_cache_file_path(cls, symbol: str) -> Path:
        """获取缓存文件路径"""
        cache_dir = cls._get_cache_dir()
        return cache_dir / f"market_{symbol}.pkl"

    @classmethod
    def _is_cache_valid(cls, cache_file: Path) -> bool:
        """检查缓存是否有效"""
        if cls._force_update:
            # 强制更新模式下，缓存始终无效
            return False
            
        if not cache_file.exists():
            return False
        try:
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            return datetime.now() - file_mtime < timedelta(hours=cls._CACHE_VALID_HOURS)
        except Exception:
            return False

    @classmethod
    def _cleanup_corrupted_cache(cls, cache_file: Path):
        """清理损坏的缓存文件"""
        try:
            if cache_file.exists():
                cache_file.unlink()  # 删除文件
                logger.info(f"已清理损坏的缓存文件: {cache_file}")
        except Exception as e:
            logger.warning(f"清理损坏的缓存文件失败 {cache_file}: {e}")

    @classmethod
    def load(cls, symbol: str) -> Optional[pd.DataFrame]:
        """从缓存加载大盘数据"""
        cache_file = cls._get_cache_file_path(symbol)
        
        # 如果强制更新模式，直接返回 None
        if cls._force_update:
            return None
            
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            # 简单检查数据格式
            if data is not None and not data.empty and 'date' in data.columns:
                latest_date_in_cache = pd.to_datetime(data['date'].iloc[-1]).date()
                logger.debug(f"[大盘数据缓存] 从缓存加载 {symbol} 数据成功，最新日期: {latest_date_in_cache}")
                return data
            else:
                logger.warning(f"[大盘数据缓存] {symbol} 缓存数据格式不正确")
                return None
        except Exception as e:
            logger.warning(f"[大盘数据缓存] 加载缓存失败: {e}，将尝试清理损坏的文件")
            cls._cleanup_corrupted_cache(cache_file)
            return None

    @classmethod
    def save(cls, symbol: str, data: pd.DataFrame) -> None:
        """保存大盘数据到缓存"""
        cache_file = cls._get_cache_file_path(symbol)
        try:
            # 先保存到临时文件，避免写入过程中中断导致文件损坏
            temp_file = cache_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(data, f)
            
            # 原子性替换：重命名临时文件为目标文件
            if cache_file.exists():
                cache_file.unlink()
            temp_file.rename(cache_file)
            
            logger.info(f"[大盘数据缓存] 保存 {symbol} 数据到缓存成功，共 {len(data)} 条")
        except Exception as e:
            logger.warning(f"[大盘数据缓存] 保存缓存失败: {e}")
            # 清理临时文件
            temp_file = cache_file.with_suffix(".tmp")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
    
    @classmethod
    def _load_without_validation(cls, symbol: str) -> Optional[pd.DataFrame]:
        """
        从缓存文件加载数据，不进行日期严格验证（用于智能时间窗方法）
        """
        cache_file = cls._get_cache_file_path(symbol)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if data is not None and not data.empty and 'date' in data.columns:
                return data
            return None
        except Exception as e:
            logger.warning(f"[智能大盘数据] 加载缓存失败（宽松模式）: {e}")
            cls._cleanup_corrupted_cache(cache_file)
            return None
    
    @classmethod
    def get_complete_index_data(cls, symbol: str, data_provider=None) -> Optional[pd.DataFrame]:
        """
        获取完整可用的大盘指数数据（智能时间窗策略）
        
        策略：
        if 当前的时间是交易日盘中或收盘2小时内:
            直接联网更新大盘数据并保存
        else:
            if 当前缓存文件的更新时间是最近交易日收盘2小时后:
                不用更新，使用缓存文件的历史数据
            else:
                联网更新大盘数据并保存
        
        Args:
            symbol: 指数代码（sh000001 / sz399001）
            data_provider: 数据提供者实例（可选，用于获取实时行情）
        
        Returns:
            完整的大盘数据DataFrame
        """
        from datetime import date, datetime, time, timedelta
        from stock_selector.trading_calendar import get_previous_trading_day, is_trading_day
        
        # 获取当前时间和日期
        now = datetime.now()
        today = date.today()
        current_time = now.time()
        
        # 开盘时间和收盘时间
        market_open = time(9, 15, 0)
        market_close = time(15, 0, 0)
        market_close_plus_2h = time(17, 0, 0)
        
        # 判断：当前的时间是交易日盘中或收盘2小时内
        is_trading_day_today = is_trading_day(today)
        is_in_trading_window = False
        if is_trading_day_today:
            # 交易日
            if current_time >= market_open and current_time <= market_close_plus_2h:
                # 盘中或收盘2小时内
                is_in_trading_window = True
        
        if is_in_trading_window:
            # 交易日盘中或收盘2小时内：直接联网更新大盘数据
            logger.info(f"[智能大盘数据] 当前在交易时间窗口（盘中或收盘2小时内），联网更新数据")
            if data_provider is not None:
                try:
                    complete_data = data_provider.get_index_daily_data(symbol)
                    if complete_data is not None and not complete_data.empty:
                        latest_date = pd.to_datetime(complete_data['date'].iloc[-1]).date()
                        logger.info(f"[智能大盘数据] 成功获取最新数据，最新日期：{latest_date}")
                        # 自动保存缓存
                        cls.save(symbol, complete_data)
                        return complete_data
                except Exception as e:
                    logger.warning(f"[智能大盘数据] 联网获取数据失败：{e}")
            # 回退到缓存
            logger.info(f"[智能大盘数据] 联网更新失败，回退到缓存")
            cached_data = cls._load_without_validation(symbol)
            if cached_data is not None and not cached_data.empty:
                return cached_data
            return None
        
        # 不在交易时间窗口内：检查缓存文件
        logger.info(f"[智能大盘数据] 当前不在交易时间窗口内，检查缓存")
        
        # 先加载缓存数据
        cached_data = cls._load_without_validation(symbol)
        if cached_data is None or cached_data.empty:
            logger.warning(f"[智能大盘数据] 缓存不存在或为空，联网更新")
            if data_provider is not None:
                try:
                    complete_data = data_provider.get_index_daily_data(symbol)
                    if complete_data is not None and not complete_data.empty:
                        # 自动保存缓存
                        cls.save(symbol, complete_data)
                        return complete_data
                except Exception as e:
                    logger.warning(f"[智能大盘数据] 联网获取数据失败：{e}")
            return None
        
        # 获取最近的交易日
        latest_trading_day = get_previous_trading_day(today)
        logger.info(f"[智能大盘数据] 最近的交易日：{latest_trading_day}")
        
        # 检查缓存文件的更新时间
        cache_file = cls._get_cache_file_path(symbol)
        if cache_file.exists():
            try:
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                logger.info(f"[智能大盘数据] 缓存文件更新时间：{file_mtime}")
                
                # 构建最近交易日收盘2小时后的时间点
                latest_trading_day_close_plus_2h = datetime.combine(latest_trading_day, market_close_plus_2h)
                
                # 判断缓存是否已经更新过了
                if file_mtime >= latest_trading_day_close_plus_2h:
                    logger.info(f"[智能大盘数据] 缓存已在最近交易日收盘2小时后更新，使用缓存")
                    return cached_data
            except Exception as e:
                logger.warning(f"[智能大盘数据] 检查缓存文件时间失败：{e}")
        
        # 缓存需要更新
        logger.info(f"[智能大盘数据] 缓存需要更新，联网获取最新数据")
        if data_provider is not None:
            try:
                complete_data = data_provider.get_index_daily_data(symbol)
                if complete_data is not None and not complete_data.empty:
                    latest_date = pd.to_datetime(complete_data['date'].iloc[-1]).date()
                    logger.info(f"[智能大盘数据] 成功获取最新数据，最新日期：{latest_date}")
                    # 自动保存缓存
                    cls.save(symbol, complete_data)
                    return complete_data
            except Exception as e:
                logger.warning(f"[智能大盘数据] 联网获取数据失败：{e}")
        
        # 联网失败，回退到缓存
        logger.info(f"[智能大盘数据] 联网更新失败，使用现有缓存")
        return cached_data
