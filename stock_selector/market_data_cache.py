
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
    _CACHE_VALID_HOURS = 24  # 默认缓存有效期为24小时
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
        
        if not cls._is_cache_valid(cache_file):
            return None
            
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"[大盘数据缓存] 从缓存加载 {symbol} 数据成功")
            return data
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
