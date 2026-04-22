
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
            
            # 检查数据是否有效：看数据中最新的日期是不是最近的交易日
            if data is not None and not data.empty and 'date' in data.columns:
                from datetime import date
                from stock_selector.trading_calendar import get_previous_trading_day
                
                # 获取数据中最新的日期
                latest_date_in_cache = pd.to_datetime(data['date'].iloc[-1]).date()
                
                # 获取离今天最近的交易日
                today = date.today()
                latest_trading_day = get_previous_trading_day(today)
                
                # 检查缓存中的最新日期是否 >= 最近的交易日
                if latest_date_in_cache >= latest_trading_day:
                    logger.debug(f"[大盘数据缓存] 从缓存加载 {symbol} 数据成功，最新日期: {latest_date_in_cache}")
                    return data
                else:
                    logger.warning(f"[大盘数据缓存] {symbol} 缓存数据日期({latest_date_in_cache})早于最近交易日({latest_trading_day})，请更新数据")
                    return None
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
        - 交易中（9:15-15:00）：用历史数据 + 实时大盘收盘价补充
        - 收盘后：用完整历史数据
        
        Args:
            symbol: 指数代码（sh000001 / sz399001）
            data_provider: 数据提供者实例（可选，用于获取实时行情）
        
        Returns:
            完整的大盘数据DataFrame
        """
        from datetime import date, datetime, time, timedelta
        from stock_selector.trading_calendar import get_previous_trading_day
        
        # 获取当前时间和日期
        now = datetime.now()
        today = date.today()
        current_time = now.time()
        
        # 开盘时间和收盘时间
        market_open = time(9, 15, 0)
        market_close = time(15, 0, 0)
        
        # 先从缓存加载历史数据（使用宽松模式）
        historical_data = cls._load_without_validation(symbol)
        
        if historical_data is None or historical_data.empty:
            logger.warning(f"[智能大盘数据] 没有历史数据可用")
            return None
        
        latest_date_in_data = pd.to_datetime(historical_data['date'].iloc[-1]).date()
        logger.info(f"[智能大盘数据] 历史数据最新日期：{latest_date_in_data}")
        
        # 判断是否需要补充实时数据
        need_realtime_supplement = False
        market_close_plus_2h = time(17, 0, 0)
        
        if current_time >= market_open and current_time < market_close:
            # 交易时间内
            target_trading_day = get_previous_trading_day(today - timedelta(days=1))
            logger.info(f"[智能大盘数据] 当前时间 {current_time} 在交易时间内")
            
            if latest_date_in_data < today:
                # 历史数据没有今天，需要补充
                need_realtime_supplement = True
                logger.info(f"[智能大盘数据] 需要补充今天（{today}）的实时数据")
        elif market_close <= current_time <= market_close_plus_2h:
            # 收盘后2小时内
            target_trading_day = get_previous_trading_day(today)
            logger.info(f"[智能大盘数据] 当前时间 {current_time} 在收盘后2小时内")
            
            # 检查历史数据是否已更新到今天
            if latest_date_in_data < today:
                # 历史数据还没更新，补充实时数据
                need_realtime_supplement = True
                logger.info(f"[智能大盘数据] 收盘后历史数据未更新，需要补充今天（{today}）的实时数据")
            elif latest_date_in_data < target_trading_day:
                logger.warning(f"[智能大盘数据] 历史数据不完整，最新日期 {latest_date_in_data} < {target_trading_day}")
        elif current_time > market_close_plus_2h:
            # 收盘2小时后
            target_trading_day = get_previous_trading_day(today)
            logger.info(f"[智能大盘数据] 当前时间 {current_time} 已过收盘时间2小时以上")
            
            if latest_date_in_data < target_trading_day:
                logger.warning(f"[智能大盘数据] 历史数据不完整，最新日期 {latest_date_in_data} < {target_trading_day}")
                # 尝试获取最新的完整历史数据
                if data_provider is not None:
                    try:
                        logger.info(f"[智能大盘数据] 尝试获取最新的完整历史数据...")
                        complete_data = data_provider.get_index_daily_data(symbol)
                        if complete_data is not None and not complete_data.empty:
                            new_latest_date = pd.to_datetime(complete_data['date'].iloc[-1]).date()
                            if new_latest_date >= target_trading_day:
                                logger.info(f"[智能大盘数据] 成功获取最新历史数据，最新日期：{new_latest_date}")
                                return complete_data
                            else:
                                logger.warning(f"[智能大盘数据] 获取的历史数据仍然不完整，最新日期：{new_latest_date}")
                    except Exception as e:
                        logger.warning(f"[智能大盘数据] 获取最新历史数据失败：{e}")
        else:
            # 开盘前
            target_trading_day = get_previous_trading_day(today - timedelta(days=1))
            logger.info(f"[智能大盘数据] 当前时间 {current_time} 在开盘前")
        
        # 如果不需要补充实时数据，直接返回历史数据
        if not need_realtime_supplement:
            logger.info(f"[智能大盘数据] 直接使用历史数据")
            return historical_data
        
        # 需要补充实时数据
        if data_provider is None:
            logger.warning(f"[智能大盘数据] 需要补充实时数据但没有data_provider")
            return historical_data
        
        # 获取实时大盘行情
        try:
            logger.info(f"[智能大盘数据] 尝试获取实时大盘行情")
            main_indices = data_provider.get_main_indices(region="cn")
            
            if main_indices is None or len(main_indices) == 0:
                logger.warning(f"[智能大盘数据] 获取实时行情失败，使用历史数据")
                return historical_data
            
            # 找到对应的指数
            target_index = None
            for idx in main_indices:
                if idx.get("code") == symbol:
                    target_index = idx
                    break
            
            if target_index is None:
                logger.warning(f"[智能大盘数据] 没找到指数 {symbol} 的实时数据")
                return historical_data
            
            # 获取实时价格
            current_price = target_index.get("current")
            if current_price is None:
                logger.warning(f"[智能大盘数据] 实时数据没有价格")
                return historical_data
            
            logger.info(f"[智能大盘数据] 实时指数点位：{current_price}")
            
            # 准备补充的数据
            last_row = historical_data.iloc[-1].copy()
            new_row = {
                "date": today,
                "open": last_row.get("close", current_price),
                "high": max(last_row.get("close", current_price), current_price),
                "low": min(last_row.get("close", current_price), current_price),
                "close": current_price,
                "volume": last_row.get("volume", 0),
            }
            
            # 如果有成交额字段，也保留
            if "amount" in last_row:
                new_row["amount"] = last_row.get("amount", 0)
            
            # 检查今天的数据是否已经存在
            if latest_date_in_data >= today:
                logger.info(f"[智能大盘数据] 历史数据已经包含今天，不重复添加")
                return historical_data
            
            # 追加新数据
            complete_data = pd.concat([
                historical_data,
                pd.DataFrame([new_row])
            ], ignore_index=True)
            
            logger.info(f"[智能大盘数据] 成功补充实时数据，现在共 {len(complete_data)} 条")
            return complete_data
            
        except Exception as e:
            logger.warning(f"[智能大盘数据] 获取实时数据失败：{e}，使用历史数据")
            return historical_data
