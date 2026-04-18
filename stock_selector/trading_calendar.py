
# -*- coding: utf-8 -*-
"""
交易日历管理模块（简化版）

从 AKShare 获取 A 股交易日历并缓存到本地。
"""

import logging
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingCalendar:
    """
    A 股交易日历管理类。
    """

    def __init__(self, cache_path=None):
        if cache_path is None:
            cache_path = Path(__file__).parent.parent / "data" / "trading_calendar.pkl"
        
        self.cache_path = cache_path
        self._trading_days = None
        self._trading_days_sorted = None

    def _fetch_from_akshare(self):
        logger.info("从 AKShare 获取交易日历...")
        
        try:
            import akshare as ak
            
            df = ak.tool_trade_date_hist_sina()
            
            trading_days = set()
            for _, row in df.iterrows():
                date_val = row['trade_date']
                try:
                    if isinstance(date_val, date):
                        d = date_val
                    else:
                        d = datetime.strptime(str(date_val), '%Y-%m-%d').date()
                    trading_days.add(d)
                except (ValueError, TypeError):
                    continue
            
            logger.info("从 AKShare 成功获取 %d 个交易日", len(trading_days))
            return trading_days
            
        except ImportError:
            logger.error("akshare 库未安装，请运行: pip install akshare")
            raise
        except Exception as e:
            logger.error("从 AKShare 获取交易日历失败: %s", e)
            raise

    def _save_cache(self, trading_days):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(trading_days, f)
            
            logger.info("交易日历已保存到: %s", self.cache_path)
        except Exception as e:
            logger.warning("保存交易日历缓存失败: %s", e)

    def _load_cache(self):
        if not self.cache_path.exists():
            return None
        
        try:
            with open(self.cache_path, 'rb') as f:
                trading_days = pickle.load(f)
            
            logger.info("从缓存加载交易日历: %d 个交易日", len(trading_days))
            return trading_days
        except Exception as e:
            logger.warning("加载交易日历缓存失败: %s", e)
            return None

    def _get_trading_days(self, force_refresh=False):
        if self._trading_days is not None and not force_refresh:
            return self._trading_days
        
        if not force_refresh:
            cached = self._load_cache()
            if cached is not None:
                self._trading_days = cached
                return cached
        
        trading_days = self._fetch_from_akshare()
        self._trading_days = trading_days
        self._save_cache(trading_days)
        
        return trading_days

    def refresh(self):
        logger.info("强制刷新交易日历...")
        self._get_trading_days(force_refresh=True)

    def is_trading_day(self, target_date):
        trading_days = self._get_trading_days()
        return target_date in trading_days

    def get_trading_days(self, start_date, end_date):
        trading_days = self._get_trading_days()
        
        result = []
        current_date = start_date
        
        while current_date.__le__(end_date):
            if current_date in trading_days:
                result.append(current_date)
            current_date = current_date.__add__(timedelta(days=1))
        
        return result

    def get_all_trading_days(self):
        if self._trading_days_sorted is None:
            trading_days = self._get_trading_days()
            self._trading_days_sorted = sorted(trading_days)
        
        return self._trading_days_sorted
    
    def get_previous_trading_day(self, target_date=None):
        """获取指定日期之前（包括当天）的最近一个交易日"""
        if target_date is None:
            from datetime import date
            target_date = date.today()
        
        all_trading_days = self.get_all_trading_days()
        
        # 从后往前找第一个 <= target_date 的交易日
        for d in reversed(all_trading_days):
            if d <= target_date:
                return d
        
        # 如果没找到，返回最后一个交易日
        if all_trading_days:
            return all_trading_days[-1]
        return target_date


# 全局单例
_trading_calendar_instance = None


def get_trading_calendar():
    global _trading_calendar_instance
    
    if _trading_calendar_instance is None:
        _trading_calendar_instance = TradingCalendar()
    
    return _trading_calendar_instance


def is_trading_day(target_date):
    return get_trading_calendar().is_trading_day(target_date)


def get_trading_days(start_date, end_date):
    return get_trading_calendar().get_trading_days(start_date, end_date)


def get_previous_trading_day(target_date=None):
    """获取指定日期之前（包括当天）的最近一个交易日"""
    return get_trading_calendar().get_previous_trading_day(target_date)

