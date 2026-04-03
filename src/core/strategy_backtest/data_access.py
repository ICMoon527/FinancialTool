# -*- coding: utf-8 -*-
"""
时间隔离数据访问模块。

确保策略在任意时间点T仅能访问该时间点之前可获取的历史数据。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional, List, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseFirstDataProvider:
    """
    数据库优先的数据提供器包装器。
    
    优先从数据库读取数据，如果数据库中没有数据，再使用原始数据提供器。
    """
    
    def __init__(self, original_data_provider: Any, db_manager: Any):
        """
        初始化数据库优先数据提供器。
        
        Args:
            original_data_provider: 原始数据提供器
            db_manager: 数据库管理器
        """
        self._original = original_data_provider
        self._db_manager = db_manager
        logger.info("数据库优先数据提供器已初始化")
    
    def get_daily_data(
        self,
        stock_code: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, str]]:
        """
        获取日线数据，优先从数据库读取。
        """
        try:
            # 标准化股票代码
            from data_provider.base import normalize_stock_code
            code = normalize_stock_code(stock_code)
            
            # 解析日期范围
            from datetime import date as date_type, timedelta
            end_date_obj = None
            start_date_obj = None
            
            if end_date:
                if isinstance(end_date, str):
                    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
                elif isinstance(end_date, datetime):
                    end_date_obj = end_date.date()
                elif isinstance(end_date, date_type):
                    end_date_obj = end_date
            else:
                end_date_obj = date.today()
            
            if start_date:
                if isinstance(start_date, str):
                    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
                elif isinstance(start_date, datetime):
                    start_date_obj = start_date.date()
                elif isinstance(start_date, date_type):
                    start_date_obj = start_date
            elif days:
                start_date_obj = end_date_obj - timedelta(days=int(days * 1.5))  # 多留一些缓冲
            
            if start_date_obj is None:
                start_date_obj = end_date_obj - timedelta(days=365)
            
            # 首先尝试从数据库读取指定日期范围的数据
            logger.info(f"数据库优先: 尝试从数据库读取 {code} 的数据: {start_date_obj} ~ {end_date_obj}")
            stock_dailies = self._db_manager.get_data_range_optimized(
                code=code,
                start_date=start_date_obj,
                end_date=end_date_obj
            )
            
            if stock_dailies:
                # 转换为DataFrame
                df = pd.DataFrame([
                    {
                        'date': s.date,
                        'open': s.open,
                        'high': s.high,
                        'low': s.low,
                        'close': s.close,
                        'volume': s.volume,
                        'amount': s.amount,
                        'pct_chg': s.pct_chg,
                        'ma5': s.ma5,
                        'ma10': s.ma10,
                        'ma20': s.ma20,
                        'volume_ratio': s.volume_ratio
                    }
                    for s in stock_dailies
                ])
                
                if not df.empty:
                    logger.info(f"数据库优先: 成功从数据库读取 {code} 的 {len(df)} 条数据")
                    return df, "database"
                else:
                    logger.warning(f"数据库优先: 数据库返回了 {len(stock_dailies)} 条记录，但转换后的DataFrame为空")
            
            # 如果指定日期范围没有数据，尝试读取该股票的所有可用数据
            logger.warning(f"数据库优先: 数据库中没有 {code} 在 {start_date_obj} ~ {end_date_obj} 的数据，尝试读取该股票的所有数据...")
            
            # 读取该股票的所有数据（不限制日期范围）
            from src.storage import StockDaily
            from sqlalchemy import select
            
            with self._db_manager.get_session() as session:
                all_stock_dailies = session.execute(
                    select(StockDaily)
                    .where(StockDaily.code == code)
                    .order_by(StockDaily.date)
                ).scalars().all()
            
            if all_stock_dailies:
                # 转换为DataFrame
                df = pd.DataFrame([
                    {
                        'date': s.date,
                        'open': s.open,
                        'high': s.high,
                        'low': s.low,
                        'close': s.close,
                        'volume': s.volume,
                        'amount': s.amount,
                        'pct_chg': s.pct_chg,
                        'ma5': s.ma5,
                        'ma10': s.ma10,
                        'ma20': s.ma20,
                        'volume_ratio': s.volume_ratio
                    }
                    for s in all_stock_dailies
                ])
                
                if not df.empty:
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    logger.info(f"数据库优先: 成功从数据库读取 {code} 的 {len(df)} 条数据（可用范围: {min_date} ~ {max_date}）")
                    return df, "database"
                else:
                    logger.warning(f"数据库优先: 数据库返回了 {len(all_stock_dailies)} 条记录，但转换后的DataFrame为空")
            else:
                logger.warning(f"数据库优先: 数据库中完全没有 {code} 的数据，使用原始数据提供器")
            
        except Exception as e:
            logger.error(f"数据库优先: 从数据库读取失败 {stock_code}: {e}", exc_info=True)
        
        # 数据库读取失败，使用原始数据提供器
        return self._original.get_daily_data(
            stock_code,
            days=days,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
    
    def get_index_daily_data(self, index_symbol: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """获取指数日线数据，代理到原始数据提供器。"""
        return self._original.get_index_daily_data(index_symbol, **kwargs)
    
    def get_realtime_quote(self, stock_code: str, **kwargs: Any) -> Any:
        """获取实时行情数据，代理到原始数据提供器。"""
        return self._original.get_realtime_quote(stock_code, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理访问原始数据提供器的属性和方法。"""
        return getattr(self._original, name)


class TimeIsolatedDataProvider:
    """
    时间隔离数据提供器。

    此类封装了数据访问逻辑，确保在回测时间点T只能访问T之前的数据。
    """

    def __init__(
        self,
        data_provider: Any,
        current_date: Optional[date] = None,
    ):
        """
        初始化时间隔离数据提供器。

        Args:
            data_provider: 原始数据提供器
            current_date: 当前回测时间点，默认为None（可后续设置）
        """
        self._data_provider = data_provider
        self._current_date: Optional[date] = current_date
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def set_current_date(self, current_date: date) -> None:
        """
        设置当前回测时间点。

        Args:
            current_date: 当前日期
        """
        self._current_date = current_date

    def get_current_date(self) -> Optional[date]:
        """
        获取当前回测时间点。

        Returns:
            当前日期
        """
        return self._current_date

    def _filter_data_by_date(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        过滤数据，只保留当前日期之前的数据。

        Args:
            df: 原始数据
            date_col: 日期列名

        Returns:
            过滤后的数据
        """
        if self._current_date is None:
            return df

        if date_col not in df.columns:
            return df

        df_copy = df.copy()

        if pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
            df_copy[date_col] = df_copy[date_col].dt.date
        else:
            try:
                df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.date
            except Exception:
                return df

        return df_copy[df_copy[date_col] <= self._current_date]

    def get_daily_data(
        self,
        stock_code: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, str]]:
        """
        获取日线数据，自动进行时间过滤。

        Args:
            stock_code: 股票代码
            days: 获取天数
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数

        Returns:
            过滤后的日线数据
        """
        cache_key = f"daily_{stock_code}"
        if cache_key not in self._data_cache:
            result = self._data_provider.get_daily_data(
                stock_code,
                days=days,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )

            if isinstance(result, tuple) and len(result) == 2:
                df, source = result
                self._data_cache[cache_key] = (df, source)
            else:
                self._data_cache[cache_key] = (result, "unknown")

        cached_data = self._data_cache[cache_key]
        df, source = cached_data

        filtered_df = self._filter_data_by_date(df)

        if isinstance(cached_data[0], tuple):
            return filtered_df, source
        return filtered_df

    def get_index_daily_data(self, index_symbol: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """
        获取指数日线数据，自动进行时间过滤。

        Args:
            index_symbol: 指数代码
            **kwargs: 其他参数

        Returns:
            过滤后的指数日线数据
        """
        cache_key = f"index_{index_symbol}"
        if cache_key not in self._data_cache:
            data = self._data_provider.get_index_daily_data(index_symbol, **kwargs)
            self._data_cache[cache_key] = data

        cached_data = self._data_cache[cache_key]
        if cached_data is None:
            return None

        return self._filter_data_by_date(cached_data)

    def get_realtime_quote(self, stock_code: str, **kwargs: Any) -> Any:
        """
        获取实时行情数据（回测时使用历史数据）。

        在回测中，不获取真实的实时行情，而是返回当前时间点的历史收盘价。

        Args:
            stock_code: 股票代码
            **kwargs: 其他参数

        Returns:
            模拟的行情数据对象，包含price属性
        """
        try:
            # 获取当前日期的历史数据
            if self._current_date:
                # 尝试获取该股票在当前日期的收盘价
                try:
                    data = self.get_daily_data(
                        stock_code,
                        days=1,
                        start_date=self._current_date.strftime("%Y-%m-%d"),
                        end_date=self._current_date.strftime("%Y-%m-%d")
                    )
                    
                    df = None
                    if isinstance(data, tuple):
                        df, _ = data
                    else:
                        df = data
                    
                    if df is not None and not df.empty:
                        # 确保日期列格式正确
                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"]).dt.date
                            target_df = df[df["date"] == self._current_date]
                            if not target_df.empty:
                                # 获取收盘价
                                close_price = None
                                if "close" in target_df.columns:
                                    close_price = float(target_df["close"].iloc[-1])
                                elif "Close" in target_df.columns:
                                    close_price = float(target_df["Close"].iloc[-1])
                                
                                if close_price is not None:
                                    # 返回一个模拟的行情对象
                                    class MockQuote:
                                        def __init__(self, price):
                                            self.price = price
                                    
                                    return MockQuote(close_price)
                except Exception as e:
                    logger.debug(f"获取回测行情数据失败 {stock_code}: {e}")
            
            # 如果无法获取历史数据，返回一个简单的mock对象
            class MockQuote:
                def __init__(self):
                    self.price = None
            
            return MockQuote()
            
        except Exception as e:
            logger.debug(f"回测获取行情失败 {stock_code}: {e}")
            # 返回一个简单的mock对象
            class MockQuote:
                def __init__(self):
                    self.price = None
            
            return MockQuote()

    def clear_cache(self) -> None:
        """
        清除数据缓存。
        """
        self._data_cache.clear()

    def __getattr__(self, name: str) -> Any:
        """
        代理访问原始数据提供器的属性和方法。

        Args:
            name: 属性名

        Returns:
            属性值
        """
        return getattr(self._data_provider, name)
