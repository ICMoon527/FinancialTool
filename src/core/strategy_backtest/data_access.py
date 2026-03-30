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
        获取实时行情数据。

        在回测中，返回当前时间点对应的行情数据。

        Args:
            stock_code: 股票代码
            **kwargs: 其他参数

        Returns:
            行情数据
        """
        return self._data_provider.get_realtime_quote(stock_code, **kwargs)

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
