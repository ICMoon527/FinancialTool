# -*- coding: utf-8 -*-
"""
强势起爆指标股票筛选策略 - Python implementation.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


class MarketDataCache:
    """大盘数据缓存管理器"""

    _CACHE_DIR = None
    _CACHE_VALID_HOURS = 24

    @classmethod
    def _get_cache_dir(cls) -> Path:
        """获取缓存目录路径"""
        if cls._CACHE_DIR is None:
            project_root = Path(__file__).parent.parent.parent.parent
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
        if not cache_file.exists():
            return False
        try:
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            return datetime.now() - file_mtime < timedelta(hours=cls._CACHE_VALID_HOURS)
        except Exception:
            return False

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
            logger.warning(f"[大盘数据缓存] 加载缓存失败: {e}")
            return None

    @classmethod
    def save(cls, symbol: str, data: pd.DataFrame) -> None:
        """保存大盘数据到缓存"""
        cache_file = cls._get_cache_file_path(symbol)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[大盘数据缓存] 保存 {symbol} 数据到缓存成功")
        except Exception as e:
            logger.warning(f"[大盘数据缓存] 保存缓存失败: {e}")


@register_strategy
class StrongDetonationSelectorStrategy(StockSelectorStrategy):
    """
    强势起爆指标股票筛选策略 - Python implementation.

    该策略基于强势起爆指标筛选股票，筛选出最新强势起爆指标中出现紫色箱体的个股。

    紫色箱体判定：
    - AAA（价格基准）同时高于牛线和大盘中线
    """

    _market_data_cache = None

    def __init__(self):
        metadata = StrategyMetadata(
            id="strong_detonation_selector",
            name="strong_detonation_selector",
            display_name="强势起爆选股(Python)",
            description="Python实现的强势起爆指标选股策略，筛选最新出现紫色箱体的个股。",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """Lowest Low Value"""
        return data.rolling(window=period).min()

    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """获取大盘数据（带缓存）"""
        symbol = "sh000001"

        if StrongDetonationSelectorStrategy._market_data_cache is not None:
            return StrongDetonationSelectorStrategy._market_data_cache

        cached_data = MarketDataCache.load(symbol)
        if cached_data is not None:
            StrongDetonationSelectorStrategy._market_data_cache = cached_data
            return cached_data

        if self._data_provider:
            try:
                market_data = self._data_provider.get_index_daily_data(symbol)
                if market_data is not None and not market_data.empty:
                    MarketDataCache.save(symbol, market_data)
                    StrongDetonationSelectorStrategy._market_data_cache = market_data
                    return market_data
            except Exception as e:
                logger.warning(f"获取大盘数据失败: {e}")

        return None

    def _calculate_strong_detonation_indicators(
        self, df: pd.DataFrame, market_data: Optional[pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """计算强势起爆指标。"""
        if df is None or len(df) < 60:
            return None

        try:
            df = df.copy()

            close = df["close"]
            open_price = df["open"]
            high = df["high"]
            low = df["low"]

            aaa = (3 * close + open_price + high + low) / 6
            var1 = self._ema(aaa, 35)
            var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
            var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
            bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
            bear_line = (self._llv(var3, 5) + self._llv(var3, 15) + self._llv(var3, 30)) / 3

            ema120_stock = self._ema(close, 120)

            if market_data is not None and not market_data.empty:
                market_df = market_data.copy()
                if "date" in market_df.columns:
                    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.date

                market_df_renamed = market_df[["date", "close"]].rename(
                    columns={"close": "close_index"}
                )
                merged_df = df.merge(market_df_renamed, on="date", how="left")

                if "close_index" in merged_df.columns:
                    merged_df["close_index"] = merged_df["close_index"].ffill().bfill()
                    if not merged_df["close_index"].isna().any():
                        index_close = merged_df["close_index"]
                        ema120_index = self._ema(index_close, 120)
                        a1 = index_close / ema120_index
                        market_midline = self._ema(ema120_stock * a1, 2)
                    else:
                        a1 = close / ema120_stock
                        market_midline = self._ema(ema120_stock * a1, 2)
                else:
                    a1 = close / ema120_stock
                    market_midline = self._ema(ema120_stock * a1, 2)
            else:
                a1 = close / ema120_stock
                market_midline = self._ema(ema120_stock * a1, 2)

            purple_box = (aaa > bull_line) & (aaa > market_midline)
            red_box = (aaa > market_midline) & (aaa < bull_line)
            weak = ((aaa <= market_midline) & (aaa <= bull_line)) | (
                (aaa >= bull_line) & (aaa < market_midline) & (bull_line < market_midline)
            )

            latest_purple = bool(purple_box.iloc[-1]) if pd.notna(purple_box.iloc[-1]) else False
            latest_red = bool(red_box.iloc[-1]) if pd.notna(red_box.iloc[-1]) else False
            latest_weak = bool(weak.iloc[-1]) if pd.notna(weak.iloc[-1]) else False

            purple_days = int(purple_box.tail(10).sum())

            latest_aaa = float(aaa.iloc[-1]) if pd.notna(aaa.iloc[-1]) else None
            latest_bull = float(bull_line.iloc[-1]) if pd.notna(bull_line.iloc[-1]) else None
            latest_market = float(market_midline.iloc[-1]) if pd.notna(market_midline.iloc[-1]) else None

            return {
                "purple_box": latest_purple,
                "red_box": latest_red,
                "weak": latest_weak,
                "purple_days_10": purple_days,
                "latest_aaa": latest_aaa,
                "latest_bull_line": latest_bull,
                "latest_market_midline": latest_market,
            }
        except Exception as e:
            logger.debug(f"强势起爆指标计算失败: {e}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        执行强势起爆选股策略。

        Args:
            stock_code: 股票代码
            stock_name: 可选的股票名称

        Returns:
            StrategyMatch 结果
        """
        match_details = {}
        conditions_met = []
        conditions_failed = []
        total_score = 0.0
        max_score = 100.0

        try:
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                daily_data_result = self._data_provider.get_daily_data(stock_code, days=150)

                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, data_source = daily_data_result
                else:
                    daily_data = daily_data_result
                    data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["strong_detonation_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    market_data = self._get_market_data()
                    indicators = self._calculate_strong_detonation_indicators(daily_data, market_data)

                    if indicators:
                        match_details["strong_detonation_indicators"] = indicators

                        if indicators.get("purple_box", False):
                            total_score = 100.0
                            conditions_met.append("出现紫色箱体")
                            match_details["conditions"]["purple_box"] = {"passed": True}

                            purple_days = indicators.get("purple_days_10", 0)
                            if purple_days >= 3:
                                conditions_met.append(f"近10天紫色箱体持续{purple_days}天")
                        else:
                            conditions_failed.append("未出现紫色箱体")
                            match_details["conditions"]["purple_box"] = {"passed": False}

                            if indicators.get("red_box", False):
                                conditions_failed.append("处于红色箱体")
                            elif indicators.get("weak", False):
                                conditions_failed.append("处于弱势阶段")

        except Exception as e:
            logger.warning(f"执行策略时出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 50

        if conditions_met:
            reason = f"评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"评分 {raw_score:.0f}/{max_score:.0f}：未满足紫色箱体条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
