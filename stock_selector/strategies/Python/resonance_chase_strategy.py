# -*- coding: utf-8 -*-
"""
共振追涨策略 - Python implementation.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class ResonanceChaseStrategy(StockSelectorStrategy):
    """
    共振追涨策略 - Python实现.

    该策略通过共振追涨指标识别紫色共振柱，筛选最新出现共振追涨信号的股票。
    紫色共振柱判定条件：中线多头趋势(mid_bullish) 且 短线趋势走强(resonance)
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="resonance_chase",
            name="resonance_chase",
            display_name="共振追涨(Python)",
            description="Python实现的共振追涨策略，识别最新出现紫色共振柱特征的股票。紫色共振柱表示中线和短线趋势共振走强。",
            strategy_type=StrategyType.PYTHON,
            category="momentum",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最高高值"""
        return data.rolling(window=period, min_periods=1).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最低低值"""
        return data.rolling(window=period, min_periods=1).min()

    def _calculate_resonance_chase_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算共振追涨指标，识别紫色共振柱。

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含指标计算结果和紫色共振柱信息的字典
        """
        if df is None or len(df) < 60:
            logger.debug(f"Data length too short: {len(df) if df is not None else 0}")
            return None

        try:
            df = df.copy()

            close = df['close']
            open_price = df['open']
            high = df['high']
            low = df['low']

            AAA = (3 * close + high + low + open_price) / 6

            VAR1 = self._ema(AAA, 35)

            VAR2 = (self._hhv(VAR1, 10) + self._hhv(VAR1, 30) + self._hhv(VAR1, min(60, len(df)))) / 3
            VAR3 = (self._llv(VAR1, 10) + self._llv(VAR1, 30) + self._llv(VAR1, min(60, len(df)))) / 3

            bull_line = (self._hhv(VAR2, 5) + self._hhv(VAR2, 10) + self._hhv(VAR2, 20)) / 3
            bear_line = (self._llv(VAR3, 5) + self._llv(VAR3, 10) + self._llv(VAR3, 20)) / 3

            ema_aaa_2 = self._ema(self._ema(AAA, 2), 2)
            mid_bullish = ema_aaa_2 > bear_line

            exp6 = self._ema(close, 6)
            exp18 = self._ema(close, 18)
            OUT1 = 500 * (exp6 - exp18) / exp18 + 2
            OUT2 = self._ema(OUT1, 3)

            resonance = OUT1 > 2

            purple_resonance = mid_bullish & resonance

            latest_mid_bullish = bool(mid_bullish.iloc[-1]) if pd.notna(mid_bullish.iloc[-1]) else False
            latest_resonance = bool(resonance.iloc[-1]) if pd.notna(resonance.iloc[-1]) else False
            latest_purple_resonance = bool(purple_resonance.iloc[-1]) if pd.notna(purple_resonance.iloc[-1]) else False

            resonance_dates = []
            if 'trade_date' in df.columns:
                for idx in df.index:
                    if bool(purple_resonance.loc[idx]) if pd.notna(purple_resonance.loc[idx]) else False:
                        resonance_dates.append(str(df.loc[idx, 'trade_date']))
            elif isinstance(df.index, pd.DatetimeIndex):
                for idx in df.index:
                    if bool(purple_resonance.loc[idx]) if pd.notna(purple_resonance.loc[idx]) else False:
                        resonance_dates.append(idx.strftime('%Y-%m-%d'))

            latest_resonance_date = resonance_dates[-1] if resonance_dates else None
            resonance_count = len(resonance_dates)

            raw_score = 100.0 if latest_purple_resonance else 0.0
            if latest_mid_bullish and not latest_purple_resonance:
                raw_score = 50.0
            if latest_resonance and not latest_purple_resonance:
                raw_score = max(raw_score, 60.0)

            return {
                'bull_line': float(bull_line.iloc[-1]) if pd.notna(bull_line.iloc[-1]) else None,
                'bear_line': float(bear_line.iloc[-1]) if pd.notna(bear_line.iloc[-1]) else None,
                'mid_bullish': latest_mid_bullish,
                'resonance': latest_resonance,
                'purple_resonance': latest_purple_resonance,
                'OUT1': float(OUT1.iloc[-1]) if pd.notna(OUT1.iloc[-1]) else None,
                'OUT2': float(OUT2.iloc[-1]) if pd.notna(OUT2.iloc[-1]) else None,
                'resonance_dates': resonance_dates,
                'latest_resonance_date': latest_resonance_date,
                'resonance_count': resonance_count,
                'raw_score': raw_score,
            }
        except Exception as e:
            logger.warning(f"Resonance chase indicator calculation failed: {e}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        执行共振追涨策略，分析单只股票。

        Args:
            stock_code: 股票代码
            stock_name: 可选的股票名称

        Returns:
            StrategyMatch结果
        """
        match_details = {}
        conditions_met = []
        conditions_failed = []
        total_score = 0.0
        max_score = 100.0

        try:
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                daily_data_result = self._data_provider.get_daily_data(stock_code, days=120)

                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, data_source = daily_data_result
                else:
                    daily_data = daily_data_result
                    data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["resonance_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    indicators = self._calculate_resonance_chase_indicators(daily_data)

                    if indicators:
                        match_details["resonance_indicators"] = indicators

                        raw_score = indicators.get('raw_score', 0.0)
                        if raw_score is not None:
                            total_score = raw_score

                        if indicators.get('mid_bullish', False):
                            conditions_met.append("中线多头趋势")
                            match_details["conditions"]["mid_bullish"] = {"passed": True}

                        if indicators.get('resonance', False):
                            conditions_met.append("短线趋势走强")
                            match_details["conditions"]["resonance"] = {"passed": True}

                        if indicators.get('purple_resonance', False):
                            conditions_met.append("紫色共振柱")
                            match_details["conditions"]["purple_resonance"] = {"passed": True}
                            resonance_date = indicators.get('latest_resonance_date', '')
                            if resonance_date:
                                conditions_met.append(f"共振时间: {resonance_date}")
                        elif indicators.get('resonance_count', 0) > 0:
                            resonance_date = indicators.get('latest_resonance_date', '')
                            if resonance_date:
                                conditions_met.append(f"历史共振: {resonance_date}")

        except Exception as e:
            logger.warning(f"Error executing resonance chase strategy for {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 40

        if conditions_met:
            reason = f"共振追涨评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"共振追涨评分 {raw_score:.0f}/{max_score:.0f}：未出现紫色共振柱"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
