# -*- coding: utf-8 -*-
"""
Top Rise Stock Selector Strategy - Python implementation.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class TopRiseStockSelectorStrategy(StockSelectorStrategy):
    """
    Top Rise Stock Selector Strategy - Python implementation.
    
    This strategy screens stocks based on multiple technical dimensions:
    1. Price ranking - Short-term price performance
    2. Volume analysis - Volume amplification confirmation
    3. Capital flow - Main capital net inflow
    4. Technical breakout - Breaking key resistance levels
    5. Dragon leader signals - Integrating dragon leader start, main rise, second wave signals
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="top_rise_stock_selector",
            name="top_rise_stock_selector",
            display_name="领涨选股(Python)",
            description="Python实现的领涨选股策略，整合多个技术分析维度，筛选潜在的领涨股票，包括涨幅排行、量能异动、资金流向、技术形态和龙头信号。",
            strategy_type=StrategyType.PYTHON,
            category="momentum",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).min()

    def _count(self, condition: pd.Series, period: int) -> pd.Series:
        return condition.astype(int).rolling(window=period).sum()

    def _calculate_top_rise_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate all top rise stock selector indicators."""
        if df is None or len(df) < 60:
            return None

        try:
            df = df.copy()
            result = pd.DataFrame(index=df.index)

            close = df['close']
            volume = df['volume']
            open_price = df['open']
            high = df['high']
            low = df['low']

            change_pct = (close / close.shift(1) - 1) * 100

            result['change_1d'] = change_pct
            result['change_3d'] = (close / close.shift(3) - 1) * 100
            result['change_5d'] = (close / close.shift(5) - 1) * 100
            result['change_10d'] = (close / close.shift(10) - 1) * 100

            price_ranking_score = (
                result['change_1d'] * 0.4 +
                result['change_3d'] * 0.3 +
                result['change_5d'] * 0.2 +
                result['change_10d'] * 0.1
            )

            ma5_volume = self._sma(volume, 5)
            ma10_volume = self._sma(volume, 10)
            ma20_volume = self._sma(volume, 20)

            volume_ratio_5 = volume / ma5_volume
            volume_ratio_10 = volume / ma10_volume

            hhv_volume_20 = self._hhv(volume, 20)
            volume_new_high = volume >= hhv_volume_20

            volume_increasing = (volume > volume.shift(1)) & (volume.shift(1) > volume.shift(2))

            volume_score = np.where(
                volume_ratio_5 >= 2, 30,
                np.where(volume_ratio_5 >= 1.5, 20,
                np.where(volume_ratio_5 >= 1.2, 10, 0))
            )

            volume_score = np.where(volume_new_high, volume_score + 20, volume_score)
            volume_score = np.where(volume_increasing, volume_score + 10, volume_score)

            price_change = close - open_price

            if 'amount' in df.columns:
                amount = df['amount']
            else:
                amount = volume * close

            main_buy = amount.where(price_change > 0, 0)
            main_sell = amount.where(price_change < 0, 0)
            main_net = main_buy - main_sell

            main_net_ma5 = self._sma(main_net, 5)

            capital_inflow = main_net > 0
            capital_inflow_3d = (self._count(capital_inflow, 3) == 3)

            capital_score = np.where(
                capital_inflow_3d, 30,
                np.where(capital_inflow, 15, 0)
            )

            ema20 = self._ema(close, 20)
            ema60 = self._ema(close, 60)

            hhv_high_20 = self._hhv(high, 20)
            hhv_high_60 = self._hhv(high, 60)

            breakout_20 = close >= hhv_high_20.shift(1)
            breakout_60 = close >= hhv_high_60.shift(1)

            above_ema20 = close > ema20
            above_ema60 = close > ema60

            ema_golden_cross = (ema20 > ema60) & (ema20.shift(1) <= ema60.shift(1))

            technical_score = np.where(breakout_60, 40,
                                        np.where(breakout_20, 25, 0))
            technical_score = np.where(above_ema60, technical_score + 15,
                                        np.where(above_ema20, technical_score + 10, technical_score))
            technical_score = np.where(ema_golden_cross, technical_score + 20, technical_score)

            limit_up = change_pct >= 9.0
            consecutive_limit_up = (self._count(limit_up, 2) == 2)
            has_body = close > low

            dragon_leader_signal = limit_up & has_body

            dragon_score = np.where(consecutive_limit_up, 40,
                                      np.where(limit_up, 25, 0))

            top_rise_score = (
                price_ranking_score * 0.2 +
                volume_score * 0.25 +
                capital_score * 0.2 +
                technical_score * 0.2 +
                dragon_score * 0.15
            )

            latest = result.iloc[-1]

            return {
                'change_1d': float(result['change_1d'].iloc[-1]) if pd.notna(result['change_1d'].iloc[-1]) else None,
                'change_3d': float(result['change_3d'].iloc[-1]) if pd.notna(result['change_3d'].iloc[-1]) else None,
                'change_5d': float(result['change_5d'].iloc[-1]) if pd.notna(result['change_5d'].iloc[-1]) else None,
                'change_10d': float(result['change_10d'].iloc[-1]) if pd.notna(result['change_10d'].iloc[-1]) else None,
                'price_ranking_score': float(price_ranking_score.iloc[-1]) if pd.notna(price_ranking_score.iloc[-1]) else None,
                'volume_ratio_5': float(volume_ratio_5.iloc[-1]) if pd.notna(volume_ratio_5.iloc[-1]) else None,
                'volume_new_high': bool(volume_new_high.iloc[-1]) if pd.notna(volume_new_high.iloc[-1]) else False,
                'volume_score': float(volume_score.iloc[-1]) if pd.notna(volume_score.iloc[-1]) else None,
                'capital_inflow': bool(capital_inflow.iloc[-1]) if pd.notna(capital_inflow.iloc[-1]) else False,
                'capital_inflow_3d': bool(capital_inflow_3d.iloc[-1]) if pd.notna(capital_inflow_3d.iloc[-1]) else False,
                'capital_score': float(capital_score.iloc[-1]) if pd.notna(capital_score.iloc[-1]) else None,
                'breakout_20': bool(breakout_20.iloc[-1]) if pd.notna(breakout_20.iloc[-1]) else False,
                'breakout_60': bool(breakout_60.iloc[-1]) if pd.notna(breakout_60.iloc[-1]) else False,
                'above_ema20': bool(above_ema20.iloc[-1]) if pd.notna(above_ema20.iloc[-1]) else False,
                'above_ema60': bool(above_ema60.iloc[-1]) if pd.notna(above_ema60.iloc[-1]) else False,
                'ema_golden_cross': bool(ema_golden_cross.iloc[-1]) if pd.notna(ema_golden_cross.iloc[-1]) else False,
                'technical_score': float(technical_score.iloc[-1]) if pd.notna(technical_score.iloc[-1]) else None,
                'limit_up': bool(limit_up.iloc[-1]) if pd.notna(limit_up.iloc[-1]) else False,
                'consecutive_limit_up': bool(consecutive_limit_up.iloc[-1]) if pd.notna(consecutive_limit_up.iloc[-1]) else False,
                'dragon_score': float(dragon_score.iloc[-1]) if pd.notna(dragon_score.iloc[-1]) else None,
                'top_rise_score': float(top_rise_score.iloc[-1]) if pd.notna(top_rise_score.iloc[-1]) else None,
            }
        except Exception as e:
            logger.debug(f"Top rise indicator calculation failed: {e}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute the top rise stock selector strategy for a single stock.

        Args:
            stock_code: Stock code to analyze
            stock_name: Optional stock name

        Returns:
            StrategyMatch result
        """
        match_details = {}
        conditions_met = []
        conditions_failed = []
        total_score = 0.0
        max_score = 100.0

        try:
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                daily_data_result = self._data_provider.get_daily_data(stock_code, days=60)
                
                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, data_source = daily_data_result
                else:
                    daily_data = daily_data_result
                    data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["top_rise_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    volume_ratio = getattr(realtime_quote, "volume_ratio", None)

                    match_details["realtime_quote"] = {
                        "price": price,
                        "volume_ratio": volume_ratio,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    indicators = self._calculate_top_rise_indicators(daily_data)

                    if indicators:
                        match_details["top_rise_indicators"] = indicators

                        top_rise_score = indicators.get('top_rise_score', 0.0)
                        if top_rise_score is not None:
                            total_score = top_rise_score

                        if indicators.get('change_1d', 0) > 3:
                            conditions_met.append("1日涨幅>3%")
                            match_details["conditions"]["change_1d"] = {"passed": True}

                        if indicators.get('volume_ratio_5', 0) > 1.5:
                            conditions_met.append("量比>1.5")
                            match_details["conditions"]["volume_ratio"] = {"passed": True}

                        if indicators.get('capital_inflow', False):
                            conditions_met.append("资金净流入")
                            match_details["conditions"]["capital_inflow"] = {"passed": True}

                        if indicators.get('breakout_20', False) or indicators.get('breakout_60', False):
                            conditions_met.append("技术突破")
                            match_details["conditions"]["technical_breakout"] = {"passed": True}

                        if indicators.get('limit_up', False):
                            conditions_met.append("涨停")
                            match_details["conditions"]["limit_up"] = {"passed": True}

        except Exception as e:
            logger.warning(f"Error executing strategy for {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 50

        if conditions_met:
            reason = f"领涨评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"领涨评分 {raw_score:.0f}/{max_score:.0f}：未满足核心条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
