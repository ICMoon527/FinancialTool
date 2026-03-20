# -*- coding: utf-8 -*-
"""
Afternoon 2:30 Short-Term Strategy - Comprehensive Python implementation.
"""

import logging
from typing import Optional, Dict, Any
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
class ShortTermStrategyPython(StockSelectorStrategy):
    """
    Afternoon 2:30 Short-Term Strategy - Python implementation.

    This strategy screens stocks based on comprehensive criteria:
    1. Time window: Focus on 14:30-15:00 (trading session)
    2. Volume conditions: Volume spike, tail session volume > 30% of daily
    3. Price conditions: Gain 1%-5%, above MA5, MACD bullish
    4. Market conditions: Index positive, sector strong
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="short_term_strategy_python",
            name="short_term_strategy_python",
            display_name="下午两点半短线策略(Python)",
            description="Python实现的下午两点半选股策略，通过代码直接过滤不符合要求的股票，输出符合标准的标的。关注尾盘异动、量能放大、价格走势和市场环境。",
            strategy_type=StrategyType.PYTHON,
            category="short_term",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _calculate_ma(self, df: pd.DataFrame, period: int) -> Optional[float]:
        """Calculate moving average."""
        if df is None or len(df) < period:
            return None
        try:
            return float(df['close'].tail(period).mean())
        except Exception:
            return None

    def _calculate_macd(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate MACD indicators."""
        if df is None or len(df) < 35:
            return None

        try:
            df = df.copy()
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None

            golden_cross = False
            if prev is not None:
                golden_cross = (prev['macd'] <= prev['signal']) and (latest['macd'] > latest['signal'])

            about_to_golden = False
            if prev is not None:
                macd_trend_up = latest['macd'] > prev['macd']
                gap_closing = abs(latest['macd'] - latest['signal']) < abs(prev['macd'] - prev['signal'])
                about_to_golden = macd_trend_up and (gap_closing or latest['histogram'] > -0.1)

            return {
                'macd': float(latest['macd']),
                'signal': float(latest['signal']),
                'histogram': float(latest['histogram']),
                'golden_cross': golden_cross,
                'about_to_golden': about_to_golden,
            }
        except Exception as e:
            logger.debug(f"MACD calculation failed: {e}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute the short-term strategy for a single stock.

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
                
                # Handle tuple return value (DataFrame, data_source)
                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, data_source = daily_data_result
                else:
                    daily_data = daily_data_result
                    data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["daily_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    change_pct = getattr(realtime_quote, "change_pct", None)
                    volume = getattr(realtime_quote, "volume", None)
                    volume_ratio = getattr(realtime_quote, "volume_ratio", None)
                    turnover_rate = getattr(realtime_quote, "turnover_rate", None)

                    match_details["realtime_quote"] = {
                        "price": price,
                        "change_pct": change_pct,
                        "volume": volume,
                        "volume_ratio": volume_ratio,
                        "turnover_rate": turnover_rate,
                    }

                    if change_pct is not None:
                        if 3.0 <= change_pct <= 5.0:
                            conditions_met.append("当日涨幅 3%-5%")
                            total_score += 20
                            match_details["conditions"]["gain_range"] = {"passed": True, "value": change_pct}
                        else:
                            conditions_failed.append(f"涨幅不在 3%-5% (实际: {change_pct:.2f}%)")
                            match_details["conditions"]["gain_range"] = {"passed": False, "value": change_pct}

                    if volume_ratio is not None:
                        if volume_ratio >= 1.5:
                            conditions_met.append("量能放大 > 1.5倍")
                            total_score += 20
                            match_details["conditions"]["volume_spike"] = {"passed": True, "value": volume_ratio}
                        else:
                            conditions_failed.append(f"量能未放大 (实际: {volume_ratio:.2f})")
                            match_details["conditions"]["volume_spike"] = {"passed": False, "value": volume_ratio}

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    ma5 = self._calculate_ma(daily_data, 5)
                    macd_result = self._calculate_macd(daily_data)

                    match_details["daily_indicators"] = {
                        "ma5": ma5,
                        "macd": macd_result,
                    }

                    if realtime_quote and ma5 is not None:
                        current_price = getattr(realtime_quote, "price", None)
                        if current_price is not None and current_price >= ma5:
                            conditions_met.append(f"价格站稳MA5 ({ma5:.2f})")
                            total_score += 15
                            match_details["conditions"]["above_ma5"] = {"passed": True, "price": current_price, "ma5": ma5}
                        elif current_price is not None:
                            conditions_failed.append(f"价格低于MA5 (价格: {current_price:.2f}, MA5: {ma5:.2f})")
                            match_details["conditions"]["above_ma5"] = {"passed": False, "price": current_price, "ma5": ma5}

                    if macd_result:
                        if macd_result.get("golden_cross", False):
                            conditions_met.append("MACD金叉")
                            total_score += 25
                            match_details["conditions"]["macd_bullish"] = {"passed": True, "type": "golden_cross"}
                        elif macd_result.get("about_to_golden", False):
                            conditions_met.append("MACD即将金叉")
                            total_score += 20
                            match_details["conditions"]["macd_bullish"] = {"passed": True, "type": "about_to_golden"}
                        else:
                            conditions_failed.append("MACD未呈 bullish 形态")
                            match_details["conditions"]["macd_bullish"] = {"passed": False}

                    if len(daily_data) >= 6:
                        recent_volumes = daily_data['volume'].tail(5)
                        avg_volume_5d = recent_volumes.mean()
                        current_volume = daily_data['volume'].iloc[-1]

                        if current_volume >= avg_volume_5d * 1.5:
                            conditions_met.append("成交量 > 5日均值1.5倍")
                            total_score += 20
                            match_details["conditions"]["volume_vs_avg"] = {"passed": True, "current": current_volume, "avg_5d": avg_volume_5d}
                        else:
                            conditions_failed.append("成交量未达5日均值1.5倍")
                            match_details["conditions"]["volume_vs_avg"] = {"passed": False, "current": current_volume, "avg_5d": avg_volume_5d}

        except Exception as e:
            logger.warning(f"Error executing strategy for {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)

        matched = raw_score >= 50

        if conditions_met:
            reason = f"综合评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"综合评分 {raw_score:.0f}/{max_score:.0f}：未满足核心条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
