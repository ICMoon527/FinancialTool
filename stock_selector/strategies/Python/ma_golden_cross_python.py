# -*- coding: utf-8 -*-
"""
MA Golden Cross Strategy - Python implementation.
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
class MAGoldenCrossStrategyPython(StockSelectorStrategy):
    """
    MA Golden Cross Strategy - Python implementation.

    This strategy screens stocks based on:
    1. Golden cross detection: MA5 crossing above MA10 or MA10 crossing above MA20
    2. Volume confirmation: Volume > 5-day average and volume ratio > 1.2
    3. Price position: Price near or above crossing MAs, deviation < 5%
    4. MACD confirmation: Golden cross above zero line
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="ma_golden_cross_python",
            name="ma_golden_cross_python",
            display_name="均线金叉(Python)",
            description="Python实现的均线金叉策略，通过代码直接过滤不符合要求的股票，输出符合标准的标的。关注均线金叉、量能确认和价格位置。",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
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

            return {
                'macd': float(latest['macd']),
                'signal': float(latest['signal']),
                'histogram': float(latest['histogram']),
                'golden_cross': golden_cross,
                'above_zero': float(latest['macd']) > 0,
            }
        except Exception as e:
            logger.debug(f"MACD calculation failed: {e}")
            return None

    def _detect_golden_cross(self, df: pd.DataFrame, ma1: int, ma2: int, days: int = 3) -> bool:
        """Detect golden cross between two MAs in recent days."""
        if df is None or len(df) < max(ma1, ma2) + 1:
            return False

        try:
            # Calculate MAs
            df['ma_short'] = df['close'].rolling(window=ma1, min_periods=1).mean()
            df['ma_long'] = df['close'].rolling(window=ma2, min_periods=1).mean()

            # Check for golden cross in recent days
            for i in range(1, days + 1):
                if len(df) >= i + 1:
                    current = df.iloc[-i]
                    previous = df.iloc[-(i + 1)]
                    if previous['ma_short'] <= previous['ma_long'] and current['ma_short'] > current['ma_long']:
                        return True
            return False
        except Exception:
            return False

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute the MA golden cross strategy for a single stock.

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
                    volume_ratio = getattr(realtime_quote, "volume_ratio", None)

                    match_details["realtime_quote"] = {
                        "price": price,
                        "volume_ratio": volume_ratio,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    # Calculate MAs
                    ma5 = self._calculate_ma(daily_data, 5)
                    ma10 = self._calculate_ma(daily_data, 10)
                    ma20 = self._calculate_ma(daily_data, 20)
                    
                    # Calculate volume indicators
                    if len(daily_data) >= 6:
                        avg_volume_5d = daily_data['volume'].tail(5).mean()
                        current_volume = daily_data['volume'].iloc[-1]
                        volume_above_avg = current_volume > avg_volume_5d
                    else:
                        avg_volume_5d = None
                        current_volume = None
                        volume_above_avg = False

                    # Calculate MACD
                    macd_result = self._calculate_macd(daily_data)

                    match_details["daily_indicators"] = {
                        "ma5": ma5,
                        "ma10": ma10,
                        "ma20": ma20,
                        "avg_volume_5d": avg_volume_5d,
                        "current_volume": current_volume,
                        "macd": macd_result,
                    }

                    # Detect golden crosses
                    ma5_ma10_cross = self._detect_golden_cross(daily_data, 5, 10, 3)
                    ma10_ma20_cross = self._detect_golden_cross(daily_data, 10, 20, 3)

                    # Check volume conditions
                    volume_conditions = False
                    if volume_above_avg and realtime_quote and volume_ratio is not None and volume_ratio > 1.2:
                        volume_conditions = True

                    # Check price position
                    price_position = False
                    if realtime_quote and price is not None and ma5 is not None and ma10 is not None:
                        # Price near or above crossing MAs
                        if price >= min(ma5, ma10) * 0.95:  # Deviation < 5%
                            price_position = True

                    # Score calculation
                    if ma5_ma10_cross and volume_conditions:
                        conditions_met.append("MA5 × MA10 金叉配合量能")
                        total_score += 10
                        match_details["conditions"]["ma5_ma10_cross"] = {"passed": True}
                    else:
                        if not ma5_ma10_cross:
                            conditions_failed.append("未出现 MA5 × MA10 金叉")
                            match_details["conditions"]["ma5_ma10_cross"] = {"passed": False}
                        if not volume_conditions:
                            conditions_failed.append("量能条件未满足")
                            match_details["conditions"]["volume_conditions"] = {"passed": False}

                    if ma10_ma20_cross:
                        conditions_met.append("MA10 × MA20 金叉")
                        total_score += 8
                        match_details["conditions"]["ma10_ma20_cross"] = {"passed": True}
                    else:
                        conditions_failed.append("未出现 MA10 × MA20 金叉")
                        match_details["conditions"]["ma10_ma20_cross"] = {"passed": False}

                    if macd_result and macd_result.get("golden_cross") and macd_result.get("above_zero"):
                        conditions_met.append("MACD零轴上方金叉")
                        total_score += 5
                        match_details["conditions"]["macd_golden_cross"] = {"passed": True}
                    else:
                        conditions_failed.append("MACD条件未满足")
                        match_details["conditions"]["macd_golden_cross"] = {"passed": False}

                    if price_position:
                        conditions_met.append("价格位置合理")
                        total_score += 5
                        match_details["conditions"]["price_position"] = {"passed": True}
                    else:
                        conditions_failed.append("价格位置不合理")
                        match_details["conditions"]["price_position"] = {"passed": False}

        except Exception as e:
            logger.warning(f"Error executing strategy for {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        match_score = min(total_score, max_score)

        matched = match_score >= 15  # Minimum score to match

        if conditions_met:
            reason = f"综合评分 {match_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"综合评分 {match_score:.0f}/{max_score:.0f}：未满足核心条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return StrategyMatch(
            strategy_id=self.id,
            strategy_name=self.display_name,
            matched=matched,
            score=match_score,
            reason=reason,
            match_details=match_details,
        )
