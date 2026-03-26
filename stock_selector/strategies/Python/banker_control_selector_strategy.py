# -*- coding: utf-8 -*-
"""
Banker Control Stock Selector Strategy - Python implementation.
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
class BankerControlSelectorStrategy(StockSelectorStrategy):
    """
    Banker Control Stock Selector Strategy - Python implementation.
    
    该策略基于庄家控盘指标筛选股票，筛选出最新控盘度大于50的个股。
    
    控盘度计算公式：
    - AAA：(3 * Close + Open + High + Low) / 6（价格基准）
    - MA12：AAA的12周期EMA
    - MA36：AAA的36周期EMA
    - 控盘度：(MA12 - REF(MA36, 1)) / REF(MA36, 1) * 100 + 50
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="banker_control_selector",
            name="banker_control_selector",
            display_name="庄家控盘选股(Python)",
            description="Python实现的庄家控盘选股策略，筛选最新控盘度大于50的个股。",
            strategy_type=StrategyType.PYTHON,
            category="institutional",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    def _calculate_banker_control_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """计算庄家控盘指标。"""
        if df is None or len(df) < 40:
            logger.debug(f"Data insufficient: df is None or len(df)={len(df) if df is not None else 'None'}")
            return None

        try:
            df = df.copy()

            logger.debug(f"Data columns: {df.columns.tolist()}")
            logger.debug(f"Data shape: {df.shape}")

            close = df['close']
            open_price = df['open']
            high = df['high']
            low = df['low']

            aaa = (3 * close + open_price + high + low) / 6
            ma12 = self._ema(aaa, 12)
            ma36 = self._ema(aaa, 36)
            ma36_prev = ma36.shift(1)

            control_degree = (ma12 - ma36_prev) / ma36_prev * 100 + 50

            low_control = control_degree >= 50
            medium_control = control_degree >= 60
            high_control = control_degree >= 80

            latest_control = float(control_degree.iloc[-1]) if pd.notna(control_degree.iloc[-1]) else None
            latest_low_control = bool(low_control.iloc[-1]) if pd.notna(low_control.iloc[-1]) else False
            latest_medium_control = bool(medium_control.iloc[-1]) if pd.notna(medium_control.iloc[-1]) else False
            latest_high_control = bool(high_control.iloc[-1]) if pd.notna(high_control.iloc[-1]) else False

            logger.debug(f"Calculated control_degree: {latest_control}")

            return {
                'control_degree': latest_control,
                'low_control': latest_low_control,
                'medium_control': latest_medium_control,
                'high_control': latest_high_control,
            }
        except Exception as e:
            logger.error(f"Banker control indicator calculation failed: {e}", exc_info=True)
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        执行庄家控盘选股策略。

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
                daily_data_result = self._data_provider.get_daily_data(stock_code, days=60)

                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, data_source = daily_data_result
                else:
                    daily_data = daily_data_result
                    data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["banker_control_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    indicators = self._calculate_banker_control_indicators(daily_data)

                    if indicators:
                        match_details["banker_control_indicators"] = indicators

                        control_degree = indicators.get('control_degree', 0.0)
                        if control_degree is not None:
                            total_score = min(control_degree, max_score)

                        if indicators.get('high_control', False):
                            conditions_met.append("高控盘(>=80)")
                            match_details["conditions"]["high_control"] = {"passed": True}
                        elif indicators.get('medium_control', False):
                            conditions_met.append("中控盘(>=60)")
                            match_details["conditions"]["medium_control"] = {"passed": True}
                        elif indicators.get('low_control', False):
                            conditions_met.append("低控盘(>=50)")
                            match_details["conditions"]["low_control"] = {"passed": True}
                        else:
                            conditions_failed.append("控盘度<50")
                            match_details["conditions"]["control_degree"] = {"passed": False}

        except Exception as e:
            logger.warning(f"Error executing strategy for {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 50

        if conditions_met:
            reason = f"控盘度 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"控盘度 {raw_score:.0f}/{max_score:.0f}：未满足控盘条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
