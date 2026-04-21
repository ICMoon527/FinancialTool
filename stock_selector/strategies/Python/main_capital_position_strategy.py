
# -*- coding: utf-8 -*-
"""
主力建仓选股策略 - Python实现。
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy
from indicators.indicators.main_capital_absorption import MainCapitalAbsorption

logger = logging.getLogger(__name__)


@register_strategy
class MainCapitalPositionStrategy(StockSelectorStrategy):
    """
    主力建仓选股策略 - Python实现。

    该策略基于主力吸筹指标筛选股票，识别主力正在建仓的股票。
    
    策略逻辑：
    - 今日有吸筹值
    - 今日吸筹值大于阈值（默认5）
    - 今日吸筹值小于前一天吸筹值
    """

    # 默认阈值参数
    ABSORPTION_THRESHOLD = 5.0

    def __init__(self):
        metadata = StrategyMetadata(
            id="main_capital_position",
            name="main_capital_position",
            display_name="主力建仓(Python)",
            description="基于主力吸筹指标的选股策略，识别主力正在建仓的股票",
            strategy_type=StrategyType.PYTHON,
            category="institutional",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _calculate_main_capital_absorption(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算主力吸筹指标。

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含指标计算结果的字典，失败时返回None
        """
        if df is None or len(df) < 100:
            logger.debug(f"数据不足，需要至少100天，当前只有 {len(df) if df is not None else 0} 天")
            return None

        try:
            df = df.copy()

            # 处理列名大小写问题
            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                })

            indicator = MainCapitalAbsorption()
            result_df = indicator.calculate(df)

            absorption_values = result_df['main_capital_absorption']
            latest_value = float(absorption_values.iloc[-1]) if pd.notna(absorption_values.iloc[-1]) else None
            prev_value = float(absorption_values.iloc[-2]) if len(absorption_values) >= 2 and pd.notna(absorption_values.iloc[-2]) else None

            logger.debug(f"最新吸筹值: {latest_value}, 前一天: {prev_value}")

            return {
                'latest_absorption': latest_value,
                'prev_absorption': prev_value,
                'absorption_threshold': self.ABSORPTION_THRESHOLD,
            }
        except Exception as e:
            import traceback
            logger.debug(f"主力吸筹指标计算失败: {e}\n{traceback.format_exc()}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        对单只股票执行主力建仓策略。

        Args:
            stock_code: 要分析的股票代码
            stock_name: 可选的股票名称

        Returns:
            StrategyMatch结果对象
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
                match_details["main_capital_absorption_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    indicators = self._calculate_main_capital_absorption(daily_data)

                    if indicators:
                        match_details["main_capital_absorption_indicators"] = indicators

                        latest_value = indicators.get('latest_absorption')
                        prev_value = indicators.get('prev_absorption')
                        threshold = indicators.get('absorption_threshold', self.ABSORPTION_THRESHOLD)

                        # 检查匹配条件
                        condition1_passed = latest_value is not None and not np.isnan(latest_value) and abs(latest_value) >= 1.01
                        condition2_passed = latest_value is not None and latest_value > threshold
                        condition3_passed = latest_value is not None and prev_value is not None and latest_value < prev_value

                        match_details["conditions"]["has_absorption_value"] = {"passed": condition1_passed}
                        match_details["conditions"]["above_threshold"] = {"passed": condition2_passed}
                        match_details["conditions"]["less_than_prev"] = {"passed": condition3_passed}

                        if condition1_passed:
                            conditions_met.append("今日有吸筹值")
                        else:
                            conditions_failed.append("今日无有效吸筹值")

                        if condition2_passed:
                            conditions_met.append(f"吸筹值大于阈值 {threshold}")
                        else:
                            conditions_failed.append(f"吸筹值小于等于阈值 {threshold}")

                        if condition3_passed:
                            conditions_met.append("吸筹值比前一天小")
                        else:
                            conditions_failed.append("吸筹值不小于前一天")

                        # 所有条件都满足才匹配
                        if condition1_passed and condition2_passed and condition3_passed:
                            total_score = 100.0
                        else:
                            total_score = 0.0

        except Exception as e:
            logger.warning(f"执行策略时出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 50

        if conditions_met:
            reason = f"主力建仓评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"主力建仓评分 {raw_score:.0f}/{max_score:.0f}：未满足核心条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )

