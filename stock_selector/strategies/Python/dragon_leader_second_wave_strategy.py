# -*- coding: utf-8 -*-
"""
Dragon Leader Second Wave Strategy - Python implementation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy
from stock_selector.strategies.Python.dragon_leader_second_wave import DragonLeaderSecondWave

logger = logging.getLogger(__name__)


@register_strategy
class DragonLeaderSecondWaveStrategy(StockSelectorStrategy):
    """
    Dragon Leader Second Wave Strategy - Python implementation.
    
    This strategy screens stocks based on the Dragon Leader Second Wave indicator,
    which identifies the second wave signal of leading stocks.
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="dragon_leader_second_wave",
            name="dragon_leader_second_wave",
            display_name="龙头二波策略",
            description="识别龙头股第二波上涨信号的策略，整合双龙狙击、龙头量能、动能二号三个指标",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)
        self.indicator = DragonLeaderSecondWave()

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute the Dragon Leader Second Wave strategy for a single stock.

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
                daily_data_result = self._data_provider.get_daily_data(stock_code, days=100)
                
                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, data_source = daily_data_result
                else:
                    daily_data = daily_data_result
                    data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["indicator_results"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {"price": price}

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    daily_data = daily_data.copy()
                    if 'close' in daily_data.columns:
                        daily_data = daily_data.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        })
                    
                    if 'amount' in daily_data.columns:
                        daily_data = daily_data.rename(columns={'amount': 'Amount'})

                    try:
                        indicator_result = self.indicator.calculate(daily_data)
                        match_details["indicator_results"] = indicator_result.iloc[-1].to_dict() if len(indicator_result) > 0 else {}

                        if len(indicator_result) > 0:
                            latest = indicator_result.iloc[-1]
                            
                            if latest.get('second_wave_strong', False):
                                conditions_met.append("龙头二波强信号")
                                total_score += 40
                                match_details["conditions"]["strong_signal"] = {"passed": True}
                            elif latest.get('second_wave_moderate', False):
                                conditions_met.append("龙头二波中等信号")
                                total_score += 25
                                match_details["conditions"]["moderate_signal"] = {"passed": True}
                            else:
                                conditions_failed.append("未检测到龙头二波信号")
                                match_details["conditions"]["signal"] = {"passed": False}

                            if latest.get('second_wave_signal', False):
                                conditions_met.append("二波信号触发")
                                total_score += 30
                                match_details["conditions"]["wave_signal"] = {"passed": True}

                            if latest.get('strong_momentum', False):
                                conditions_met.append("强动能")
                                total_score += 15
                                match_details["conditions"]["momentum"] = {"passed": True}

                            if latest.get('double_volume', False):
                                conditions_met.append("倍量")
                                total_score += 15
                                match_details["conditions"]["volume"] = {"passed": True}
                    except Exception as e:
                        logger.warning(f"Indicator calculation failed: {e}")
                        conditions_failed.append(f"指标计算错误: {str(e)[:50]}")
        except Exception as e:
            logger.warning(f"Error executing strategy for {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 40

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
