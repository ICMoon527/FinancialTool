# -*- coding: utf-8 -*-
"""
Dragon Leader Combined Strategy - Python implementation.
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
class DragonLeaderCombinedStrategy(StockSelectorStrategy):
    """
    Dragon Leader Combined Strategy - Python implementation.
    
    This is a combined strategy that includes three Dragon Leader strategies:
    1. Dragon Leader Start
    2. Dragon Leader Main Rise
    3. Dragon Leader Second Wave
    
    When this strategy is activated, all three sub-strategies are also activated.
    """

    SUB_STRATEGIES = [
        "dragon_leader_start",
        "dragon_leader_main_rise",
        "dragon_leader_second_wave"
    ]

    def __init__(self):
        metadata = StrategyMetadata(
            id="dragon_leader",
            name="dragon_leader",
            display_name="DragonLeader 龙头策略组合",
            description="DragonLeader 龙头策略组合，包含龙头启动、龙头主升、龙头二波三个策略。激活此策略将同时激活所有三个子策略。",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute the Dragon Leader Combined strategy for a single stock.
        
        Note: This is a placeholder implementation since the combined strategy
        mainly serves as an activator for the three sub-strategies. The actual
        matching is done by the individual sub-strategies.

        Args:
            stock_code: Stock code to analyze
            stock_name: Optional stock name

        Returns:
            StrategyMatch result
        """
        match_details = {
            "is_combined_strategy": True,
            "sub_strategies": self.SUB_STRATEGIES,
        }
        
        raw_score = 0.0
        matched = False
        reason = "这是一个组合策略，实际匹配由子策略执行"

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
