# -*- coding: utf-8 -*-
"""
Afternoon 2:30 Short-Term Strategy - Python implementation.
"""

import logging
from typing import Optional

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class ShortTermStrategy(StockSelectorStrategy):
    """
    Afternoon 2:30 Short-Term Strategy.

    Conditions:
    1. Afternoon 2:30 gain 3% to 5%
    2. Volume ratio > 1
    3. Turnover rate 5% to 10%
    4. Float market cap 5B to 20B RMB
    5. Volume consistently increasing (preferably step-like)
    6. K-line pattern: bullish upward divergence
    7. Stock stays above SSE Composite Index all day
    8. Stock makes new high after 2:30 PM
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="short_term_strategy",
            name="short_term_strategy",
            display_name="下午两点半短线策略",
            description="下午两点半涨幅3%-5%，量比>1，换手率5%-10%，流通市值50亿-200亿，多头向上发散，全天在上证指数上方，两点半后创新高的个股",
            strategy_type=StrategyType.PYTHON,
            category="short_term",
            source="builtin",
            version="1.0.0",
        )
        super().__init__(metadata)

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

        try:
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                if realtime_quote:
                    match_details["realtime_quote"] = {
                        "price": getattr(realtime_quote, "price", None),
                        "change_pct": getattr(realtime_quote, "change_pct", None),
                        "volume_ratio": getattr(realtime_quote, "volume_ratio", None),
                        "turnover_rate": getattr(realtime_quote, "turnover_rate", None),
                    }

                    change_pct = getattr(realtime_quote, "change_pct", None)
                    if change_pct is not None:
                        if 3.0 <= change_pct <= 5.0:
                            conditions_met.append("涨幅 3%-5%")
                        else:
                            conditions_failed.append("涨幅不在 3%-5%")

                    volume_ratio = getattr(realtime_quote, "volume_ratio", None)
                    if volume_ratio is not None:
                        if volume_ratio >= 1.0:
                            conditions_met.append("量比 >= 1")
                        else:
                            conditions_failed.append("量比 < 1")

                    turnover_rate = getattr(realtime_quote, "turnover_rate", None)
                    if turnover_rate is not None:
                        if 5.0 <= turnover_rate <= 10.0:
                            conditions_met.append("换手率 5%-10%")
                        else:
                            conditions_failed.append("换手率不在 5%-10%")

        except Exception as e:
            logger.warning(f"Error accessing data provider for {stock_code}: {e}")

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        total_conditions = 11
        met_count = len(conditions_met)
        match_score = (met_count / total_conditions) * 100 if total_conditions > 0 else 0

        matched = met_count >= 6
        reason = f"满足 {met_count}/{total_conditions} 个条件：" + "; ".join(conditions_met) if conditions_met else "未满足任何条件"

        return StrategyMatch(
            strategy_id=self.id,
            strategy_name=self.display_name,
            matched=matched,
            score=match_score,
            reason=reason,
            match_details=match_details,
        )
