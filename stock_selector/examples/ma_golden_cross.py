# -*- coding: utf-8 -*-
"""
MA Golden Cross Strategy - Python Implementation
均线金叉策略 - Python代码实现
"""

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy


@register_strategy
class MAGoldenCrossStrategy(StockSelectorStrategy):
    """MA Golden Cross Strategy - Python Implementation"""

    def __init__(self):
        metadata = StrategyMetadata(
            id="ma_golden_cross_py",
            name="ma_golden_cross_py",
            display_name="均线金叉(Python)",
            description="均线金叉策略的Python代码实现，检测MA5上穿MA10配合量能确认。",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
        )
        super().__init__(metadata)

    def select(self, stock_code: str, stock_name: str = None) -> StrategyMatch:
        """
        Execute MA Golden Cross strategy.

        Args:
            stock_code: Stock code to analyze
            stock_name: Optional stock name

        Returns:
            StrategyMatch result
        """
        score = 0.0
        matched = False
        reason = ""
        details = {}

        try:
            if self._data_provider:
                quote = self._data_provider.get_realtime_quote(stock_code)
                if quote:
                    details["current_price"] = getattr(quote, "price", 0.0)
                    details["stock_name"] = getattr(quote, "name", stock_code or "")

                history = self._data_provider.get_daily_history(stock_code, limit=30)
                if history and len(history) >= 20:
                    closes = [h.close for h in history]
                    volumes = [h.volume for h in history]

                    ma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else 0
                    ma10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else 0
                    ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else 0
                    avg_vol_5 = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else 0
                    current_vol = volumes[-1] if volumes else 0

                    details["ma5"] = ma5
                    details["ma10"] = ma10
                    details["ma20"] = ma20
                    details["current_volume"] = current_vol
                    details["avg_vol_5"] = avg_vol_5

                    if ma5 > ma10 and closes[-2] < ma10:
                        score += 10
                        reason = "MA5上穿MA10"
                        if current_vol > avg_vol_5 * 1.2:
                            score += 5
                            reason += "，配合量能放大"
                        if ma10 > ma20:
                            score += 3
                            reason += "，多头排列确认"
                        matched = True
                    elif ma10 > ma20 and closes[-2] < ma20:
                        score += 8
                        reason = "MA10上穿MA20"
                        if current_vol > avg_vol_5 * 1.2:
                            score += 3
                        matched = True
                    else:
                        reason = "未形成金叉信号"
            else:
                reason = "无数据提供者，策略需要真实数据"
        except Exception as e:
            reason = f"策略执行错误: {str(e)}"

        return StrategyMatch(
            strategy_id=self.id,
            strategy_name=self.display_name,
            matched=matched,
            score=score,
            reason=reason,
            match_details=details,
        )
