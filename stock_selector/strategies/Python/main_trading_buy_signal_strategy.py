# -*- coding: utf-8 -*-
"""
主力操盘指标买入信号筛选策略 - Python实现。
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
from indicators.indicators.main_trading import MainTrading

logger = logging.getLogger(__name__)


@register_strategy
class MainTradingBuySignalStrategy(StockSelectorStrategy):
    """
    主力操盘指标买入信号筛选策略 - Python实现。

    该策略基于主力操盘指标（MainTrading）筛选股票，仅识别今日出现买入信号的股票。
    主力操盘指标包含三条线：攻击线、操盘线、防守线，当三条线向上发散且上升时产生买入信号。

    买入信号判定标准：
    - 攻击线 > 操盘线 > 防守线（向上发散）
    - 攻击线和操盘线都在上升
    - 必须是今日最新生成的买入信号
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="main_trading_buy_signal",
            name="main_trading_buy_signal",
            display_name="主力操盘买入信号(Python)",
            description="Python实现的主力操盘指标买入信号筛选策略，通过分析攻击线、操盘线、防守线的走势，筛选出最近出现买入信号的股票。",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _calculate_main_trading_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算主力操盘指标。

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含指标计算结果的字典，失败时返回None
        """
        if df is None or len(df) < 30:
            logger.debug(f"数据不足，需要至少30天，当前只有 {len(df) if df is not None else 0} 天")
            return None

        try:
            df = df.copy()

            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})

            indicator = MainTrading()
            result_df = indicator.calculate(df)

            latest = result_df.iloc[-1]

            buy_signals = result_df['buy_signal']
            
            logger.debug(f"最后10天买入信号: {list(buy_signals.tail(10))}")
            logger.debug(f"今日买入信号: {bool(latest['buy_signal'] == 1)}")

            return {
                'attack_line': float(latest['attack_line']) if pd.notna(latest['attack_line']) else None,
                'trading_line': float(latest['trading_line']) if pd.notna(latest['trading_line']) else None,
                'defense_line': float(latest['defense_line']) if pd.notna(latest['defense_line']) else None,
                'latest_buy_signal': bool(latest['buy_signal'] == 1) if pd.notna(latest['buy_signal']) else False,
                'latest_sell_signal': bool(latest['sell_signal'] == 1) if pd.notna(latest['sell_signal']) else False,
            }
        except Exception as e:
            import traceback
            logger.debug(f"主力操盘指标计算失败: {e}\n{traceback.format_exc()}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        对单只股票执行主力操盘买入信号策略。

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
                match_details["main_trading_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    indicators = self._calculate_main_trading_indicators(daily_data)

                    if indicators:
                        match_details["main_trading_indicators"] = indicators

                        latest_buy = indicators.get('latest_buy_signal', False)

                        if latest_buy:
                            total_score = 100
                            conditions_met.append("今日主力操盘买入信号")
                            match_details["conditions"]["today_buy_signal"] = {"passed": True}
                        else:
                            total_score = 0
                            conditions_failed.append("今日无主力操盘买入信号")
                            match_details["conditions"]["today_buy_signal"] = {"passed": False}

        except Exception as e:
            logger.warning(f"执行策略时出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 60

        if conditions_met:
            reason = f"主力操盘评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"主力操盘评分 {raw_score:.0f}/{max_score:.0f}：未满足核心条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
