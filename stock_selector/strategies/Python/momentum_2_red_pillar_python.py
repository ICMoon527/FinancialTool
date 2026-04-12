# -*- coding: utf-8 -*-
"""
动能二号红色柱策略 - Python实现。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from indicators.indicators.momentum_2 import Momentum2

logger = logging.getLogger(__name__)


@register_strategy
class Momentum2RedPillarStrategyPython(StockSelectorStrategy):
    """
    动能二号红色柱策略 - Python实现。

    该策略筛选动能二号指标呈现红色柱状态的个股：
    - 红色柱表示资金正在主导价格上涨或即将主导价格上涨
    - 判断条件：动能价格 > 0 且动能价格 > 前一日动能价格
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="momentum_2_red_pillar_python",
            name="momentum_2_red_pillar_python",
            display_name="动能二号红色柱(Python)",
            description="Python实现的动能二号红色柱策略，筛选最新交易日动能二号指标呈现红色柱状态的个股。红色柱表示资金正在主导价格上涨或即将主导价格上涨。",
            strategy_type=StrategyType.PYTHON,
            category="momentum",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)
        self._momentum2_indicator = Momentum2()

    def _calculate_momentum2(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算动能二号指标

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含指标计算结果的字典，数据不足时返回None
        """
        if df is None or len(df) < 6:
            logger.debug(f"_calculate_momentum2: 数据不足 df={df is None}, len={len(df) if df is not None else 0}")
            return None

        try:
            df = df.copy()
            logger.debug(f"_calculate_momentum2: 原始列: {list(df.columns)}")
            
            if 'Open' not in df.columns and 'open' in df.columns:
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                logger.debug(f"_calculate_momentum2: 重命名后列: {list(df.columns)}")

            result_df = self._momentum2_indicator.calculate(df)
            logger.debug(f"_calculate_momentum2: result_df 行数={len(result_df)}")

            latest = result_df.iloc[-1]
            prev = result_df.iloc[-2] if len(result_df) > 1 else None

            is_red_pillar = bool(latest['strong_momentum1']) if pd.notna(latest['strong_momentum1']) else False

            logger.debug(f"_calculate_momentum2: is_red_pillar={is_red_pillar}, latest['strong_momentum1']={latest['strong_momentum1']}")

            return {
                'momentum_price': float(latest['momentum_price']),
                'prev_momentum_price': float(prev['momentum_price']) if prev is not None else None,
                'is_red_pillar': is_red_pillar,
                'strong_momentum': float(latest['strong_momentum']),
                'medium_momentum': float(latest['medium_momentum']),
                'no_momentum': float(latest['no_momentum']),
                'recovery_momentum': float(latest['recovery_momentum']),
            }
        except Exception as e:
            logger.debug(f"动能二号指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None, daily_data: Optional[pd.DataFrame] = None, precomputed_metrics: Optional[Dict[str, Any]] = None) -> StrategyMatch:
        """
        执行动能二号红色柱策略对单只股票进行筛选。

        Args:
            stock_code: 要分析的股票代码
            stock_name: 可选的股票名称

        Returns:
            StrategyMatch 结果对象
        """
        match_details = {}
        conditions_met = []
        conditions_failed = []
        total_score = 0.0
        max_score = 100.0

        try:
            data_source = "preloaded"
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                
                # 如果传入了 daily_data，直接使用；否则从数据提供者获取
                if daily_data is None:
                    daily_data_result = self._data_provider.get_daily_data(stock_code, days=60)
                    if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                        daily_data, data_source = daily_data_result
                    else:
                        daily_data = daily_data_result
                        data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["momentum2_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source
                match_details["stock_name"] = stock_name

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    momentum2_result = self._calculate_momentum2(daily_data)
                    match_details["momentum2_indicators"] = momentum2_result

                    if momentum2_result:
                        is_red_pillar = momentum2_result.get('is_red_pillar', False)
                        momentum_price = momentum2_result.get('momentum_price', 0)

                        if is_red_pillar:
                            conditions_met.append(f"动能二号红色柱，动能价格: {momentum_price:.4f}")
                            total_score = 100.0
                            match_details["conditions"]["red_pillar"] = {"passed": True}
                        else:
                            conditions_failed.append("最新交易日非动能二号红色柱")
                            match_details["conditions"]["red_pillar"] = {"passed": False}
                    else:
                        conditions_failed.append("动能二号指标计算失败或数据不足")
                        match_details["conditions"]["red_pillar"] = {"passed": False}

        except Exception as e:
            logger.warning(f"执行策略时出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 100.0

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
