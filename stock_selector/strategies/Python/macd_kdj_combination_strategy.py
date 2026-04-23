# -*- coding: utf-8 -*-
"""
MACD + KDJ 组合共振策略 - Python实现
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

from indicators.indicators.macd import MACD
from indicators.indicators.kdj import KDJ

logger = logging.getLogger(__name__)


@register_strategy
class MACDKDJCombinationStrategyPython(StockSelectorStrategy):
    """
    MACD + KDJ 组合共振策略 - Python实现
    
    选股条件：
    1. MACD金叉：DIF 上穿 DEA
    2. KDJ金叉：K 上穿 D
    3. KDJ低位：K < 30 或 D < 30（可选增强）
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="macd_kdj_combination_python",
            name="macd_kdj_combination_python",
            display_name="MACD+KDJ共振(Python)",
            description="Python实现的MACD+KDJ组合共振策略，筛选MACD和KDJ同时出现金叉信号的个股。KDJ在低位金叉信号更可靠。",
            strategy_type=StrategyType.PYTHON,
            category="technical",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)
        self._macd_indicator = MACD()
        self._kdj_indicator = KDJ()

    def _calculate_macd_kdj(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算MACD和KDJ指标
        
        Returns:
            包含指标计算结果的字典
        """
        if df is None or len(df) < 35:
            logger.debug(f"_calculate_macd_kdj: 数据不足，df长度={len(df) if df is not None else 0}")
            return None

        try:
            df = df.copy()
            logger.debug(f"_calculate_macd_kdj: 原始列: {list(df.columns)}")

            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                logger.debug(f"_calculate_macd_kdj: 重命名后列: {list(df.columns)}")

            macd_df = self._macd_indicator.calculate(df)
            kdj_df = self._kdj_indicator.calculate(df)
            logger.debug(f"_calculate_macd_kdj: 指标计算完成，macd行数={len(macd_df)}, kdj行数={len(kdj_df)}")

            latest = macd_df.iloc[-1]
            prev_macd = macd_df.iloc[-2] if len(macd_df) > 1 else None

            macd_golden_cross = bool(latest['golden_cross']) if pd.notna(latest['golden_cross']) else False
            kdj_golden_cross = bool(kdj_df.iloc[-1]['golden_cross']) if pd.notna(kdj_df.iloc[-1]['golden_cross']) else False
            kdj_k = float(kdj_df.iloc[-1]['K']) if pd.notna(kdj_df.iloc[-1]['K']) else 50
            kdj_d = float(kdj_df.iloc[-1]['D']) if pd.notna(kdj_df.iloc[-1]['D']) else 50
            kdj_low = (kdj_k < 30) or (kdj_d < 30)

            logger.debug(f"_calculate_macd_kdj: macd金叉={macd_golden_cross}, kdj金叉={kdj_golden_cross}, kdj低位={kdj_low}")

            return {
                'macd_dif': float(latest['DIF']) if pd.notna(latest['DIF']) else 0.0,
                'macd_dea': float(latest['DEA']) if pd.notna(latest['DEA']) else 0.0,
                'macd_bar': float(latest['MACD_Bar']) if pd.notna(latest['MACD_Bar']) else 0.0,
                'prev_macd_bar': float(prev_macd['MACD_Bar']) if prev_macd is not None and pd.notna(prev_macd['MACD_Bar']) else None,
                'macd_golden_cross': macd_golden_cross,
                'kdj_k': kdj_k,
                'kdj_d': kdj_d,
                'kdj_j': float(kdj_df.iloc[-1]['J']) if pd.notna(kdj_df.iloc[-1]['J']) else 0.0,
                'kdj_golden_cross': kdj_golden_cross,
                'kdj_low': kdj_low,
            }
        except Exception as e:
            logger.debug(f"MACD+KDJ指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None, daily_data: Optional[pd.DataFrame] = None, precomputed_metrics: Optional[Dict[str, Any]] = None) -> StrategyMatch:
        """
        执行MACD+KDJ组合共振策略
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

                if daily_data is None:
                    daily_data_result = self._data_provider.get_daily_data(stock_code, days=60)
                    if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                        daily_data, data_source = daily_data_result
                    else:
                        daily_data = daily_data_result
                        data_source = "unknown"

                match_details["realtime_quote"] = {}
                match_details["macd_kdj_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source
                match_details["stock_name"] = stock_name

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    macd_kdj_result = self._calculate_macd_kdj(daily_data)
                    match_details["macd_kdj_indicators"] = macd_kdj_result

                    if macd_kdj_result:
                        macd_golden_cross = macd_kdj_result.get('macd_golden_cross', False)
                        kdj_golden_cross = macd_kdj_result.get('kdj_golden_cross', False)
                        kdj_low = macd_kdj_result.get('kdj_low', False)
                        macd_bar = macd_kdj_result.get('macd_bar', 0)
                        prev_macd_bar = macd_kdj_result.get('prev_macd_bar', None)

                        if macd_golden_cross:
                            conditions_met.append("MACD金叉")
                            total_score += 40
                            match_details["conditions"]["macd_golden_cross"] = {"passed": True}
                        else:
                            conditions_failed.append("未出现MACD金叉")
                            match_details["conditions"]["macd_golden_cross"] = {"passed": False}

                        if kdj_golden_cross:
                            conditions_met.append("KDJ金叉")
                            total_score += 40
                            match_details["conditions"]["kdj_golden_cross"] = {"passed": True}
                        else:
                            conditions_failed.append("未出现KDJ金叉")
                            match_details["conditions"]["kdj_golden_cross"] = {"passed": False}

                        if kdj_low:
                            conditions_met.append("KDJ在低位")
                            total_score += 20
                            match_details["conditions"]["kdj_low"] = {"passed": True}
                        else:
                            conditions_failed.append("KDJ不在低位")
                            match_details["conditions"]["kdj_low"] = {"passed": False}

                        if len(daily_data) >= 20:
                            avg_volume_20d = daily_data['volume'].tail(20).mean()
                            current_volume = daily_data['volume'].iloc[-1]
                            volume_above_avg = current_volume > avg_volume_20d * 2
                            if volume_above_avg:
                                conditions_met.append("成交量放大")
                                total_score += 20
                                match_details["conditions"]["volume_above_avg"] = {"passed": True}

                        if prev_macd_bar is not None and prev_macd_bar < 0 and macd_bar > 0:
                            conditions_met.append("MACD柱由负转正")
                            total_score += 10
                            match_details["conditions"]["macd_bar_turn_positive"] = {"passed": True}

                        if len(daily_data) >= 20:
                            ma5 = daily_data['close'].tail(5).mean()
                            ma10 = daily_data['close'].tail(10).mean()
                            ma20 = daily_data['close'].tail(20).mean()
                            if ma5 > ma10 > ma20:
                                conditions_met.append("均线多头排列")
                                total_score += 15
                                match_details["conditions"]["ma_alignment"] = {"passed": True}

                        if self.sector_manager:
                            try:
                                is_hot, sector_name, sector_change_pct = self.sector_manager.is_stock_in_hot_sector(stock_code)
                                if is_hot and sector_name and sector_change_pct:
                                    conditions_met.append(f"板块热点({sector_name}: {sector_change_pct:.2f}%)")
                                    total_score += 10
                                    match_details["conditions"]["sector_hot"] = {"passed": True}
                            except Exception as e:
                                logger.debug(f"板块热点检查失败: {e}")
                    else:
                        conditions_failed.append("MACD+KDJ指标计算失败或数据不足")
                        match_details["conditions"]["indicator_calculation"] = {"passed": False}

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
