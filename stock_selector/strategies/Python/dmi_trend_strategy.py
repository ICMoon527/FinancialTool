# -*- coding: utf-8 -*-
"""
DMI趋势选股策略 - Python实现
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

from indicators.indicators.dmi import DMI

logger = logging.getLogger(__name__)


@register_strategy
class DMITrendStrategyPython(StockSelectorStrategy):
    """
    DMI趋势选股策略 - Python实现
    
    基于DMI（Directional Movement Index）指标的趋势选股策略，通过分析+DI、-DI和ADX的关系，识别强势上涨趋势的股票。
    
    评分规则（满分100分）：
    - +DI > -DI：20分
    - ADX > 20：15分
    - ADX > 25：15分（叠加）
    - ADX > 30：10分（叠加）
    - +DI上穿-DI（近3天）：20分
    - ADX上升趋势（近3天）：10分
    - +DI > 25：10分
    
    匹配条件：raw_score >= 60分
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="dmi_trend_strategy",
            name="dmi_trend_strategy",
            display_name="DMI趋势选股(Python)",
            description="基于DMI指标的趋势选股策略，筛选ADX显示强势趋势且+DI上穿-DI的个股。",
            strategy_type=StrategyType.PYTHON,
            category="technical",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)
        self._dmi_indicator = DMI()

    def _calculate_dmi_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算DMI指标
        
        Returns:
            包含DMI指标计算结果的字典
        """
        if df is None or len(df) < 35:
            logger.debug(f"_calculate_dmi_indicators: 数据不足，df长度={len(df) if df is not None else 0}")
            return None

        try:
            df = df.copy()
            logger.debug(f"_calculate_dmi_indicators: 原始列: {list(df.columns)}")

            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                logger.debug(f"_calculate_dmi_indicators: 重命名后列: {list(df.columns)}")

            dmi_df = self._dmi_indicator.calculate(df)
            logger.debug(f"_calculate_dmi_indicators: 指标计算完成，行数={len(dmi_df)}")

            latest = dmi_df.iloc[-1]
            
            pdi = float(latest['PDI']) if pd.notna(latest['PDI']) else 0.0
            mdi = float(latest['MDI']) if pd.notna(latest['MDI']) else 0.0
            adx = float(latest['ADX']) if pd.notna(latest['ADX']) else 0.0
            
            pdi_above_mdi = pdi > mdi
            
            # 检查最近3天是否有+DI上穿-DI
            golden_cross = False
            lookback_days = min(3, len(dmi_df) - 1)
            for i in range(1, lookback_days + 1):
                if i >= len(dmi_df):
                    break
                prev_pdi = float(dmi_df['PDI'].iloc[-i-1]) if pd.notna(dmi_df['PDI'].iloc[-i-1]) else 0.0
                prev_mdi = float(dmi_df['MDI'].iloc[-i-1]) if pd.notna(dmi_df['MDI'].iloc[-i-1]) else 0.0
                curr_pdi = float(dmi_df['PDI'].iloc[-i]) if pd.notna(dmi_df['PDI'].iloc[-i]) else 0.0
                curr_mdi = float(dmi_df['MDI'].iloc[-i]) if pd.notna(dmi_df['MDI'].iloc[-i]) else 0.0
                
                if prev_pdi <= prev_mdi and curr_pdi > curr_mdi:
                    golden_cross = True
                    break
            
            # 检查ADX上升趋势（最近3天）
            adx_rising = False
            if len(dmi_df) >= 4:
                adx_list = []
                for i in range(3):
                    adx_val = float(dmi_df['ADX'].iloc[-i-1]) if pd.notna(dmi_df['ADX'].iloc[-i-1]) else 0.0
                    adx_list.append(adx_val)
                # 检查是否单调上升
                adx_rising = adx_list[0] > adx_list[1] > adx_list[2]
            
            logger.debug(f"_calculate_dmi_indicators: PDI={pdi:.2f}, MDI={mdi:.2f}, ADX={adx:.2f}, "
                        f"PDI>MDI={pdi_above_mdi}, 金叉={golden_cross}, ADX上升={adx_rising}")

            return {
                'pdi': pdi,
                'mdi': mdi,
                'adx': adx,
                'pdi_above_mdi': pdi_above_mdi,
                'adx_trending': adx > 25,
                'golden_cross': golden_cross,
                'adx_rising': adx_rising,
            }
        except Exception as e:
            logger.debug(f"DMI指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def select(
        self, 
        stock_code: str, 
        stock_name: Optional[str] = None, 
        daily_data: Optional[pd.DataFrame] = None, 
        precomputed_metrics: Optional[Dict[str, Any]] = None
    ) -> StrategyMatch:
        """
        执行DMI趋势选股策略
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
                match_details["dmi_indicators"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source
                match_details["stock_name"] = stock_name

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    change_pct = getattr(realtime_quote, "change_pct", None)
                    match_details["realtime_quote"] = {
                        "price": price,
                        "change_pct": change_pct,
                    }

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    dmi_result = self._calculate_dmi_indicators(daily_data)
                    match_details["dmi_indicators"] = dmi_result

                    if dmi_result:
                        pdi_above_mdi = dmi_result.get('pdi_above_mdi', False)
                        adx = dmi_result.get('adx', 0.0)
                        golden_cross = dmi_result.get('golden_cross', False)
                        adx_rising = dmi_result.get('adx_rising', False)
                        pdi = dmi_result.get('pdi', 0.0)

                        if pdi_above_mdi:
                            conditions_met.append("+DI > -DI")
                            total_score += 20
                            match_details["conditions"]["pdi_above_mdi"] = {"passed": True, "score": 20}
                        else:
                            conditions_failed.append("+DI <= -DI")
                            match_details["conditions"]["pdi_above_mdi"] = {"passed": False, "score": 0}

                        if adx > 20:
                            conditions_met.append(f"ADX > 20 ({adx:.2f})")
                            total_score += 15
                            match_details["conditions"]["adx_gt_20"] = {"passed": True, "score": 15}
                        else:
                            conditions_failed.append(f"ADX <= 20 ({adx:.2f})")
                            match_details["conditions"]["adx_gt_20"] = {"passed": False, "score": 0}

                        if adx > 25:
                            conditions_met.append(f"ADX > 25 ({adx:.2f})")
                            total_score += 15
                            match_details["conditions"]["adx_gt_25"] = {"passed": True, "score": 15}
                        else:
                            conditions_failed.append(f"ADX <= 25 ({adx:.2f})")
                            match_details["conditions"]["adx_gt_25"] = {"passed": False, "score": 0}

                        if adx > 30:
                            conditions_met.append(f"ADX > 30 ({adx:.2f})")
                            total_score += 10
                            match_details["conditions"]["adx_gt_30"] = {"passed": True, "score": 10}
                        else:
                            conditions_failed.append(f"ADX <= 30 ({adx:.2f})")
                            match_details["conditions"]["adx_gt_30"] = {"passed": False, "score": 0}

                        if golden_cross:
                            conditions_met.append("+DI上穿-DI（近3天）")
                            total_score += 20
                            match_details["conditions"]["golden_cross"] = {"passed": True, "score": 20}
                        else:
                            conditions_failed.append("未出现+DI上穿-DI")
                            match_details["conditions"]["golden_cross"] = {"passed": False, "score": 0}

                        if adx_rising:
                            conditions_met.append("ADX上升趋势（近3天）")
                            total_score += 10
                            match_details["conditions"]["adx_rising"] = {"passed": True, "score": 10}
                        else:
                            conditions_failed.append("ADX无上升趋势")
                            match_details["conditions"]["adx_rising"] = {"passed": False, "score": 0}

                        if pdi > 25:
                            conditions_met.append(f"+DI > 25 ({pdi:.2f})")
                            total_score += 10
                            match_details["conditions"]["pdi_gt_25"] = {"passed": True, "score": 10}
                        else:
                            conditions_failed.append(f"+DI <= 25 ({pdi:.2f})")
                            match_details["conditions"]["pdi_gt_25"] = {"passed": False, "score": 0}

                        if len(daily_data) >= 20:
                            ma5 = daily_data['close'].tail(5).mean() if 'close' in daily_data.columns else daily_data['Close'].tail(5).mean()
                            ma10 = daily_data['close'].tail(10).mean() if 'close' in daily_data.columns else daily_data['Close'].tail(10).mean()
                            ma20 = daily_data['close'].tail(20).mean() if 'close' in daily_data.columns else daily_data['Close'].tail(20).mean()
                            if ma5 > ma10 > ma20:
                                conditions_met.append("均线多头排列")
                                total_score += 15
                                match_details["conditions"]["ma_alignment"] = {"passed": True, "score": 15}

                        if self.sector_manager:
                            try:
                                is_hot, sector_name, sector_change_pct = self.sector_manager.is_stock_in_hot_sector(stock_code)
                                if is_hot and sector_name and sector_change_pct:
                                    conditions_met.append(f"板块热点({sector_name}: {sector_change_pct:.2f}%)")
                                    total_score += 10
                                    match_details["conditions"]["sector_hot"] = {"passed": True, "score": 10}
                            except Exception as e:
                                logger.debug(f"板块热点检查失败: {e}")
                    else:
                        conditions_failed.append("DMI指标计算失败或数据不足")
                        match_details["conditions"]["indicator_calculation"] = {"passed": False, "score": 0}

        except Exception as e:
            logger.warning(f"执行策略时出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = raw_score >= 60.0

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
