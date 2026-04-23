# -*- coding: utf-8 -*-
"""
量能异动策略 - Python实现
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

from indicators.indicators.volume_anomaly import VolumeAnomaly

logger = logging.getLogger(__name__)


@register_strategy
class VolumeAnomalyStrategyPython(StockSelectorStrategy):
    """
    量能异动策略 - Python实现
    
    选股条件：
    1. 黄柱出现：成交额亿元 >= REF(HHV(成交额亿元, 10), 1)
    2. 当日涨幅 > 0（放量上涨）
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="volume_anomaly_python",
            name="volume_anomaly_python",
            display_name="量能异动(Python)",
            description="Python实现的量能异动策略，筛选出现黄柱（放量异动）且当日上涨的个股。放量异动往往预示价格可能出现突破。",
            strategy_type=StrategyType.PYTHON,
            category="volume",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)
        self._volume_anomaly_indicator = VolumeAnomaly()

    def _calculate_volume_anomaly(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算量能异动指标
        """
        if df is None or len(df) < 15:
            logger.debug(f"_calculate_volume_anomaly: 数据不足，df长度={len(df) if df is not None else 0}")
            return None

        try:
            df = df.copy()
            logger.debug(f"_calculate_volume_anomaly: 原始列: {list(df.columns)}")

            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                logger.debug(f"_calculate_volume_anomaly: 重命名后列: {list(df.columns)}")

            result_df = self._volume_anomaly_indicator.calculate(df)
            logger.debug(f"_calculate_volume_anomaly: 指标计算完成，行数={len(result_df)}")

            latest = result_df.iloc[-1]

            high_volume_anomaly = bool(latest['high_volume_anomaly']) if pd.notna(latest['high_volume_anomaly']) else False

            prev_close = df['Close'].iloc[-2] if len(df) > 1 else None
            current_close = df['Close'].iloc[-1]
            price_up = False
            pct_change = 0.0
            if prev_close is not None and prev_close != 0:
                pct_change = (current_close - prev_close) / prev_close * 100
                price_up = pct_change > 0

            break_20d_high = False
            if len(df) >= 20:
                high_20d = df['High'].tail(20).max()
                break_20d_high = current_close > high_20d

            logger.debug(f"_calculate_volume_anomaly: 黄柱={high_volume_anomaly}, 上涨={price_up}, 涨跌幅={pct_change:.2f}%")

            return {
                'turnover_billion': float(latest['turnover_billion']) if pd.notna(latest['turnover_billion']) else 0.0,
                'ma5': float(latest['ma5']) if pd.notna(latest['ma5']) else 0.0,
                'ma10': float(latest['ma10']) if pd.notna(latest['ma10']) else 0.0,
                'h10': float(latest['h10']) if pd.notna(latest['h10']) else 0.0,
                'high_volume_anomaly': high_volume_anomaly,
                'price_up': price_up,
                'pct_change': pct_change,
                'break_20d_high': break_20d_high,
            }
        except Exception as e:
            logger.debug(f"量能异动指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None, daily_data: Optional[pd.DataFrame] = None, precomputed_metrics: Optional[Dict[str, Any]] = None) -> StrategyMatch:
        """
        执行量能异动策略
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
                match_details["volume_anomaly_indicators"] = {}
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
                    volume_anomaly_result = self._calculate_volume_anomaly(daily_data)
                    match_details["volume_anomaly_indicators"] = volume_anomaly_result

                    if volume_anomaly_result:
                        high_volume_anomaly = volume_anomaly_result.get('high_volume_anomaly', False)
                        price_up = volume_anomaly_result.get('price_up', False)
                        pct_change = volume_anomaly_result.get('pct_change', 0)
                        break_20d_high = volume_anomaly_result.get('break_20d_high', False)

                        if high_volume_anomaly:
                            conditions_met.append("出现黄柱（放量异动）")
                            total_score += 60
                            match_details["conditions"]["high_volume_anomaly"] = {"passed": True}
                        else:
                            conditions_failed.append("未出现黄柱")
                            match_details["conditions"]["high_volume_anomaly"] = {"passed": False}

                        if price_up:
                            conditions_met.append(f"当日上涨 {pct_change:.2f}%")
                            total_score += 40
                            match_details["conditions"]["price_up"] = {"passed": True}
                        else:
                            conditions_failed.append("当日未上涨")
                            match_details["conditions"]["price_up"] = {"passed": False}

                        if break_20d_high:
                            conditions_met.append("突破20日高点")
                            total_score += 25
                            match_details["conditions"]["break_20d_high"] = {"passed": True}

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

                        if realtime_quote and change_pct is not None:
                            if change_pct >= 9.8:
                                conditions_met.append("涨停")
                                total_score += 20
                                match_details["conditions"]["limit_up"] = {"passed": True}
                    else:
                        conditions_failed.append("量能异动指标计算失败或数据不足")
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
