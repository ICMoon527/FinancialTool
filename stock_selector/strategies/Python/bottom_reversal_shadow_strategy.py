# -*- coding: utf-8 -*-
"""
底部反转长下影线选股策略 - Python实现。

基于通达信公式翻译：
A1:= (MIN(C,O)-L)/ABS(C-O)≥1.5;   -- 下影线长度 ≥ 实体1.5倍
XL:=BARSLAST(L=LLV(L,30));         -- 距30日最低点的周期数
A2:=L＞REF(L,XL);                  -- 当前最低价 > 30日最低点（已反弹）
XG:=BARSLAST(H=HHV(H,10));         -- 距10日最高点的周期数
A3:=XG＞XL;                        -- 30日低点出现在10日高点之后（先见顶后见底，经历下跌）
A4:=REF(C,XG)/REF(L,XL)＞1.3;      -- 高点收盘价 / 低点最低价 > 1.3（跌幅超30%）
A5:=LLV(L,4)=L;                    -- 当前最低价是4日内最低（局部低点）
RESULT: A1 AND A2 AND A3 AND A4 AND A5;
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class BottomReversalShadowStrategy(StockSelectorStrategy):
    """
    底部反转长下影线选股策略。

    筛选逻辑：
    1. 当日K线具有长下影线（下影线长度 ≥ 实体1.5倍），表明下方有较强的买盘支撑
    2. 当前价格已从30日最低点反弹，不再创新低
    3. 走势结构为先见顶后见底（10日高点在前，30日低点在后），经历了一轮下跌
    4. 从高点至低点的跌幅超过30%，回调充分
    5. 当前处于4日内的局部低点，是潜在的反转位置

    综合来看，该策略用于捕捉大幅下跌后出现底部反转信号的股票。
    """

    # 最少需要的数据天数
    MIN_DATA_DAYS = 35

    def __init__(self):
        metadata = StrategyMetadata(
            id="bottom_reversal_shadow",
            name="bottom_reversal_shadow",
            display_name="底部反转长下影线(Python)",
            description="筛选大幅下跌后出现长下影线底部反转信号的股票。要求：长下影线、30日低点后反弹、先涨后跌结构、跌幅超30%、4日局部低点。",
            strategy_type=StrategyType.PYTHON,
            category="reversal",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        计算选股所需的所有技术指标。

        Args:
            df: 包含OHLCV数据的DataFrame，按时间升序排列

        Returns:
            包含各条件判断结果的字典，失败时返回None
        """
        if df is None or len(df) < self.MIN_DATA_DAYS:
            logger.debug(f"数据不足，需要至少{self.MIN_DATA_DAYS}天，当前只有 {len(df) if df is not None else 0} 天")
            return None

        try:
            # 统一列名为小写
            df = df.copy()
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ('close', 'open', 'high', 'low', 'volume'):
                    col_map[col] = col_lower
            if col_map:
                df = df.rename(columns=col_map)

            close = df['close'].to_numpy(dtype=float)
            open_ = df['open'].to_numpy(dtype=float)
            high = df['high'].to_numpy(dtype=float)
            low = df['low'].to_numpy(dtype=float)
            n = len(df)

            # ---- A1: 下影线长度 ≥ 实体1.5倍 ----
            # 实体 = ABS(C-O)，下影线 = MIN(C,O) - L
            body = np.abs(close[-1] - open_[-1])
            lower_shadow = min(close[-1], open_[-1]) - low[-1]

            if body > 0:
                shadow_ratio = lower_shadow / body
            else:
                # 十字星（C=O），实体为0，只要有下影线就认为满足
                shadow_ratio = 1.5 if lower_shadow > 0 else 0.0

            a1 = shadow_ratio >= 1.5

            # ---- XL: 距30日最低点的周期数 ----
            # LLV(L,30): 最近30根K线的最低价
            lookback_30 = min(30, n)
            recent_lows = low[-lookback_30:]
            llv_30 = np.min(recent_lows)

            # BARSLAST(L=LLV(L,30)): 最近一次等于30日最低点的位置
            xl = None
            for i in range(n - 1, -1, -1):
                if low[i] == llv_30:
                    xl = n - 1 - i  # bars since that bar
                    break

            if xl is None:
                return None

            # ---- A2: 当前最低价 > 30日最低点 ----
            a2 = low[-1] > low[n - 1 - xl]

            # ---- XG: 距10日最高点的周期数 ----
            lookback_10 = min(10, n)
            recent_highs = high[-lookback_10:]
            hhv_10 = np.max(recent_highs)

            xg = None
            for i in range(n - 1, -1, -1):
                if high[i] == hhv_10:
                    xg = n - 1 - i
                    break

            if xg is None:
                return None

            # ---- A3: 10日高点出现在30日低点之前（XG > XL，先见顶后见底，经历下跌） ----
            a3 = xg > xl

            # ---- A4: 高点收盘价 / 低点最低价 > 1.3 ----
            close_at_xg = close[n - 1 - xg]
            low_at_xl = low[n - 1 - xl]

            if low_at_xl > 0:
                price_ratio = close_at_xg / low_at_xl
            else:
                price_ratio = 0.0

            a4 = price_ratio > 1.3

            # ---- A5: 当前最低价是4日内最低 ----
            lookback_4 = min(4, n)
            a5 = low[-1] == np.min(low[-lookback_4:])

            # ---- 综合结果 ----
            result = a1 and a2 and a3 and a4 and a5

            return {
                "a1": bool(a1),
                "a2": bool(a2),
                "a3": bool(a3),
                "a4": bool(a4),
                "a5": bool(a5),
                "result": bool(result),
                "shadow_ratio": round(float(shadow_ratio), 2),
                "lower_shadow": round(float(lower_shadow), 4),
                "body": round(float(body), 4),
                "xl": int(xl),
                "llv_30": round(float(llv_30), 4),
                "xg": int(xg),
                "hhv_10": round(float(hhv_10), 4),
                "price_ratio": round(float(price_ratio), 2),
                "close_at_xg": round(float(close_at_xg), 4),
                "low_at_xl": round(float(low_at_xl), 4),
                "current_low": round(float(low[-1]), 4),
                "llv_4": round(float(np.min(low[-lookback_4:])), 4),
            }
        except Exception as e:
            logger.debug(f"指标计算失败: {e}")
            return None

    def select(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        daily_data: Optional[pd.DataFrame] = None,
        precomputed_metrics: Optional[Dict[str, Any]] = None,
    ) -> StrategyMatch:
        """
        对单只股票执行底部反转长下影线策略。

        Args:
            stock_code: 股票代码
            stock_name: 可选的股票名称
            daily_data: 预计算好的日线数据（可选，用于批量优化）
            precomputed_metrics: 预计算指标（可选）

        Returns:
            StrategyMatch结果对象
        """
        match_details: Dict[str, Any] = {}
        conditions_met: list = []
        conditions_failed: list = []
        total_score = 0.0
        max_score = 100.0
        indicators = None

        try:
            if self._data_provider:
                # 获取日线数据
                if daily_data is None or not isinstance(daily_data, pd.DataFrame) or daily_data.empty:
                    daily_data_result = self._data_provider.get_daily_data(
                        stock_code, days=self.MIN_DATA_DAYS * 3
                    )
                    if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                        daily_data, data_source = daily_data_result
                    else:
                        daily_data = daily_data_result
                        data_source = "unknown"
                else:
                    data_source = "precomputed"

                match_details["data_source"] = data_source
                match_details["conditions"] = {}

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    indicators = self._calculate_indicators(daily_data)

                    if indicators is None:
                        conditions_failed.append("指标计算失败（数据不足或异常）")
                    else:
                        match_details["indicators"] = indicators

                        # 逐条件评分
                        condition_scores = {
                            "a1": ("长下影线（下影≥实体1.5倍）", 25),
                            "a2": ("价格已从30日低点反弹", 15),
                            "a3": ("走势结构：先见顶后见底", 20),
                            "a4": ("高点至低点跌幅超30%", 20),
                            "a5": ("当前处于4日局部低点", 20),
                        }

                        for key, (desc, score) in condition_scores.items():
                            if indicators.get(key, False):
                                conditions_met.append(desc)
                                total_score += score
                                match_details["conditions"][key] = {"passed": True, "desc": desc}
                            else:
                                conditions_failed.append(f"未满足：{desc}")
                                match_details["conditions"][key] = {"passed": False, "desc": desc}

                        # 附加关键数值信息
                        match_details["summary"] = {
                            "下影线/实体比值": indicators.get("shadow_ratio", "N/A"),
                            "30日最低价": indicators.get("llv_30", "N/A"),
                            "当前最低价": indicators.get("current_low", "N/A"),
                            "10日最高价": indicators.get("hhv_10", "N/A"),
                            "高低点价格比": indicators.get("price_ratio", "N/A"),
                        }
                else:
                    conditions_failed.append("无法获取日线数据")

        except Exception as e:
            logger.warning(f"执行策略时出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        matched = indicators.get("result", False) if indicators else False

        if conditions_met:
            reason = f"底部反转评分 {raw_score:.0f}/{max_score:.0f}：" + "; ".join(conditions_met)
        else:
            reason = f"底部反转评分 {raw_score:.0f}/{max_score:.0f}：未满足核心条件"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )