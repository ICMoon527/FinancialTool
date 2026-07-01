# -*- coding: utf-8 -*-
"""
金蛤蟆形态选股策略 - Python 实现。

识别日K线金蛤蟆形态（6条铁律版本）：
  铁律1: 右眼价格必须高于左眼（右眼≤左眼直接排除）
  铁律2: 右爪价格必须高于左爪（低点逐步抬高）
  铁律3: 左爪和右爪必须缩量（芝麻点）
  铁律4: 左腿长（吸筹时间长）> 右腿短（洗盘时间短）
  铁律5: 右爪悬空在60日线上方（强势股特征）
  铁律6: 左右眼RSI接近（排除顶背离的假M头）

完整结构：
  左爪（缩量低点，60日线附近）→ 左眼（放量波段高点）→ 塌背（缩量洗盘）
  → 右眼（放量波段高点，>左眼）→ 右爪（缩量低点，60日线上方，>左爪）

两个买点：
  - 买点1（右爪缩量低吸）：右爪缩量企稳，左侧低吸
  - 买点2（放量突破颈线）：放量突破左右眼连线颈线，右侧追涨
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stock_selector.base import StockSelectorStrategy, StrategyMatch, StrategyMetadata, StrategyType
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 策略参数
# ---------------------------------------------------------------------------

MIN_DATA_LENGTH = 60  # 最少需要的数据条数
LEFT_EYE_SEARCH_START = 0.20  # 左眼搜索区间起始（数据比例）
LEFT_EYE_SEARCH_END = 0.60  # 左眼搜索区间结束（数据比例）
LOCAL_HIGH_WINDOW = 5  # 局部高点检测窗口（前后各N天）
LEFT_EYE_VOL_RATIO = 1.5  # 左眼量能阈值（> 前20日均量的倍数）
RIGHT_EYE_VOL_RATIO = 1.2  # 右眼量能阈值
RIGHT_EYE_UPPER_SHADOW_MAX = 0.40  # 右眼上影线占比上限（避免冲高回落的假突破）
MIN_EYE_DISTANCE = 15  # 左右眼最小间隔（交易日）
RIGHT_EYE_HEIGHT_RATIO = 0.90  # 右眼不低于左眼的最低比例（量能/上影线等其他条件通过时的宽松阈值）
EYE_UPTREND_MIN_DAYS = 5  # 眼睛波段：上升段最少天数
EYE_TOP_ZONE_PCT = 2.0  # 眼睛波段：筑顶区价格范围（%）
EYE_PULLBACK_MIN_DAYS = 3  # 眼睛波段：回踩最少天数
MA60_TREND_SLOPE_PCT = -1.0  # MA60趋势斜率阈值（%）
MA60_SLOPE_LOOKBACK = 20  # MA60斜率计算回看天数
MA60_ABOVE_RATIO_MIN = 0.70  # 左眼到右眼之间，收盘价在MA60上方的最小比例
VOL_MA_PERIOD = 20  # 均量计算周期
ESSENTIAL_THRESHOLD = 50  # 必要条件匹配阈值
BUY_POINT_1_MA_DIST_PCT = 5.0  # 买点1：价格距MA60的最大距离（%）
BUY_POINT_2_VOL_RATIO = 1.0  # 买点2：量能阈值

# --- 6条铁律新增参数 ---
LEFT_CLAW_VOL_RATIO = 0.8  # 左爪缩量阈值（< 前20日均量的倍数）
LEFT_CLAW_MA_DIST_PCT = 8.0  # 左爪距MA60最大距离（%）
RIGHT_CLAW_VOL_RATIO = 0.8  # 右爪缩量阈值（< 前20日均量的倍数）
RIGHT_CLAW_MA_DIST_PCT = 5.0  # 右爪距MA60最大距离（%），悬空要求
RIGHT_CLAW_MIN_DOWN_DAYS = 2  # 右爪最少连续下跌天数
RIGHT_CLAW_RECOVERY_PCT = 2.0  # 右爪回踩后回升确认比例（%）
EYE_RSI_PERIOD = 14  # RSI计算周期
EYE_RSI_MAX_DIFF = 0.20  # 左右眼RSI最大差异（比例，排除顶背离）
LEFT_LEG_RIGHT_LEG_MIN_RATIO = 1.0  # 左腿/右腿最短比例（左腿不能比右腿短）


@register_strategy
class GoldenToadStrategy(StockSelectorStrategy):
    """
    金蛤蟆形态选股策略（6条铁律版本）。

    识别日K线金蛤蟆形态，判断两个买点：
    1. 右爪缩量低吸买点（左侧低吸）
    2. 放量突破颈线买点（右侧追涨）
    """

    # ------------------------------------------------------------------
    # 评分常量
    # ------------------------------------------------------------------
    SCORE_MA60_TREND = 15  # 60日均线趋势
    SCORE_LEFT_CLAW = 10  # 左爪形态（缩量+近60日线）
    SCORE_LEFT_EYE = 20  # 左眼形态（波段+放量）
    SCORE_RIGHT_EYE = 25  # 右眼形态（波段+放量+上影线）
    SCORE_RIGHT_CLAW = 20  # 右爪形态（缩量+悬空+回升）
    SCORE_RSI_MATCH = 10  # 左右眼RSI一致（排除顶背离）
    SCORE_ESSENTIAL_MAX = 100  # 必要条件满分

    SCORE_BUY_POINT_1 = 10  # 加分项：买点1触发
    SCORE_BUY_POINT_2 = 10  # 加分项：买点2触发
    SCORE_MA_BULLISH = 5  # 加分项：短期均线多头
    SCORE_BONUS_MAX = 25  # 加分项满分

    # ------------------------------------------------------------------
    # 6条铁律（硬约束，不满足直接排除）
    # ------------------------------------------------------------------
    # 铁律1: 右眼 > 左眼（在 select 中检查）
    # 铁律2: 右爪 > 左爪（在 select 中检查）
    # 铁律5: 右爪悬空在 MA60 上方（在 select 中检查）

    def __init__(self) -> None:
        metadata = StrategyMetadata(
            id="golden_toad",
            name="golden_toad",
            display_name="金蛤蟆形态",
            description=(
                "识别日K线金蛤蟆形态（6条铁律）：60日均线走平向上→左爪缩量→"
                "左眼放量→塌背洗盘→右眼放量(>左眼)→右爪缩量悬空(>左爪)→"
                "突破颈线。两个买点：右爪缩量低吸/放量突破追涨。"
            ),
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="2.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)

    # ==================================================================
    # 辅助方法：技术指标计算
    # ==================================================================

    @staticmethod
    def _calc_ma(data: "pd.Series", period: int) -> "pd.Series":
        """计算移动平均线。"""
        return data.rolling(window=period, min_periods=1).mean()  # type: ignore[return-value]

    @staticmethod
    def _calc_vol_ma(df: pd.DataFrame, period: int = VOL_MA_PERIOD) -> "pd.Series":
        """计算成交量均线。"""
        return df["volume"].rolling(window=period, min_periods=1).mean()  # type: ignore[return-value]

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = EYE_RSI_PERIOD) -> pd.Series:
        """计算RSI指标。"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)  # type: ignore[return-value]

    @staticmethod
    def _linear_slope(y: np.ndarray) -> float:
        """计算一维数组的线性回归斜率（最小二乘法）。"""
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    # ==================================================================
    # 辅助方法：形态识别
    # ==================================================================

    def _find_local_highs(self, high_series: pd.Series, window: int = LOCAL_HIGH_WINDOW) -> List[int]:
        """
        在序列中寻找局部高点（索引列表）。

        局部高点定义：该点 > 前后各 window 天的最高价。
        """
        n = len(high_series)
        if n < 2 * window + 1:
            return []

        roll_max = high_series.rolling(window=2 * window + 1, center=True, min_periods=1).max()

        highs: List[int] = []
        for i in range(window, n - window):
            if high_series.iloc[i] == roll_max.iloc[i]:
                left = max(0, i - window)
                right = min(n - 1, i + window)
                if high_series.iloc[i] == high_series.iloc[left : right + 1].max():
                    highs.append(i)

        return highs

    def _calc_ma60_trend(self, df: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        计算60日均线趋势。

        Returns:
            (trend_ok, slope_pct, ma60_latest)
        """
        if len(df) < 60:
            return False, 0.0, 0.0

        ma60 = self._calc_ma(df["close"], 60)  # type: ignore[arg-type]
        ma60_latest = float(ma60.iloc[-1])

        recent_ma60 = ma60.iloc[-MA60_SLOPE_LOOKBACK:].dropna()
        if len(recent_ma60) < 5:
            return False, 0.0, ma60_latest

        slope = self._linear_slope(recent_ma60.values)
        mean_ma60 = float(recent_ma60.mean())
        if mean_ma60 == 0:
            slope_pct = 0.0
        else:
            slope_pct = (slope / mean_ma60) * 100.0

        trend_ok = slope_pct >= MA60_TREND_SLOPE_PCT
        return trend_ok, slope_pct, ma60_latest

    def _detect_eye_wave(
        self,
        df: pd.DataFrame,
        search_start: int,
        search_end: int,
        ma60: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """
        在指定区间内搜索完整的眼睛波段：上升趋势 → 筑顶 → 回踩。

        算法：
        1. 找局部高点候选
        2. 对每个候选，验证：上升段（≥EYE_UPTREND_MIN_DAYS）→
           筑顶区（峰值±EYE_TOP_ZONE_PCT内）→ 回踩段（≥EYE_PULLBACK_MIN_DAYS，在MA60上方）

        Returns:
            None 表示未找到，否则返回眼睛信息字典（含 uptrend_start 用于左爪检测）。
        """
        n = len(df)

        # Step 1: 找候选局部高点
        local_highs = self._find_local_highs(df["high"], LOCAL_HIGH_WINDOW)  # type: ignore[arg-type]
        candidates = [i for i in local_highs if search_start <= i <= search_end]

        if not candidates:
            return None

        # 按价格从高到低排序，优先验证高价候选
        candidates.sort(key=lambda i: float(df["high"].iloc[i]), reverse=True)

        for cand_idx in candidates:
            cand_high = float(df["high"].iloc[cand_idx])

            # ----------------------------------------------------------
            # Step 2: 验证上升趋势（从 cand_idx 往前找连续走高的起点）
            # ----------------------------------------------------------
            uptrend_start = cand_idx
            for i in range(cand_idx - 1, max(0, cand_idx - 30), -1):
                prev_close = float(df["close"].iloc[i])
                curr_close = float(df["close"].iloc[i + 1])
                prev_high = float(df["high"].iloc[i])
                curr_high = float(df["high"].iloc[i + 1])
                if prev_close <= curr_close or prev_high <= curr_high:
                    uptrend_start = i
                else:
                    break

            uptrend_days = cand_idx - uptrend_start
            if uptrend_days < EYE_UPTREND_MIN_DAYS:
                continue

            # ----------------------------------------------------------
            # Step 3: 验证筑顶区（峰值附近在 EYE_TOP_ZONE_PCT 范围内）
            # ----------------------------------------------------------
            top_zone_start = cand_idx
            top_zone_end = cand_idx
            for i in range(cand_idx - 1, max(0, cand_idx - 3), -1):
                if float(df["high"].iloc[i]) >= cand_high * (1 - EYE_TOP_ZONE_PCT / 100):
                    top_zone_start = i
                else:
                    break
            for i in range(cand_idx + 1, min(n, cand_idx + 3)):
                if float(df["high"].iloc[i]) >= cand_high * (1 - EYE_TOP_ZONE_PCT / 100):
                    top_zone_end = i
                else:
                    break

            eye_position = top_zone_end

            # ----------------------------------------------------------
            # Step 4: 验证回踩段（筑顶区后至少 EYE_PULLBACK_MIN_DAYS 天收盘价走低，且在 MA60 上方）
            # ----------------------------------------------------------
            pullback_start = top_zone_end + 1
            if pullback_start >= n - EYE_PULLBACK_MIN_DAYS:
                continue

            down_count = 0
            pullback_low_close = float("inf")
            pullback_low_pos = pullback_start
            pullback_low_low = float("inf")

            for i in range(pullback_start, min(n, pullback_start + 20)):
                close_i = float(df["close"].iloc[i])
                low_i = float(df["low"].iloc[i])

                if i == pullback_start or close_i < float(df["close"].iloc[i - 1]):
                    down_count += 1
                    if close_i < pullback_low_close:
                        pullback_low_close = close_i
                        pullback_low_low = low_i
                        pullback_low_pos = i
                else:
                    break

                ma60_val = float(ma60.iloc[i]) if pd.notna(ma60.iloc[i]) else 0.0
                if ma60_val > 0 and close_i <= ma60_val:
                    down_count = 0
                    break

            if down_count < EYE_PULLBACK_MIN_DAYS:
                continue

            # 眼睛有效！
            return {
                "eye_position": eye_position,
                "eye_price": cand_high,
                "eye_close": float(df["close"].iloc[eye_position]),
                "pullback_low_close": pullback_low_close,
                "pullback_low_low": pullback_low_low,
                "pullback_position": pullback_low_pos,
                "uptrend_days": uptrend_days,
                "pullback_days": down_count,
                "top_zone_start": top_zone_start,
                "top_zone_end": top_zone_end,
                "uptrend_start": uptrend_start,  # 用于左爪检测
            }

        return None

    def _detect_left_eye(self, df: pd.DataFrame, ma60: pd.Series) -> Dict[str, Any]:
        """
        识别左眼波段：上升趋势 → 筑顶 → 回踩在 MA60 上方。

        Returns:
            {
                "found": bool, "position": int, "price": float,
                "volume_ratio": float, "pullback_position": int,
                "pullback_close": float, "pullback_days": int,
                "uptrend_days": int, "uptrend_start": int, "ok": bool
            }
        """
        n = len(df)
        start_idx = int(n * LEFT_EYE_SEARCH_START)
        end_idx = int(n * LEFT_EYE_SEARCH_END)

        wave = self._detect_eye_wave(df, start_idx, end_idx, ma60)
        if wave is None:
            return {
                "found": False, "position": -1, "price": 0.0,
                "volume_ratio": 0.0, "pullback_position": -1,
                "pullback_close": 0.0, "pullback_days": 0,
                "uptrend_days": 0, "uptrend_start": -1, "ok": False,
            }

        best_idx = wave["eye_position"]
        best_price = wave["eye_price"]

        vol_ma = self._calc_vol_ma(df)
        vol_ma_val = float(vol_ma.iloc[best_idx]) if best_idx < len(vol_ma) else 0.0
        if vol_ma_val > 0:
            vol_ratio = float(df["volume"].iloc[best_idx]) / vol_ma_val
        else:
            vol_ratio = 0.0

        vol_ok = vol_ratio > LEFT_EYE_VOL_RATIO
        ok = vol_ok

        return {
            "found": True,
            "position": best_idx,
            "price": best_price,
            "volume_ratio": round(vol_ratio, 2),
            "pullback_position": wave["pullback_position"],
            "pullback_close": round(wave["pullback_low_close"], 2),
            "pullback_days": wave["pullback_days"],
            "uptrend_days": wave["uptrend_days"],
            "uptrend_start": wave["uptrend_start"],
            "ok": ok,
        }

    def _detect_left_claw(
        self,
        df: pd.DataFrame,
        left_eye_uptrend_start: int,
        ma60: pd.Series,
    ) -> Dict[str, Any]:
        """
        识别左爪：左眼上升段起点附近，靠近60日线的缩量低点。

        铁律3: 左爪必须缩量（量芝麻点）
        铁律4: 左爪位置用于计算左腿长度

        Returns:
            {
                "found": bool, "position": int, "close_price": float,
                "low_price": float, "volume_ratio": float,
                "ma60_dist_pct": float, "ok": bool
            }
        """
        n = len(df)
        search_start = max(0, left_eye_uptrend_start - 3)
        search_end = min(n - 1, left_eye_uptrend_start + 3)

        if search_start > search_end:
            return {
                "found": False, "position": -1, "close_price": 0.0,
                "low_price": 0.0, "volume_ratio": 0.0,
                "ma60_dist_pct": 0.0, "ok": False,
            }

        # 在搜索区间内找收盘价最低点（代表左爪）
        best_idx = search_start
        best_close = float("inf")
        for i in range(search_start, search_end + 1):
            close_i = float(df["close"].iloc[i])
            if close_i < best_close:
                best_close = close_i
                best_idx = i

        low_price = float(df["low"].iloc[best_idx])

        # 量能确认（缩量 = 芝麻点）
        vol_ma = self._calc_vol_ma(df)
        vol_ma_val = float(vol_ma.iloc[best_idx]) if best_idx < len(vol_ma) else 0.0
        if vol_ma_val > 0:
            vol_ratio = float(df["volume"].iloc[best_idx]) / vol_ma_val
        else:
            vol_ratio = 0.0

        vol_ok = vol_ratio < LEFT_CLAW_VOL_RATIO

        # 距60日线距离
        ma60_val = float(ma60.iloc[best_idx]) if pd.notna(ma60.iloc[best_idx]) else 0.0
        if ma60_val > 0:
            ma60_dist_pct = abs(best_close - ma60_val) / ma60_val * 100.0
        else:
            ma60_dist_pct = 100.0

        near_ma60 = ma60_dist_pct <= LEFT_CLAW_MA_DIST_PCT

        ok = vol_ok and near_ma60

        return {
            "found": True,
            "position": best_idx,
            "close_price": round(best_close, 2),
            "low_price": round(low_price, 2),
            "volume_ratio": round(vol_ratio, 2),
            "ma60_dist_pct": round(ma60_dist_pct, 2),
            "ok": ok,
        }

    def _detect_right_eye(
        self,
        df: pd.DataFrame,
        left_eye_pos: int,
        left_eye_price: float,
        ma60: pd.Series,
    ) -> Dict[str, Any]:
        """
        识别右眼波段：上升趋势 → 筑顶 → 回踩在 MA60 上方。

        在左眼之后搜索，使用与左眼相同的波段检测算法。
        额外检查：上影线占比、高度比例、量能。

        Returns:
            {
                "found": bool, "position": int, "price": float,
                "volume_ratio": float, "height_ratio": float,
                "upper_shadow_ratio": float, "pullback_position": int,
                "pullback_close": float, "pullback_days": int,
                "uptrend_days": int, "ok": bool
            }
        """
        n = len(df)
        search_start = left_eye_pos + MIN_EYE_DISTANCE
        search_end = n - 20

        if search_start >= search_end or search_start >= n:
            return {
                "found": False, "position": -1, "price": 0.0,
                "volume_ratio": 0.0, "height_ratio": 0.0,
                "upper_shadow_ratio": 0.0, "pullback_position": -1,
                "pullback_close": 0.0, "pullback_days": 0,
                "uptrend_days": 0, "ok": False,
            }

        wave = self._detect_eye_wave(df, search_start, search_end, ma60)
        if wave is None:
            return {
                "found": False, "position": -1, "price": 0.0,
                "volume_ratio": 0.0, "height_ratio": 0.0,
                "upper_shadow_ratio": 0.0, "pullback_position": -1,
                "pullback_close": 0.0, "pullback_days": 0,
                "uptrend_days": 0, "ok": False,
            }

        best_idx = wave["eye_position"]
        best_price = wave["eye_price"]

        height_ratio = best_price / left_eye_price if left_eye_price > 0 else 0.0

        vol_ma = self._calc_vol_ma(df)
        vol_ma_val = float(vol_ma.iloc[best_idx]) if best_idx < len(vol_ma) else 0.0
        if vol_ma_val > 0:
            vol_ratio = float(df["volume"].iloc[best_idx]) / vol_ma_val
        else:
            vol_ratio = 0.0

        # K 线质量：上影线占比
        row = df.iloc[best_idx]
        candle_high = float(row["high"])
        candle_low = float(row["low"])
        candle_open = float(row["open"])
        candle_close = float(row["close"])
        total_range = candle_high - candle_low
        if total_range > 0:
            upper_shadow = candle_high - max(candle_open, candle_close)
            upper_shadow_ratio = upper_shadow / total_range
        else:
            upper_shadow_ratio = 0.0

        shadow_ok = upper_shadow_ratio <= RIGHT_EYE_UPPER_SHADOW_MAX
        vol_ok = vol_ratio > RIGHT_EYE_VOL_RATIO
        height_ok = height_ratio >= RIGHT_EYE_HEIGHT_RATIO
        ok = vol_ok and height_ok and shadow_ok

        return {
            "found": True,
            "position": best_idx,
            "price": best_price,
            "volume_ratio": round(vol_ratio, 2),
            "height_ratio": round(height_ratio, 2),
            "upper_shadow_ratio": round(upper_shadow_ratio, 2),
            "pullback_position": wave["pullback_position"],
            "pullback_close": round(wave["pullback_low_close"], 2),
            "pullback_days": wave["pullback_days"],
            "uptrend_days": wave["uptrend_days"],
            "ok": ok,
        }

    def _detect_right_claw(
        self,
        df: pd.DataFrame,
        right_eye_pos: int,
        right_eye_close: float,
        left_eye_pos: int,
        ma60: pd.Series,
    ) -> Dict[str, Any]:
        """
        识别右爪（右眼后的缩量回踩）。

        铁律3: 右爪必须缩量（芝麻点）
        铁律5: 右爪悬空在60日线上方（强势股）

        算法：
        1. 找收盘价连续下跌波段（至少 RIGHT_CLAW_MIN_DOWN_DAYS 天）
        2. 取该波段中收盘价最低的那天作为右爪
        3. 回踩低点收盘价必须低于右眼收盘价（真正回踩）
        4. 确认缩量、悬空在MA60上方
        5. 确认从低点已回升 >= RIGHT_CLAW_RECOVERY_PCT %

        Returns:
            {
                "found": bool, "low_position": int, "low_price": float,
                "close_price": float, "volume_ratio": float,
                "above_ma60": bool, "recovery_pct": float,
                "below_right_eye_close": bool, "ok": bool
            }
        """
        n = len(df)
        search_start = right_eye_pos + 1
        search_end = n - 1

        if search_start >= search_end:
            return {
                "found": False, "low_position": -1, "low_price": 0.0,
                "close_price": 0.0, "volume_ratio": 0.0,
                "above_ma60": False, "recovery_pct": 0.0,
                "below_right_eye_close": False, "ok": False,
            }

        vol_ma = self._calc_vol_ma(df)

        # Step 1: 找收盘价连续下跌波段
        closes = df["close"].iloc[search_start : search_end + 1]
        if len(closes) < RIGHT_CLAW_MIN_DOWN_DAYS:
            return {
                "found": False, "low_position": -1, "low_price": 0.0,
                "close_price": 0.0, "volume_ratio": 0.0,
                "above_ma60": False, "recovery_pct": 0.0,
                "below_right_eye_close": False, "ok": False,
            }

        down_segments: List[Tuple[int, int]] = []
        seg_start = search_start
        for i in range(search_start + 1, search_end + 1):
            if float(df["close"].iloc[i]) < float(df["close"].iloc[i - 1]):
                continue
            else:
                if i - seg_start >= RIGHT_CLAW_MIN_DOWN_DAYS:
                    down_segments.append((seg_start, i - 1))
                seg_start = i

        if search_end + 1 - seg_start >= RIGHT_CLAW_MIN_DOWN_DAYS:
            down_segments.append((seg_start, search_end))

        if not down_segments:
            return {
                "found": False, "low_position": -1, "low_price": 0.0,
                "close_price": 0.0, "volume_ratio": 0.0,
                "above_ma60": False, "recovery_pct": 0.0,
                "below_right_eye_close": False, "ok": False,
            }

        # Step 2: 在每个下跌波段中找收盘价最低点
        best_seg_idx = -1
        best_close = float("inf")
        for seg_s, seg_e in down_segments:
            seg_closes = df["close"].iloc[seg_s : seg_e + 1]
            seg_min_close = float(seg_closes.min())
            if seg_min_close < best_close:
                best_close = seg_min_close
                best_seg_idx = seg_s + int(seg_closes.idxmin() - seg_s)

        if best_seg_idx == -1:
            return {
                "found": False, "low_position": -1, "low_price": 0.0,
                "close_price": 0.0, "volume_ratio": 0.0,
                "above_ma60": False, "recovery_pct": 0.0,
                "below_right_eye_close": False, "ok": False,
            }

        low_idx = best_seg_idx
        low_close = float(df["close"].iloc[low_idx])
        low_low = float(df["low"].iloc[low_idx])

        # Step 2.5: 回踩低点收盘价必须低于右眼收盘价
        below_right_eye_close = low_close < right_eye_close

        # Step 3: 量能确认（缩量 = 芝麻点）
        vol_ma_val = float(vol_ma.iloc[low_idx]) if low_idx < len(vol_ma) else 0.0
        if vol_ma_val > 0:
            vol_ratio = float(df["volume"].iloc[low_idx]) / vol_ma_val
        else:
            vol_ratio = 0.0

        vol_ok = vol_ratio < RIGHT_CLAW_VOL_RATIO

        # Step 4: 悬空确认（收盘价在 MA60 上方）
        ma60_val = float(ma60.iloc[low_idx]) if low_idx < len(ma60) else 0.0
        above_ma60 = low_close > ma60_val if ma60_val > 0 else False

        # 左眼到右眼之间的最低收盘价，代表支撑位
        segment_low = df["close"].iloc[left_eye_pos:right_eye_pos].min()
        above_left_eye_low = low_close > float(segment_low) if pd.notna(segment_low) else False

        # Step 5: 回升确认
        current_close = float(df["close"].iloc[-1])
        if low_close > 0:
            recovery_pct = (current_close - low_close) / low_close * 100.0
        else:
            recovery_pct = 0.0

        recovery_ok = recovery_pct >= RIGHT_CLAW_RECOVERY_PCT and current_close > low_close

        ok = vol_ok and above_ma60 and above_left_eye_low and recovery_ok and below_right_eye_close

        return {
            "found": True,
            "low_position": low_idx,
            "low_price": low_low,
            "close_price": round(low_close, 2),
            "volume_ratio": round(vol_ratio, 2),
            "above_ma60": above_ma60,
            "recovery_pct": round(recovery_pct, 2),
            "below_right_eye_close": below_right_eye_close,
            "ok": ok,
        }

    def _calc_neckline_price(
        self,
        left_eye_pos: int,
        left_eye_price: float,
        right_eye_pos: int,
        right_eye_price: float,
        current_idx: int,
    ) -> Tuple[float, float]:
        """
        计算颈线价格（左右眼高点连线在当前时间的外推值）。

        Returns:
            (neckline_price, slope)
        """
        if right_eye_pos == left_eye_pos:
            return left_eye_price, 0.0

        slope = (right_eye_price - left_eye_price) / (right_eye_pos - left_eye_pos)
        neckline_price = left_eye_price + slope * (current_idx - left_eye_pos)
        return neckline_price, slope

    def _check_buy_point_1(
        self,
        df: pd.DataFrame,
        ma60: pd.Series,
    ) -> Tuple[bool, float]:
        """
        判断买点1：右爪缩量低吸。

        当前价格在MA60上方且距离 < 5%，成交量 < 20日均量0.8倍。
        """
        current_close = float(df["close"].iloc[-1])
        ma60_val = float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else 0.0

        if ma60_val <= 0:
            return False, 0.0

        price_to_ma60_pct = (current_close - ma60_val) / ma60_val * 100.0

        vol_ma = self._calc_vol_ma(df)
        current_vol = float(df["volume"].iloc[-1])
        vol_ma_val = float(vol_ma.iloc[-1]) if pd.notna(vol_ma.iloc[-1]) and vol_ma.iloc[-1] > 0 else 0.0

        if vol_ma_val <= 0:
            return False, round(price_to_ma60_pct, 2)

        vol_ratio = current_vol / vol_ma_val

        triggered = (
            price_to_ma60_pct > 0
            and price_to_ma60_pct < BUY_POINT_1_MA_DIST_PCT
            and vol_ratio < RIGHT_CLAW_VOL_RATIO
        )
        return triggered, round(price_to_ma60_pct, 2)

    def _check_buy_point_2(
        self,
        df: pd.DataFrame,
        neckline_price: float,
    ) -> Tuple[bool, bool, bool]:
        """
        判断买点2：放量突破颈线。

        当前收盘价 > 颈线价格，成交量 > 20日均量1.0倍。
        """
        current_close = float(df["close"].iloc[-1])
        price_above_neckline = current_close > neckline_price

        vol_ma = self._calc_vol_ma(df)
        current_vol = float(df["volume"].iloc[-1])
        vol_ma_val = float(vol_ma.iloc[-1]) if pd.notna(vol_ma.iloc[-1]) and vol_ma.iloc[-1] > 0 else 0.0

        if vol_ma_val <= 0:
            volume_ok = False
        else:
            vol_ratio = current_vol / vol_ma_val
            volume_ok = vol_ratio > BUY_POINT_2_VOL_RATIO

        triggered = price_above_neckline and volume_ok
        return triggered, price_above_neckline, volume_ok

    # ==================================================================
    # 核心方法：select
    # ==================================================================

    def select(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        daily_data: Optional[pd.DataFrame] = None,
        precomputed_metrics: Optional[Dict[str, Any]] = None,
    ) -> StrategyMatch:
        """
        执行金蛤蟆形态选股策略（6条铁律版本）。

        Args:
            stock_code: 股票代码
            stock_name: 股票名称（可选）
            daily_data: 预加载的日线数据（可选）
            precomputed_metrics: 预计算指标（可选）

        Returns:
            StrategyMatch 结果
        """
        match_details: Dict[str, Any] = {}
        conditions_met: List[str] = []
        conditions_failed: List[str] = []
        iron_rules_pass: List[str] = []
        iron_rules_fail: List[str] = []
        essential_score = 0.0
        bonus_score = 0.0
        bp1_triggered = False
        bp2_triggered = False

        try:
            # ----------------------------------------------------------
            # 获取日线数据
            # ----------------------------------------------------------
            data_source = "unknown"
            if daily_data is None or not isinstance(daily_data, pd.DataFrame) or daily_data.empty:
                if self._data_provider:
                    daily_data_result = self._data_provider.get_daily_data(stock_code, days=150)
                    if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                        daily_data = daily_data_result[0]  # type: ignore[assignment]
                        data_source = daily_data_result[1]
                    else:
                        daily_data = daily_data_result  # type: ignore[assignment]
                else:
                    conditions_failed.append("无数据提供者")
                    match_details["conditions_met"] = conditions_met
                    match_details["conditions_failed"] = conditions_failed
                    return self.create_strategy_match(
                        raw_score=0.0, matched=False,
                        reason="无数据提供者，无法获取日线数据",
                        match_details=match_details,
                    )
            else:
                data_source = "precomputed"

            match_details["data_source"] = data_source

            if daily_data is None or not isinstance(daily_data, pd.DataFrame) or daily_data.empty:
                conditions_failed.append("无日线数据")
                match_details["conditions_met"] = conditions_met
                match_details["conditions_failed"] = conditions_failed
                return self.create_strategy_match(
                    raw_score=0.0, matched=False, reason="无日线数据",
                    match_details=match_details,
                )

            daily_data = daily_data.copy()
            if "date" in daily_data.columns:
                daily_data = daily_data.sort_values("date", ascending=True).reset_index(drop=True)

            n = len(daily_data)
            match_details["data_length"] = n

            # ----------------------------------------------------------
            # Step 1: 前置条件检查
            # ----------------------------------------------------------
            if n < MIN_DATA_LENGTH:
                conditions_failed.append(f"数据不足（{n}条 < {MIN_DATA_LENGTH}条）")
                match_details["conditions_met"] = conditions_met
                match_details["conditions_failed"] = conditions_failed
                return self.create_strategy_match(
                    raw_score=0.0, matched=False,
                    reason=f"数据不足（{n}条 < {MIN_DATA_LENGTH}条）",
                    match_details=match_details,
                )

            # ----------------------------------------------------------
            # Step 2: 60日均线趋势
            # ----------------------------------------------------------
            ma60 = self._calc_ma(daily_data["close"], 60)  # type: ignore[arg-type]
            trend_ok, slope_pct, ma60_latest = self._calc_ma60_trend(daily_data)

            match_details["ma60_trend"] = {
                "slope_pct": round(slope_pct, 2),
                "ma60_latest": round(ma60_latest, 2),
                "ok": trend_ok,
            }

            if trend_ok:
                conditions_met.append(f"60日均线走平/向上（斜率 {slope_pct:.2f}%）")
                essential_score += self.SCORE_MA60_TREND
            else:
                conditions_failed.append(f"60日均线趋势向下（斜率 {slope_pct:.2f}%）")

            # ----------------------------------------------------------
            # Step 3: 识别左眼
            # ----------------------------------------------------------
            left_eye = self._detect_left_eye(daily_data, ma60)
            match_details["left_eye"] = left_eye

            if left_eye["ok"]:
                conditions_met.append(
                    f"左眼形态完整（价格 {left_eye['price']:.2f}，量比 {left_eye['volume_ratio']:.2f}）"
                )
                essential_score += self.SCORE_LEFT_EYE
            else:
                if left_eye["found"]:
                    conditions_failed.append(
                        f"左眼量能不足（量比 {left_eye['volume_ratio']:.2f} <= {LEFT_EYE_VOL_RATIO}）"
                    )
                else:
                    conditions_failed.append("未找到左眼形态")

            # ----------------------------------------------------------
            # Step 3.5: 识别左爪（铁律3: 左爪缩量）
            # ----------------------------------------------------------
            left_claw: Dict[str, Any] = {
                "found": False, "position": -1, "close_price": 0.0,
                "low_price": 0.0, "volume_ratio": 0.0,
                "ma60_dist_pct": 0.0, "ok": False,
            }

            if left_eye["found"]:
                left_claw = self._detect_left_claw(
                    daily_data, left_eye["uptrend_start"], ma60
                )
                match_details["left_claw"] = left_claw

                if left_claw["ok"]:
                    conditions_met.append(
                        f"左爪形态完整（价格 {left_claw['close_price']:.2f}，"
                        f"缩量 {left_claw['volume_ratio']:.2f}，"
                        f"距MA60 {left_claw['ma60_dist_pct']:.1f}%）"
                    )
                    essential_score += self.SCORE_LEFT_CLAW
                else:
                    if left_claw["found"]:
                        if left_claw.get("volume_ratio", 1.0) >= LEFT_CLAW_VOL_RATIO:
                            conditions_failed.append(
                                f"左爪未缩量（量比 {left_claw['volume_ratio']:.2f} >= {LEFT_CLAW_VOL_RATIO}）"
                            )
                        else:
                            conditions_failed.append(
                                f"左爪距MA60过远（{left_claw['ma60_dist_pct']:.1f}% > {LEFT_CLAW_MA_DIST_PCT}%）"
                            )
                    else:
                        conditions_failed.append("未找到左爪形态")
            else:
                match_details["left_claw"] = left_claw

            # ----------------------------------------------------------
            # Step 4: 识别右眼
            # ----------------------------------------------------------
            right_eye: Dict[str, Any] = {
                "found": False, "position": -1, "price": 0.0,
                "volume_ratio": 0.0, "height_ratio": 0.0,
                "upper_shadow_ratio": 0.0, "ok": False,
            }
            neckline_price = 0.0
            neckline_slope = 0.0

            if left_eye["found"]:
                right_eye = self._detect_right_eye(
                    daily_data, left_eye["position"], left_eye["price"], ma60
                )
                match_details["right_eye"] = right_eye

                if right_eye["ok"]:
                    conditions_met.append(
                        f"右眼形态完整（价格 {right_eye['price']:.2f}，"
                        f"高度比 {right_eye['height_ratio']:.2f}，"
                        f"量比 {right_eye['volume_ratio']:.2f}）"
                    )
                    essential_score += self.SCORE_RIGHT_EYE
                else:
                    if right_eye["found"]:
                        if right_eye.get("upper_shadow_ratio", 0) > RIGHT_EYE_UPPER_SHADOW_MAX:
                            conditions_failed.append(
                                f"右眼上影线过长（{right_eye['upper_shadow_ratio']:.2f} "
                                f"> {RIGHT_EYE_UPPER_SHADOW_MAX}），冲高回落不坚实"
                            )
                        elif right_eye["height_ratio"] < RIGHT_EYE_HEIGHT_RATIO:
                            conditions_failed.append(
                                f"右眼高度不足（{right_eye['height_ratio']:.2f} < {RIGHT_EYE_HEIGHT_RATIO}）"
                            )
                        else:
                            conditions_failed.append(
                                f"右眼量能不足（量比 {right_eye['volume_ratio']:.2f} <= {RIGHT_EYE_VOL_RATIO}）"
                            )
                    else:
                        conditions_failed.append("未找到右眼形态")

                if right_eye["found"]:
                    neckline_price, neckline_slope = self._calc_neckline_price(
                        left_eye["position"], left_eye["price"],
                        right_eye["position"], right_eye["price"],
                        n - 1,
                    )
            else:
                match_details["right_eye"] = right_eye
                conditions_failed.append("左眼未识别，跳过右眼检测")

            match_details["neckline"] = {
                "current_price": round(neckline_price, 2),
                "slope": round(neckline_slope, 4),
            }

            # ----------------------------------------------------------
            # Step 4.5: 验证关键点位在 MA60 上方
            # ----------------------------------------------------------
            if left_eye["found"] and right_eye["found"]:
                left_eye_ma60_val = float(ma60.iloc[left_eye["position"]])
                right_eye_ma60_val = float(ma60.iloc[right_eye["position"]])
                left_eye_close_val = float(daily_data["close"].iloc[left_eye["position"]])
                right_eye_close_val = float(daily_data["close"].iloc[right_eye["position"]])

                eyes_above_ma60 = (
                    left_eye_close_val > left_eye_ma60_val
                    and right_eye_close_val > right_eye_ma60_val
                )

                eye_to_eye_closes = daily_data["close"].iloc[left_eye["position"] : right_eye["position"] + 1]
                eye_to_eye_ma60 = ma60.iloc[left_eye["position"] : right_eye["position"] + 1]
                above_count = int((eye_to_eye_closes > eye_to_eye_ma60).sum())
                total_count = len(eye_to_eye_closes)
                above_ratio = above_count / total_count if total_count > 0 else 0.0
                support_ok = above_ratio >= MA60_ABOVE_RATIO_MIN

                if not eyes_above_ma60 or not support_ok:
                    if trend_ok:
                        essential_score -= self.SCORE_MA60_TREND
                        trend_ok = False
                    if left_eye_close_val <= left_eye_ma60_val:
                        conditions_failed.append(
                            f"左眼在MA60下方（C={left_eye_close_val:.2f} <= MA60={left_eye_ma60_val:.2f}）"
                        )
                    if right_eye_close_val <= right_eye_ma60_val:
                        conditions_failed.append(
                            f"右眼在MA60下方（C={right_eye_close_val:.2f} <= MA60={right_eye_ma60_val:.2f}）"
                        )
                    if not support_ok and eyes_above_ma60:
                        conditions_failed.append(
                            f"左眼到右眼之间MA60支撑不足（{above_count}/{total_count}天在MA60上方，"
                            f"比例 {above_ratio:.1%} < {MA60_ABOVE_RATIO_MIN:.0%}）"
                        )

            # ----------------------------------------------------------
            # Step 5: 识别右爪
            # ----------------------------------------------------------
            right_claw: Dict[str, Any] = {
                "found": False, "low_position": -1, "low_price": 0.0,
                "close_price": 0.0, "volume_ratio": 0.0,
                "above_ma60": False, "recovery_pct": 0.0,
                "below_right_eye_close": False, "ok": False,
            }

            if right_eye["found"]:
                right_eye_close = daily_data["close"].iloc[right_eye["position"]]
                right_claw = self._detect_right_claw(
                    daily_data,
                    right_eye["position"],
                    float(right_eye_close),
                    left_eye["position"],
                    ma60,
                )
                match_details["right_claw"] = right_claw

                if right_claw["ok"]:
                    conditions_met.append(
                        f"右爪形态完整（收盘低点 {right_claw['close_price']:.2f}，"
                        f"缩量 {right_claw['volume_ratio']:.2f}，"
                        f"回升 {right_claw['recovery_pct']:.1f}%）"
                    )
                    essential_score += self.SCORE_RIGHT_CLAW
                else:
                    if right_claw["found"]:
                        if not right_claw.get("below_right_eye_close", False):
                            conditions_failed.append(
                                f"右爪回踩未低于右眼收盘价（{right_claw['close_price']:.2f} "
                                f">= {daily_data['close'].iloc[right_eye['position']]:.2f}）"
                            )
                        elif not right_claw["above_ma60"]:
                            conditions_failed.append("右爪跌破60日线（未悬空）")
                        elif right_claw.get("recovery_pct", 0) < RIGHT_CLAW_RECOVERY_PCT:
                            conditions_failed.append(
                                f"右爪回踩后未充分回升"
                                f"（回升 {right_claw['recovery_pct']:.1f}% < {RIGHT_CLAW_RECOVERY_PCT}%）"
                            )
                        else:
                            conditions_failed.append(
                                f"右爪未缩量（量比 {right_claw['volume_ratio']:.2f} >= {RIGHT_CLAW_VOL_RATIO}）"
                            )
                    else:
                        conditions_failed.append("未找到右爪（收盘价无连续下跌波段）")
            else:
                match_details["right_claw"] = right_claw
                conditions_failed.append("右眼未识别，跳过右爪检测")

            # ----------------------------------------------------------
            # Step 6: 6条铁律硬约束检查
            # ----------------------------------------------------------
            # 铁律1: 右眼必须高于左眼
            if right_eye["found"] and left_eye["found"]:
                if right_eye["price"] > left_eye["price"]:
                    iron_rules_pass.append(
                        f"铁律1通过：右眼({right_eye['price']:.2f}) > 左眼({left_eye['price']:.2f})"
                    )
                else:
                    iron_rules_fail.append(
                        f"铁律1违反：右眼({right_eye['price']:.2f}) <= 左眼({left_eye['price']:.2f})，排除"
                    )

            # 铁律2: 右爪必须高于左爪（低点逐步抬高）
            if right_claw["found"] and left_claw["found"]:
                if right_claw["close_price"] > left_claw["close_price"]:
                    iron_rules_pass.append(
                        f"铁律2通过：右爪({right_claw['close_price']:.2f}) > 左爪({left_claw['close_price']:.2f})"
                    )
                else:
                    iron_rules_fail.append(
                        f"铁律2违反：右爪({right_claw['close_price']:.2f}) <= 左爪({left_claw['close_price']:.2f})，排除"
                    )

            # 铁律5: 右爪悬空在MA60上方
            if right_claw["found"]:
                if right_claw["above_ma60"]:
                    iron_rules_pass.append(f"铁律5通过：右爪悬空在MA60上方")
                else:
                    iron_rules_fail.append(f"铁律5违反：右爪未悬空（跌破MA60），排除")

            # 铁律7: 右眼上影线不能过长（排除冲高回落的假右眼）
            if right_eye["found"]:
                upper_shadow = right_eye.get("upper_shadow_ratio", 0)
                if upper_shadow <= RIGHT_EYE_UPPER_SHADOW_MAX:
                    iron_rules_pass.append(
                        f"铁律7通过：右眼上影线({upper_shadow:.1%}) <= {RIGHT_EYE_UPPER_SHADOW_MAX:.0%}"
                    )
                else:
                    iron_rules_fail.append(
                        f"铁律7违反：右眼上影线过长({upper_shadow:.1%} > {RIGHT_EYE_UPPER_SHADOW_MAX:.0%})，冲高回落假突破，排除"
                    )

            # 铁律8: 左眼必须放量（无量不构成左眼）
            if left_eye["found"]:
                left_vol = left_eye.get("volume_ratio", 0)
                if left_vol > LEFT_EYE_VOL_RATIO:
                    iron_rules_pass.append(
                        f"铁律8通过：左眼放量(量比{left_vol:.2f} > {LEFT_EYE_VOL_RATIO})"
                    )
                else:
                    iron_rules_fail.append(
                        f"铁律8违反：左眼量能不足(量比{left_vol:.2f} <= {LEFT_EYE_VOL_RATIO})，排除"
                    )

            # 铁律9: 右眼必须放量（无量不构成右眼）
            if right_eye["found"]:
                right_vol = right_eye.get("volume_ratio", 0)
                if right_vol > RIGHT_EYE_VOL_RATIO:
                    iron_rules_pass.append(
                        f"铁律9通过：右眼放量(量比{right_vol:.2f} > {RIGHT_EYE_VOL_RATIO})"
                    )
                else:
                    iron_rules_fail.append(
                        f"铁律9违反：右眼量能不足(量比{right_vol:.2f} <= {RIGHT_EYE_VOL_RATIO})，排除"
                    )

            # ----------------------------------------------------------
            # Step 6.5: RSI 检查（铁律6: 排除顶背离）
            # ----------------------------------------------------------
            rsi_match_ok = False
            if left_eye["found"] and right_eye["found"]:
                rsi_series = self._calc_rsi(daily_data["close"])  # type: ignore[arg-type]
                left_rsi = float(rsi_series.iloc[left_eye["position"]])
                right_rsi = float(rsi_series.iloc[right_eye["position"]])

                if left_rsi > 0:
                    rsi_diff = abs(right_rsi - left_rsi) / left_rsi
                else:
                    rsi_diff = 1.0

                rsi_match_ok = rsi_diff <= EYE_RSI_MAX_DIFF

                match_details["rsi_check"] = {
                    "left_rsi": round(left_rsi, 1),
                    "right_rsi": round(right_rsi, 1),
                    "diff_ratio": round(rsi_diff, 2),
                    "ok": rsi_match_ok,
                }

                if rsi_match_ok:
                    conditions_met.append(
                        f"RSI一致（左眼 {left_rsi:.1f}，右眼 {right_rsi:.1f}，差异 {rsi_diff:.1%}）"
                    )
                    essential_score += self.SCORE_RSI_MATCH
                else:
                    conditions_failed.append(
                        f"RSI背离（左眼 {left_rsi:.1f}，右眼 {right_rsi:.1f}，"
                        f"差异 {rsi_diff:.1%} > {EYE_RSI_MAX_DIFF:.0%}），疑似顶背离"
                    )

            # ----------------------------------------------------------
            # Step 6.6: 左腿长/右腿短检查（铁律4）
            # ----------------------------------------------------------
            leg_check_ok = False
            if left_claw["found"] and right_claw["found"] and left_eye["found"] and right_eye["found"]:
                left_leg_days = left_eye["position"] - left_claw["position"]
                right_leg_days = right_claw["low_position"] - right_eye["position"]

                match_details["leg_check"] = {
                    "left_leg_days": left_leg_days,
                    "right_leg_days": right_leg_days,
                }

                if right_leg_days > 0:
                    leg_ratio = left_leg_days / right_leg_days
                else:
                    leg_ratio = 0.0

                leg_check_ok = leg_ratio >= LEFT_LEG_RIGHT_LEG_MIN_RATIO

                if leg_check_ok:
                    iron_rules_pass.append(
                        f"铁律4通过：左腿({left_leg_days}天) >= 右腿({right_leg_days}天)，"
                        f"吸筹时间长于洗盘"
                    )
                else:
                    iron_rules_fail.append(
                        f"铁律4违反：左腿({left_leg_days}天) < 右腿({right_leg_days}天)，"
                        f"吸筹时间短于洗盘"
                    )

            # ----------------------------------------------------------
            # Step 7: 买点判断（加分项）
            # ----------------------------------------------------------
            bp1_triggered, bp1_ma_dist = self._check_buy_point_1(daily_data, ma60)
            match_details["buy_point_1"] = {
                "triggered": bp1_triggered,
                "price_to_ma60_pct": bp1_ma_dist,
            }
            if bp1_triggered:
                conditions_met.append(f"买点1触发：右爪缩量低吸（距MA60 {bp1_ma_dist:.2f}%）")
                bonus_score += self.SCORE_BUY_POINT_1

            bp2_triggered, bp2_above, bp2_vol = self._check_buy_point_2(daily_data, neckline_price)
            match_details["buy_point_2"] = {
                "triggered": bp2_triggered,
                "price_above_neckline": bp2_above,
                "volume_ok": bp2_vol,
            }
            if bp2_triggered:
                conditions_met.append(f"买点2触发：放量突破颈线（颈线 {neckline_price:.2f}）")
                bonus_score += self.SCORE_BUY_POINT_2

            # 短期均线多头
            ma5 = self._calc_ma(daily_data["close"], 5)  # type: ignore[arg-type]
            ma10 = self._calc_ma(daily_data["close"], 10)  # type: ignore[arg-type]
            ma20 = self._calc_ma(daily_data["close"], 20)  # type: ignore[arg-type]
            ma_bullish = (
                float(ma5.iloc[-1]) > float(ma10.iloc[-1]) > float(ma20.iloc[-1])
                if pd.notna(ma5.iloc[-1]) and pd.notna(ma10.iloc[-1]) and pd.notna(ma20.iloc[-1])
                else False
            )
            if ma_bullish:
                bonus_score += self.SCORE_MA_BULLISH
                conditions_met.append("短期均线多头排列（MA5 > MA10 > MA20）")

            match_details["bonus"] = {
                "ma_bullish": ma_bullish,
            }

        except Exception as e:
            logger.warning(f"金蛤蟆策略执行异常 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        # ==============================================================
        # 评分汇总
        # ==============================================================
        essential_score = min(essential_score, self.SCORE_ESSENTIAL_MAX)
        bonus_score = min(bonus_score, self.SCORE_BONUS_MAX)
        total_raw = essential_score + bonus_score

        # 匹配条件：
        # 1. 必要条件得分达标
        # 2. 右爪形态完整
        # 3. 至少一个买点触发
        # 4. 6条铁律全部通过（无违反）
        right_claw_ok = right_claw.get("ok", False)
        has_buy_point = bp1_triggered or bp2_triggered
        iron_rules_all_pass = len(iron_rules_fail) == 0

        matched = (
            essential_score >= ESSENTIAL_THRESHOLD
            and right_claw_ok
            and has_buy_point
            and iron_rules_all_pass
        )

        match_details["essential_score"] = essential_score
        match_details["bonus_score"] = bonus_score
        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed
        match_details["iron_rules_pass"] = iron_rules_pass
        match_details["iron_rules_fail"] = iron_rules_fail

        # 为了兼容旧的测试代码，保留 toad_leg 别名
        match_details["toad_leg"] = right_claw

        if matched:
            reason = (
                f"必要条件 {essential_score:.0f}/{self.SCORE_ESSENTIAL_MAX:.0f}，"
                f"加分 {bonus_score:.0f}/{self.SCORE_BONUS_MAX:.0f}，"
                f"铁律全部通过：" + "; ".join(conditions_met)
            )
        else:
            fail_reasons = list(conditions_failed)
            if iron_rules_fail:
                fail_reasons = list(iron_rules_fail) + fail_reasons
            reason = (
                f"必要条件 {essential_score:.0f}/{self.SCORE_ESSENTIAL_MAX:.0f} "
                f"(需 >= {ESSENTIAL_THRESHOLD})：" + "; ".join(fail_reasons[:5])
            )

        return self.create_strategy_match(
            raw_score=total_raw,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )