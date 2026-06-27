# -*- coding: utf-8 -*-
"""
金蛤蟆形态选股策略 - Python 实现。

识别日K线金蛤蟆形态：
  60日均线走平向上 → 左眼放量 → 右眼放量 → 缩量回踩 → 等待突破颈线

两个买点：
  - 买点1（缩量回踩）：右眼后缩量回踩60日线附近，左侧低吸
  - 买点2（放量突破）：放量突破左右眼连线颈线，右侧追涨

评分体系：
  - 必要条件（80分）：MA60趋势、左眼、右眼、蛤蟆腿。至少满足50分才匹配
  - 加分项（30分）：买点触发、均线多头、右眼高于左眼
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
TOAD_LEG_VOL_RATIO = 0.8  # 蛤蟆腿缩量阈值（< 前20日均量的倍数）
MIN_EYE_DISTANCE = 15  # 左右眼最小间隔（交易日）
RIGHT_EYE_HEIGHT_RATIO = 0.90  # 右眼不低于左眼的比例
MA60_TREND_SLOPE_PCT = -1.0  # MA60趋势斜率阈值（%）
MA60_SLOPE_LOOKBACK = 20  # MA60斜率计算回看天数
VOL_MA_PERIOD = 20  # 均量计算周期
ESSENTIAL_THRESHOLD = 50  # 必要条件匹配阈值
BUY_POINT_1_MA_DIST_PCT = 5.0  # 买点1：价格距MA60的最大距离（%）
BUY_POINT_2_VOL_RATIO = 1.0  # 买点2：量能阈值


@register_strategy
class GoldenToadStrategy(StockSelectorStrategy):
    """
    金蛤蟆形态选股策略。

    识别日K线金蛤蟆形态，判断两个买点：
    1. 缩量回踩买点（左侧低吸）
    2. 放量突破颈线买点（右侧追涨）
    """

    # ------------------------------------------------------------------
    # 评分常量
    # ------------------------------------------------------------------
    SCORE_MA60_TREND = 15  # 必要条件：60日均线趋势
    SCORE_LEFT_EYE = 20  # 必要条件：左眼形态
    SCORE_RIGHT_EYE = 25  # 必要条件：右眼形态
    SCORE_TOAD_LEG = 20  # 必要条件：蛤蟆腿回踩
    SCORE_ESSENTIAL_MAX = 80  # 必要条件满分

    SCORE_BUY_POINT_1 = 10  # 加分项：买点1触发
    SCORE_BUY_POINT_2 = 10  # 加分项：买点2触发
    SCORE_MA_BULLISH = 5  # 加分项：短期均线多头
    SCORE_RIGHT_EYE_HIGHER = 5  # 加分项：右眼高于左眼
    SCORE_BONUS_MAX = 30  # 加分项满分

    def __init__(self) -> None:
        metadata = StrategyMetadata(
            id="golden_toad",
            name="golden_toad",
            display_name="金蛤蟆形态",
            description=(
                "识别日K线金蛤蟆形态：60日均线走平向上→左眼放量→右眼放量→"
                "缩量回踩→等待突破颈线。两个买点：缩量回踩低吸/放量突破追涨。"
            ),
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
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

        Args:
            high_series: 最高价序列
            window: 前后窗口大小

        Returns:
            局部高点在序列中的索引列表（按索引升序）
        """
        n = len(high_series)
        if n < 2 * window + 1:
            return []

        # 滚动窗口最大值
        roll_max = high_series.rolling(window=2 * window + 1, center=True, min_periods=1).max()

        highs: List[int] = []
        for i in range(window, n - window):
            if high_series.iloc[i] == roll_max.iloc[i]:
                # 确保是窗口内唯一最大值
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

        # 取近 MA60_SLOPE_LOOKBACK 个 MA60 值计算斜率
        recent_ma60 = ma60.iloc[-MA60_SLOPE_LOOKBACK:].dropna()
        if len(recent_ma60) < 5:
            return False, 0.0, ma60_latest

        slope = self._linear_slope(recent_ma60.values)
        # 斜率百分比 = 斜率 / 均值 × 100
        mean_ma60 = float(recent_ma60.mean())
        if mean_ma60 == 0:
            slope_pct = 0.0
        else:
            slope_pct = (slope / mean_ma60) * 100.0

        trend_ok = slope_pct >= MA60_TREND_SLOPE_PCT
        return trend_ok, slope_pct, ma60_latest

    def _detect_left_eye(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        识别左眼（第一个放量高点）。

        Returns:
            {
                "found": bool, "position": int, "price": float,
                "volume_ratio": float, "ok": bool
            }
        """
        n = len(df)
        start_idx = int(n * LEFT_EYE_SEARCH_START)
        end_idx = int(n * LEFT_EYE_SEARCH_END)

        if end_idx - start_idx < 2 * LOCAL_HIGH_WINDOW + 1:
            return {"found": False, "position": -1, "price": 0.0, "volume_ratio": 0.0, "ok": False}

        vol_ma = self._calc_vol_ma(df)

        # 在搜索区间内找局部高点
        local_highs = self._find_local_highs(df["high"], LOCAL_HIGH_WINDOW)  # type: ignore[arg-type]  # type: ignore[arg-type]
        candidates = [i for i in local_highs if start_idx <= i <= end_idx]

        if not candidates:
            return {"found": False, "position": -1, "price": 0.0, "volume_ratio": 0.0, "ok": False}

        # 选价格最高的那个作为左眼
        best_idx = max(candidates, key=lambda i: df["high"].iloc[i])
        best_price = float(df["high"].iloc[best_idx])

        # 量能确认
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
            "ok": ok,
        }

    def _detect_right_eye(self, df: pd.DataFrame, left_eye_pos: int, left_eye_price: float) -> Dict[str, Any]:
        """
        识别右眼（第二个放量高点，在左眼之后）。

        Returns:
            {
                "found": bool, "position": int, "price": float,
                "volume_ratio": float, "height_ratio": float, "ok": bool
            }
        """
        n = len(df)
        search_start = left_eye_pos + MIN_EYE_DISTANCE
        search_end = n - 20  # 末尾留20天

        if search_start >= search_end or search_start >= n:
            return {
                "found": False,
                "position": -1,
                "price": 0.0,
                "volume_ratio": 0.0,
                "height_ratio": 0.0,
                "ok": False,
            }

        vol_ma = self._calc_vol_ma(df)

        # 在搜索区间内找局部高点
        local_highs = self._find_local_highs(df["high"], LOCAL_HIGH_WINDOW)  # type: ignore[arg-type]
        candidates = [i for i in local_highs if search_start <= i <= search_end]

        if not candidates:
            return {
                "found": False,
                "position": -1,
                "price": 0.0,
                "volume_ratio": 0.0,
                "height_ratio": 0.0,
                "ok": False,
            }

        # 选价格最高且满足高度条件的
        best_idx = -1
        best_price = 0.0
        for i in candidates:
            price = float(df["high"].iloc[i])
            height_ratio = price / left_eye_price if left_eye_price > 0 else 0.0
            if height_ratio >= RIGHT_EYE_HEIGHT_RATIO and price > best_price:
                best_idx = i
                best_price = price

        if best_idx == -1:
            # 没有满足高度条件的，选价格最高的
            best_idx = max(candidates, key=lambda i: df["high"].iloc[i])
            best_price = float(df["high"].iloc[best_idx])

        height_ratio = best_price / left_eye_price if left_eye_price > 0 else 0.0

        # 量能确认
        vol_ma_val = float(vol_ma.iloc[best_idx]) if best_idx < len(vol_ma) else 0.0
        if vol_ma_val > 0:
            vol_ratio = float(df["volume"].iloc[best_idx]) / vol_ma_val
        else:
            vol_ratio = 0.0

        vol_ok = vol_ratio > RIGHT_EYE_VOL_RATIO
        height_ok = height_ratio >= RIGHT_EYE_HEIGHT_RATIO
        ok = vol_ok and height_ok

        return {
            "found": True,
            "position": best_idx,
            "price": best_price,
            "volume_ratio": round(vol_ratio, 2),
            "height_ratio": round(height_ratio, 2),
            "ok": ok,
        }

    def _detect_toad_leg(
        self,
        df: pd.DataFrame,
        right_eye_pos: int,
        left_eye_pos: int,
        ma60: pd.Series,
    ) -> Dict[str, Any]:
        """
        识别蛤蟆腿（右眼后的缩量回踩）。

        Returns:
            {
                "found": bool, "low_position": int, "low_price": float,
                "volume_ratio": float, "above_ma60": bool, "ok": bool
            }
        """
        n = len(df)
        search_start = right_eye_pos + 1
        search_end = n - 1

        if search_start >= search_end:
            return {
                "found": False,
                "low_position": -1,
                "low_price": 0.0,
                "volume_ratio": 0.0,
                "above_ma60": False,
                "ok": False,
            }

        vol_ma = self._calc_vol_ma(df)

        # 在右眼之后找最低点
        segment = df["low"].iloc[search_start:search_end]
        if len(segment) == 0:
            return {
                "found": False,
                "low_position": -1,
                "low_price": 0.0,
                "volume_ratio": 0.0,
                "above_ma60": False,
                "ok": False,
            }

        low_idx = search_start + int(segment.idxmin() - search_start)
        low_price = float(df["low"].iloc[low_idx])

        # 量能确认（缩量）
        vol_ma_val = float(vol_ma.iloc[low_idx]) if low_idx < len(vol_ma) else 0.0
        if vol_ma_val > 0:
            vol_ratio = float(df["volume"].iloc[low_idx]) / vol_ma_val
        else:
            vol_ratio = 0.0

        vol_ok = vol_ratio < TOAD_LEG_VOL_RATIO

        # 回踩低点 > MA60（获得支撑）
        ma60_val = float(ma60.iloc[low_idx]) if low_idx < len(ma60) else 0.0
        above_ma60 = low_price > ma60_val if ma60_val > 0 else False

        # 不跌破左眼后回踩低点区域（左眼到右眼之间的最低价，代表支撑位）
        segment_low = df["low"].iloc[left_eye_pos:right_eye_pos].min()
        above_left_eye_low = low_price > float(segment_low) if pd.notna(segment_low) else False

        ok = vol_ok and above_ma60 and above_left_eye_low

        return {
            "found": True,
            "low_position": low_idx,
            "low_price": low_price,
            "volume_ratio": round(vol_ratio, 2),
            "above_ma60": above_ma60,
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
        判断买点1：缩量回踩。

        当前价格在MA60上方且距离 < 5%，成交量 < 20日均量0.8倍。

        Returns:
            (triggered, price_to_ma60_pct)
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
            price_to_ma60_pct > 0 and price_to_ma60_pct < BUY_POINT_1_MA_DIST_PCT and vol_ratio < TOAD_LEG_VOL_RATIO
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

        Returns:
            (triggered, price_above_neckline, volume_ok)
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
        执行金蛤蟆形态选股策略。

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
                        raw_score=0.0,
                        matched=False,
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
                    raw_score=0.0,
                    matched=False,
                    reason="无日线数据",
                    match_details=match_details,
                )

            # 确保按日期排序
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
                    raw_score=0.0,
                    matched=False,
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
            left_eye = self._detect_left_eye(daily_data)
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
            # Step 4: 识别右眼
            # ----------------------------------------------------------
            right_eye: Dict[str, Any] = {
                "found": False,
                "position": -1,
                "price": 0.0,
                "volume_ratio": 0.0,
                "height_ratio": 0.0,
                "ok": False,
            }
            neckline_price = 0.0
            neckline_slope = 0.0

            if left_eye["found"]:
                right_eye = self._detect_right_eye(daily_data, left_eye["position"], left_eye["price"])
                match_details["right_eye"] = right_eye

                if right_eye["ok"]:
                    conditions_met.append(
                        f"右眼形态完整（价格 {right_eye['price']:.2f}，"
                        f"高度比 {right_eye['height_ratio']:.2f}，量比 {right_eye['volume_ratio']:.2f}）"
                    )
                    essential_score += self.SCORE_RIGHT_EYE
                else:
                    if right_eye["found"]:
                        if right_eye["height_ratio"] < RIGHT_EYE_HEIGHT_RATIO:
                            conditions_failed.append(
                                f"右眼高度不足（{right_eye['height_ratio']:.2f} < {RIGHT_EYE_HEIGHT_RATIO}）"
                            )
                        else:
                            conditions_failed.append(
                                f"右眼量能不足（量比 {right_eye['volume_ratio']:.2f} <= {RIGHT_EYE_VOL_RATIO}）"
                            )
                    else:
                        conditions_failed.append("未找到右眼形态")

                # 计算颈线
                if right_eye["found"]:
                    neckline_price, neckline_slope = self._calc_neckline_price(
                        left_eye["position"],
                        left_eye["price"],
                        right_eye["position"],
                        right_eye["price"],
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
            # Step 5: 识别蛤蟆腿
            # ----------------------------------------------------------
            toad_leg: Dict[str, Any] = {
                "found": False,
                "low_position": -1,
                "low_price": 0.0,
                "volume_ratio": 0.0,
                "above_ma60": False,
                "ok": False,
            }

            if right_eye["found"]:
                toad_leg = self._detect_toad_leg(
                    daily_data,
                    right_eye["position"],
                    left_eye["position"],
                    ma60,
                )
                match_details["toad_leg"] = toad_leg

                if toad_leg["ok"]:
                    conditions_met.append(
                        f"蛤蟆腿回踩确认（低点 {toad_leg['low_price']:.2f}，"
                        f"缩量 {toad_leg['volume_ratio']:.2f}，在MA60上方）"
                    )
                    essential_score += self.SCORE_TOAD_LEG
                else:
                    if toad_leg["found"]:
                        if not toad_leg["above_ma60"]:
                            conditions_failed.append("蛤蟆腿回踩跌破60日线")
                        else:
                            conditions_failed.append(
                                f"蛤蟆腿回踩未缩量（量比 {toad_leg['volume_ratio']:.2f} >= {TOAD_LEG_VOL_RATIO}）"
                            )
                    else:
                        conditions_failed.append("未找到蛤蟆腿回踩")
            else:
                match_details["toad_leg"] = toad_leg
                conditions_failed.append("右眼未识别，跳过蛤蟆腿检测")

            # ----------------------------------------------------------
            # Step 6: 买点判断（加分项）
            # ----------------------------------------------------------
            # 买点1：缩量回踩
            bp1_triggered, bp1_ma_dist = self._check_buy_point_1(daily_data, ma60)
            match_details["buy_point_1"] = {
                "triggered": bp1_triggered,
                "price_to_ma60_pct": bp1_ma_dist,
            }
            if bp1_triggered:
                conditions_met.append(f"买点1触发：缩量回踩（距MA60 {bp1_ma_dist:.2f}%）")
                bonus_score += self.SCORE_BUY_POINT_1

            # 买点2：放量突破颈线
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

            # 右眼高于左眼
            right_eye_higher = right_eye["found"] and right_eye["price"] > left_eye["price"]
            if right_eye_higher:
                bonus_score += self.SCORE_RIGHT_EYE_HIGHER
                conditions_met.append("右眼高于左眼（强势形态）")

            match_details["bonus"] = {
                "ma_bullish": ma_bullish,
                "right_eye_higher": right_eye_higher,
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

        # 匹配条件：必要条件得分达标 + 蛤蟆腿完整 + 至少一个买点触发
        toad_leg_ok = toad_leg.get("ok", False)
        has_buy_point = bp1_triggered or bp2_triggered
        matched = (
            essential_score >= ESSENTIAL_THRESHOLD
            and toad_leg_ok
            and has_buy_point
        )

        match_details["essential_score"] = essential_score
        match_details["bonus_score"] = bonus_score
        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed

        if matched:
            reason = (
                f"必要条件 {essential_score:.0f}/{self.SCORE_ESSENTIAL_MAX:.0f}，"
                f"加分 {bonus_score:.0f}/{self.SCORE_BONUS_MAX:.0f}：" + "; ".join(conditions_met)
            )
        else:
            reason = (
                f"必要条件 {essential_score:.0f}/{self.SCORE_ESSENTIAL_MAX:.0f} "
                f"(需 >= {ESSENTIAL_THRESHOLD})：" + "; ".join(conditions_failed)
            )

        return self.create_strategy_match(
            raw_score=total_raw,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
