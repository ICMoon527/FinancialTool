# -*- coding: utf-8 -*-
"""
支撑/压力位参考线生成器

基于历史日线OHLCV数据，生成五类共10条支撑/压力位参考线：
  - 主力操盘三线（攻击线/操盘线/防守线）
  - 均线（MA5/MA10/MA20）
  - 前高/前低（30日）
  - 筹码密集区上下沿（峰值半宽法）
  - 昨收

每条参考线包含：id, label, price, category, color, style, base_weight
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReferenceLineGenerator:
    """支撑/压力位参考线生成器"""

    def __init__(self, daily_data: pd.DataFrame):
        """
        Args:
            daily_data: 日线OHLCV数据，需包含 Open, High, Low, Close, Volume 列
                        索引为日期（datetime），按时间升序排列
        """
        self.daily_data = daily_data.copy()
        if len(self.daily_data) > 0:
            self.daily_data = self.daily_data.sort_index()

    # ---------- 工具函数 ----------

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _llv(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).min()

    @staticmethod
    def _hhv(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).max()

    # ---------- 主力操盘三线 ----------

    def _calc_main_trading_lines(self) -> List[Dict[str, Any]]:
        """计算主力操盘三线：攻击线、操盘线、防守线

        公式与 indicators/indicators/main_trading.py 完全一致：
        - 攻击线 = EMA(Close, 6)
        - 操盘线 = IF(EMA26 < B, B, EMA26)  其中 B = EMA26 + (EMA26 - REF(EMA26, 1))
        - 防守线 = 操盘线 - (攻击线 - 操盘线)
        """
        if len(self.daily_data) < 6:
            return []

        import numpy as np

        close = self.daily_data["Close"]

        ema6 = self._ema(close, 6)
        ema26 = self._ema(close, 26)

        ema26_diff = ema26 - ema26.shift(1)
        B = ema26 + ema26_diff

        trading_line_series = np.where(ema26 < B, B, ema26)
        trading_line_series = pd.Series(trading_line_series, index=self.daily_data.index)

        attack_line = float(ema6.iloc[-1])
        trade_line = float(trading_line_series.iloc[-1])
        defense_line = trade_line - (attack_line - trade_line)

        return [
            {
                "id": "attack_line",
                "label": "攻击",
                "price": round(attack_line, 2),
                "category": "main_trading",
                "color": "#FF4444",
                "style": "dashed",
                "base_weight": 1.0,
            },
            {
                "id": "trading_line",
                "label": "操盘",
                "price": round(trade_line, 2),
                "category": "main_trading",
                "color": "#FFAA00",
                "style": "dashed",
                "base_weight": 1.0,
            },
            {
                "id": "defense_line",
                "label": "防守",
                "price": round(defense_line, 2),
                "category": "main_trading",
                "color": "#44AA44",
                "style": "dashed",
                "base_weight": 1.5,
            },
        ]

    # ---------- 均线 ----------

    def _calc_ma_lines(self) -> List[Dict[str, Any]]:
        """计算均线 MA5/MA10/MA20"""
        if len(self.daily_data) < 5:
            return []

        close = self.daily_data["Close"]
        result = []

        ma_configs = [
            ("MA5", 5, "#FFFFFF", "dotted", 1.0),
            ("MA10", 10, "#FFAA00", "dotted", 1.0),
            ("MA20", 20, "#AA44FF", "dotted", 1.2),
        ]

        for label, period, color, style, weight in ma_configs:
            if len(self.daily_data) >= period:
                ma_val = float(self._sma(close, period).iloc[-1])
                result.append(
                    {
                        "id": f"ma_{period}",
                        "label": label,
                        "price": round(ma_val, 2),
                        "category": "moving_average",
                        "color": color,
                        "style": style,
                        "base_weight": weight,
                    }
                )

        return result

    # ---------- EXPMA ----------

    def _calc_expma_lines(self) -> List[Dict[str, Any]]:
        """计算EXPMA13参考线

        使用与 indicators/indicators/expma.py 一致的指数移动平均公式：
        EXPMA = Close.ewm(span=13, adjust=False).mean()
        """
        if len(self.daily_data) < 13:
            return []

        close = self.daily_data["Close"]
        expma13 = float(self._ema(close, 13).iloc[-1])

        return [
            {
                "id": "expma_13",
                "label": "EXPMA13",
                "price": round(expma13, 2),
                "category": "expma",
                "color": "#FFFFFF",
                "style": "solid",
                "base_weight": 1.0,
            },
        ]

    # ---------- 前高/前低 ----------

    def _calc_extreme_lines(self) -> List[Dict[str, Any]]:
        """计算前高/前低（30日）"""
        if len(self.daily_data) < 30:
            return []

        high = self.daily_data["High"]
        low = self.daily_data["Low"]

        prev_high = float(self._hhv(high, 30).iloc[-1])
        prev_low = float(self._llv(low, 30).iloc[-1])

        return [
            {
                "id": "previous_high_30",
                "label": "前高（30日）",
                "price": round(prev_high, 2),
                "category": "extreme_price",
                "color": "#FF4444",
                "style": "solid",
                "base_weight": 1.0,
            },
            {
                "id": "previous_low_30",
                "label": "前低（30日）",
                "price": round(prev_low, 2),
                "category": "extreme_price",
                "color": "#44AA44",
                "style": "solid",
                "base_weight": 1.0,
            },
        ]

    # ---------- 筹码密集区 ----------

    def _calc_chip_dense_zone(self) -> List[Dict[str, Any]]:
        """计算筹码密集区上下沿（峰值半宽法）

        复用 ChipDistribution 指标计算筹码概率密度，然后通过主峰半高处左右扫描确定密集区边界。
        """
        if len(self.daily_data) < 30:
            return []

        try:
            from indicators.indicators.chip_distribution import ChipDistribution

            chip_calc = ChipDistribution(enable_smooth=True, max_days=120)
            chip_result = chip_calc.calculate(self.daily_data)

            if chip_result["max_chip_price"] is None or chip_result["current_price"] is None:
                logger.warning("筹码分布计算无有效结果")
                return []

            chip_vols = np.array(chip_result["chip_volumes"])
            price_bins = np.array(chip_result["price_bins"])

            if len(chip_vols) == 0 or len(price_bins) == 0:
                return []

            max_density = float(chip_vols.max())
            if max_density <= 0:
                return []

            peak_idx = int(np.argmax(chip_vols))
            threshold = max_density * 0.5

            # 从主峰向左扫描
            lower_idx = peak_idx
            while lower_idx > 0 and float(chip_vols[lower_idx]) >= threshold:
                lower_idx -= 1
            lower_edge = float(price_bins[lower_idx])

            # 从主峰向右扫描
            upper_idx = peak_idx
            while upper_idx < len(price_bins) - 1 and float(chip_vols[upper_idx]) >= threshold:
                upper_idx += 1
            upper_edge = float(price_bins[upper_idx])

            # 若上下沿过近（单尖峰），退化
            current_price = float(chip_result["current_price"])
            if lower_edge >= upper_edge or (upper_edge - lower_edge) / current_price < 0.005:
                lower_edge = current_price * 0.99
                upper_edge = current_price * 1.01

            return [
                {
                    "id": "chip_dense_lower",
                    "label": "筹码密集区下沿",
                    "price": round(lower_edge, 2),
                    "category": "chip_dense",
                    "color": "#AA44FF",
                    "style": "dashed",
                    "base_weight": 1.5,
                },
                {
                    "id": "chip_dense_upper",
                    "label": "筹码密集区上沿",
                    "price": round(upper_edge, 2),
                    "category": "chip_dense",
                    "color": "#AA44FF",
                    "style": "dashed",
                    "base_weight": 1.5,
                },
            ]

        except ImportError:
            logger.warning("scipy 未安装，无法计算筹码分布")
            return []
        except Exception as e:
            logger.warning(f"筹码分布计算失败: {e}")
            return []

    # ---------- 昨收 ----------

    def _calc_prev_close(self) -> List[Dict[str, Any]]:
        """计算昨收"""
        if len(self.daily_data) < 2:
            return []

        prev_close = float(self.daily_data["Close"].iloc[-2])

        return [
            {
                "id": "prev_close",
                "label": "昨收",
                "price": round(prev_close, 2),
                "category": "prev_close",
                "color": "#FFCC00",
                "style": "solid",
                "base_weight": 1.0,
            }
        ]

    # ---------- 综合生成 ----------

    def generate_all(self) -> List[Dict[str, Any]]:
        """生成全部参考线列表

        Returns:
            参考线字典列表，每条包含: id, label, price, category, color, style, base_weight
        """
        if len(self.daily_data) < 5:
            logger.warning("日线数据不足（<5条），无法生成参考线")
            return []

        lines: List[Dict[str, Any]] = []

        lines.extend(self._calc_main_trading_lines())
        lines.extend(self._calc_ma_lines())
        lines.extend(self._calc_expma_lines())
        lines.extend(self._calc_extreme_lines())
        lines.extend(self._calc_chip_dense_zone())
        lines.extend(self._calc_prev_close())

        # 过滤掉价格为0或负值的异常结果
        lines = [line for line in lines if line["price"] > 0]

        logger.debug(f"生成参考线 {len(lines)} 条，分布于 {len(set(l['category'] for l in lines))} 个类别")
        return lines


# ============================================================
# 连续引力场模型：根据参考线调整信号置信度
# ============================================================


def apply_gravitational_field(
    current_price: float,
    reference_lines: List[Dict[str, Any]],
    signal_type: str,
    base_confidence: float,
    decay_sigma: float = 1.5,
    smooth_width: float = 0.20,
    scale_factor: float = 0.05,
    clamp_limit: float = 0.15,
) -> float:
    """
    对单个信号的置信度应用连续引力场修正（高斯衰减 + sigmoid 软穿越）

    核心逻辑：
    - 高斯衰减：每条参考线的影响力随距离平滑衰减，无硬截断
    - sigmoid 软穿越：支撑/压力角色平滑过渡，无二值翻转
    - 买入信号：下方支撑越多 → 置信度上调；上方压力越多 → 置信度下调
    - 卖出信号：上方压力越多 → 置信度上调；下方支撑越多 → 置信度下调

    Args:
        current_price: 当前价格
        reference_lines: 参考线列表
        signal_type: 'buy' 或 'sell'
        base_confidence: 基础置信度
        decay_sigma: 高斯衰减 σ（%），控制影响力随距离的衰减速度
        smooth_width: 穿越软化宽度（%）
        scale_factor: 缩放系数
        clamp_limit: 修正幅度上下限

    Returns:
        修正后的置信度
    """
    if current_price <= 0 or not reference_lines:
        return base_confidence

    support_force = 0.0
    pressure_force = 0.0

    for line in reference_lines:
        line_price = line.get("price", 0)
        base_weight = line.get("base_weight", 1.0)

        if line_price <= 0:
            continue

        rel_diff = (line_price - current_price) / current_price * 100
        abs_dist = abs(rel_diff)

        raw_influence = base_weight * math.exp(-(abs_dist ** 2) / (2 * decay_sigma ** 2))

        support_ratio = 1.0 / (1.0 + math.exp(-rel_diff / smooth_width))
        pressure_ratio = 1.0 - support_ratio

        support_force += raw_influence * support_ratio
        pressure_force += raw_influence * pressure_ratio

    if signal_type == "buy":
        adjustment = (support_force - pressure_force) * scale_factor
    elif signal_type == "sell":
        adjustment = (pressure_force - support_force) * scale_factor
    else:
        adjustment = 0.0

    adjustment = max(-clamp_limit, min(clamp_limit, adjustment))
    adjusted = base_confidence + adjustment
    adjusted = max(0.0, min(1.0, adjusted))

    return adjusted
