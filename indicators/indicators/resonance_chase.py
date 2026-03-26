import logging
import numpy as np
import pandas as pd

from ..base import BaseIndicator

logger = logging.getLogger(__name__)


class ResonanceChase(BaseIndicator):
    """
    共振追涨指标

    该指标通过中线和短线趋势的共振来识别追涨机会。

    公式：
    - AAA：(3*C+H+L+O)/6
    - VAR1：EMA(AAA, 35)
    - VAR2：(HHV(VAR1, 10) + HHV(VAR1, 30) + HHV(VAR1, 90)) / 3
    - VAR3：(LLV(VAR1, 10) + LLV(VAR1, 30) + LLV(VAR1, 90)) / 3
    - 牛线：(HHV(VAR2, 5) + HHV(VAR2, 15) + HHV(VAR2, 30)) / 3
    - 熊线：(LLV(VAR3, 5) + LLV(VAR3, 15) + LLV(VAR3, 30)) / 3

    说明：
    1. 零轴位置的红色条形图表示中线多头趋势
    2. 零轴位置的绿色条形图表示中线空头线趋势
    3. 黄线在零轴上方运行时表示短线趋势走强
    4. 黄线在零轴下方运行时表示短线趋势走弱
    5. 紫色柱表示短线趋势和中线趋势共振走强
    """

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最高高值"""
        return data.rolling(window=period, min_periods=1).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最低低值"""
        return data.rolling(window=period, min_periods=1).min()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算共振追涨指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了相关共振追涨列的DataFrame
        """
        self.validate_input(data)
        logger.info("[共振追涨指标] 数据验证通过，开始计算指标")

        result = data.copy()

        close = data["Close"]
        open_ = data["Open"]
        high = data["High"]
        low = data["Low"]

        AAA = (3 * close + high + low + open_) / 6

        VAR1 = self._ema(AAA, 35)

        VAR2 = (self._hhv(VAR1, 10) + self._hhv(VAR1, 30) + self._hhv(VAR1, 90)) / 3
        VAR3 = (self._llv(VAR1, 10) + self._llv(VAR1, 30) + self._llv(VAR1, 90)) / 3

        bull_line = (self._hhv(VAR2, 5) + self._hhv(VAR2, 15) + self._hhv(VAR2, 30)) / 3
        bear_line = (self._llv(VAR3, 5) + self._llv(VAR3, 15) + self._llv(VAR3, 30)) / 3

        ema_aaa_2 = self._ema(self._ema(AAA, 2), 2)
        mid_bullish = ema_aaa_2 > bear_line

        exp6 = self._ema(close, 6)
        exp18 = self._ema(close, 18)
        OUT1 = 500 * (exp6 - exp18) / exp18 + 2
        OUT2 = self._ema(OUT1, 3)

        resonance = OUT1 > 2

        result["bull_line"] = bull_line.round(2)
        result["bear_line"] = bear_line.round(2)
        result["mid_bullish"] = mid_bullish
        result["OUT1"] = OUT1.round(2)
        result["OUT2"] = OUT2.round(2)
        result["resonance"] = resonance

        return result
