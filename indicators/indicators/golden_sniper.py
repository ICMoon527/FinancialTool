import numpy as np
import pandas as pd

from ..base import BaseIndicator


class GoldenSniper(BaseIndicator):
    """
    黄金狙击指标

    该指标用于识别狙击黄金般的精准买入机会。

    公式：
    - 黄金线：EMA(C, 13)
    - 狙击线：EMA(C, 34)
    - 信号：黄金线与狙击线的交叉

    说明：
    1. 黄金线上穿狙击线，发出黄金狙击信号
    2. 结合成交量放大效果更佳
    """

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均"""
        return data.rolling(window=period, min_periods=1).mean()

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
        计算黄金狙击指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'golden_line', 'sniper_line', 'golden_sniper_signal'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        volume = data["Volume"]

        golden_line = self._ema(close, 13)
        sniper_line = self._ema(close, 34)

        volume_ma = self._sma(volume, 5)
        volume_surge = volume > volume_ma * 1.2

        golden_cross = (golden_line > sniper_line) & (golden_line.shift(1) <= sniper_line.shift(1))
        golden_sniper_signal = golden_cross & volume_surge

        result["golden_line"] = golden_line
        result["sniper_line"] = sniper_line
        result["volume_ma"] = volume_ma
        result["volume_surge"] = volume_surge
        result["golden_cross"] = golden_cross
        result["golden_sniper_signal"] = golden_sniper_signal

        return result
