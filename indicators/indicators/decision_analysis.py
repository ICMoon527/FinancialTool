import numpy as np
import pandas as pd

from ..base import BaseIndicator


class DecisionAnalysis(BaseIndicator):
    """
    决策解盘指标

    该指标结合MA和MACD进行趋势分析。

    公式：
    - MA5：MA(C, 5)
    - MA10：MA(C, 10)
    - MA20：MA(C, 20)
    - DIF：EMA(C, 12) - EMA(C, 26)
    - DEA：EMA(DIF, 9)
    - MACD：2 * (DIF - DEA)

    说明：
    结合移动平均线和MACD进行综合趋势判断。
    """

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均"""
        return data.rolling(window=period, min_periods=1).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算决策解盘指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了相关决策解盘列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]

        MA5 = self._sma(close, 5)
        MA10 = self._sma(close, 10)
        MA20 = self._sma(close, 20)

        EMA12 = self._ema(close, 12)
        EMA26 = self._ema(close, 26)
        DIF = EMA12 - EMA26
        DEA = self._ema(DIF, 9)
        MACD_bar = 2 * (DIF - DEA)

        result["MA5"] = MA5
        result["MA10"] = MA10
        result["MA20"] = MA20
        result["DIF"] = DIF
        result["DEA"] = DEA
        result["MACD_bar"] = MACD_bar

        return result
