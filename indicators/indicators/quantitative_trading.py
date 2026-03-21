import numpy as np
import pandas as pd

from ..base import BaseIndicator


class QuantitativeTrading(BaseIndicator):
    """
    量化操盘指标

    该指标使用MACD相关指标来判断趋势。

    公式：
    - A1：EMA(C, 12) - EMA(C, 25)
    - A2：EMA(A1, 6)

    说明：
    通过快慢线的交叉和位置来判断趋势。
    """

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算量化操盘指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'A1', 'A2'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]

        ema12 = self._ema(close, 12)
        ema25 = self._ema(close, 25)
        A1 = ema12 - ema25
        A2 = self._ema(A1, 6)

        result["A1"] = A1
        result["A2"] = A2

        return result
