import numpy as np
import pandas as pd

from ..base import BaseIndicator


class MainMonitoring(BaseIndicator):
    """
    主力监控指标

    该指标监控主力控盘程度。

    公式：
    - N：35
    - M：35
    - N1：3
    - B1：(HHV(H, N) - C) / (HHV(H, N) - LLV(L, N)) * 100 - M
    - B2：SMA(B1, N, 1) + 100
    - B3：(C - LLV(L, N)) / (HHV(H, N) - LLV(L, N)) * 100
    - B4：SMA(B3, 7, 1)
    - B5：SMA(B4, 5, 1) + 100
    - B6：B5 - B2
    - 控盘程度：IF(B6 > N1, B6 - N1, 0) * 3.5

    说明：
    1. 监控主力控盘程度，黄柱表示开始控盘，红柱表示完成控盘。
    2. 柱体逐步增高，表示控盘程度上升。
    3. 柱体逐步降低，表示控盘程度下降。
    """

    def _sma(self, data: pd.Series, period: int, weight: int) -> pd.Series:
        """计算简单移动平均"""
        return data.rolling(window=period, min_periods=1).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最高高值"""
        return data.rolling(window=period, min_periods=1).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最低低值"""
        return data.rolling(window=period, min_periods=1).min()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算主力监控指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'control_degree'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        N = 35
        M = 35
        N1 = 3

        hhv_high_n = self._hhv(high, N)
        llv_low_n = self._llv(low, N)
        denominator = hhv_high_n - llv_low_n
        denominator = denominator.replace(0, np.nan)

        B1 = (hhv_high_n - close) / denominator * 100 - M
        B2 = self._sma(B1, N, 1) + 100

        B3 = (close - llv_low_n) / denominator * 100
        B4 = self._sma(B3, 7, 1)
        B5 = self._sma(B4, 5, 1) + 100

        B6 = B5 - B2
        control_degree = np.where(B6 > N1, (B6 - N1) * 3.5, 0)

        result["B1"] = B1
        result["B2"] = B2
        result["B3"] = B3
        result["B4"] = B4
        result["B5"] = B5
        result["B6"] = B6
        result["control_degree"] = control_degree

        return result
