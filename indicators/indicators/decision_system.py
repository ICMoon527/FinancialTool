import numpy as np
import pandas as pd

from ..base import BaseIndicator


class DecisionSystem(BaseIndicator):
    """
    决策系统指标

    该指标提供基于均线的决策信号。

    公式：
    - 上界：MA(HIGH, 20)
    - 下界：MA(LOW, 20)
    - 趋势线：MA(CLOSE, 20)
    - K：(C - LLV(L, 3)) / (HHV(H, 3) - LLV(L, 3)) * 100
    - D：SMA(K, 5, 1)
    - J：3 * K - 2 * D

    说明：
    1. 上界为压力线
    2. 下界为支撑线
    3. 趋势线表示当前趋势方向
    4. KDJ指标用于超买超卖判断
    """

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
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
        计算决策系统指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了相关决策系统列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        upper_bound = self._sma(high, 20)
        lower_bound = self._sma(low, 20)
        trend_line = self._sma(close, 20)

        llv_low_3 = self._llv(low, 3)
        hhv_high_3 = self._hhv(high, 3)
        denominator = hhv_high_3 - llv_low_3
        denominator = denominator.replace(0, np.nan)
        K = (close - llv_low_3) / denominator * 100
        D = self._sma(K, 5)
        J = 3 * K - 2 * D

        result["upper_bound"] = upper_bound
        result["lower_bound"] = lower_bound
        result["trend_line"] = trend_line
        result["K"] = K
        result["D"] = D
        result["J"] = J

        return result
