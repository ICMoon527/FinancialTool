import numpy as np
import pandas as pd

from ..base import BaseIndicator


class MainInOut(BaseIndicator):
    """
    主力进出指标

    该指标用于监控主力资金的进出情况。

    公式：
    - 主力进：(CLOSE - LLV(LOW, 30)) / (HHV(HIGH, 30) - LLV(LOW, 30)) * 100
    - 主力出：EMA(主力进, 3)
    - 进出线：EMA(主力进, 5)

    说明：
    1. 主力进线上穿主力出线，表示主力资金进入
    2. 主力进线下穿主力出线，表示主力资金流出
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
        计算主力进出指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'main_in', 'main_out', 'in_out_line'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        llv_low_30 = self._llv(low, 30)
        hhv_high_30 = self._hhv(high, 30)
        denominator = hhv_high_30 - llv_low_30
        denominator = denominator.replace(0, np.nan)

        main_in = (close - llv_low_30) / denominator * 100
        main_out = self._ema(main_in, 3)
        in_out_line = self._ema(main_in, 5)

        main_in_signal = (main_in > main_out) & (main_in.shift(1) <= main_out.shift(1))
        main_out_signal = (main_in < main_out) & (main_in.shift(1) >= main_out.shift(1))

        result["main_in"] = main_in
        result["main_out"] = main_out
        result["in_out_line"] = in_out_line
        result["main_in_signal"] = main_in_signal
        result["main_out_signal"] = main_out_signal

        return result
