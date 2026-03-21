import numpy as np
import pandas as pd

from ..base import BaseIndicator


class RocketMainRise(BaseIndicator):
    """
    火箭主升指标

    该指标用于识别火箭发射式的主升浪。

    公式：
    - 指标逻辑：识别极端上涨行情

    说明：
    识别火箭发射式的主升浪行情。
    """

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

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
        计算火箭主升指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'rocket_signal'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        ema3 = self._ema(close, 3)
        ema6 = self._ema(close, 6)
        ema18 = self._ema(close, 18)

        ema_12 = self._ema(close, 12)
        ema_25 = self._ema(close, 25)

        ema_diff = ema_12 - ema_25
        ema_signal = self._ema(ema_diff, 6)

        vol_macd = (ema_diff - ema_signal) * 2

        low_min = self._llv(low, 30)
        price_change = (close - low_min) / low_min * 100

        volume_ma = self._sma(data["Volume"], 5)
        volume_burst = data["Volume"] > volume_ma * 1.5

        rocket_signal = (ema_diff > 0) & (ema_signal > 0) & (ema_diff > ema_signal) & (price_change > 15) & volume_burst

        result["ema12"] = ema_12
        result["ema25"] = ema_25
        result["ema_diff"] = ema_diff
        result["ema_signal"] = ema_signal
        result["vol_macd"] = vol_macd
        result["price_change"] = price_change
        result["volume_burst"] = volume_burst
        result["rocket_signal"] = rocket_signal

        return result
