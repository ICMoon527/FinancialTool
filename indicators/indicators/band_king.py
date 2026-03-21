import numpy as np
import pandas as pd

from ..base import BaseIndicator


class BandKing(BaseIndicator):
    """
    波段王指标

    该指标用于识别波段的顶底机会。

    公式：
    - MTR：MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    - ATR：MA(MTR, 14)
    - 顶底指标：基于ATR的波动幅度判断

    说明：
    识别波段的顶部和底部，提供波段交易信号。
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
        计算波段王指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'band_top', 'band_bottom'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(prev_close - high)
        tr3 = abs(prev_close - low)
        mtr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = self._sma(mtr, 14)

        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        dif = ema12 - ema26
        dea = self._ema(dif, 9)
        macd = 2 * (dif - dea)

        llv_low_14 = self._llv(low, 14)
        hhv_high_14 = self._hhv(high, 14)
        rsv = (close - llv_low_14) / (hhv_high_14 - llv_low_14) * 100
        k = self._sma(rsv, 3)
        d = self._sma(k, 3)

        band_top = (k > 80) & (d > 80) & (macd < 0)
        band_bottom = (k < 20) & (d < 20) & (macd > 0)

        result["ATR"] = atr
        result["k"] = k
        result["d"] = d
        result["macd"] = macd
        result["band_top"] = band_top
        result["band_bottom"] = band_bottom

        return result
