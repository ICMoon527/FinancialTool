import numpy as np
import pandas as pd

from ..base import BaseIndicator


class MarketBottomCatch(BaseIndicator):
    """
    大盘抄底指标

    该指标用于识别大盘的底部机会，包括红色星、绿色星、蓝色星三种底部信号。

    公式：
    - Z：MA(C, 120)
    - VAR3：(MA(C, 5) - Z) / Z
    - VAR4：MA((CLOSE - LLV(LOW, 20)) / (HHV(HIGH, 20) - LLV(LOW, 20)) * 100, 3)

    信号说明：
    - 红色星底部信号：用于牛市和上升趋势中，表明大盘进入调整尾声或者进入波段底部
    - 绿色星底部信号：用于震荡市和下跌趋势中，表明大盘进入波段底部或者超跌反弹
    - 蓝色星底部信号：用于熊市和下跌趋势中，表明大盘已经进入疯狂超跌状态

    说明：
    红色星底部信号，用于牛市和上升趋势中，表明大盘进入调整尾声或者进入波段底部，未来大盘和个股形成上升趋势的概率较大；
    绿色星底部信号，用于震荡市和下跌趋势中，表明大盘进入波段底部或者超跌反弹，这个区域进场大盘可能形成一个小反弹行情或暂时止跌，更好的是形成波段上升；
    蓝色星底部信号，用于熊市和下跌趋势中，表明大盘已经进入疯狂超跌状态，此时进场机会大于风险。
    """

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均"""
        return data.rolling(window=period, min_periods=1).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最低低值"""
        return data.rolling(window=period, min_periods=1).min()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """计算最高高值"""
        return data.rolling(window=period, min_periods=1).max()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算大盘抄底指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了相关抄底信号列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        Z = self._sma(close, 120)
        ma5 = self._sma(close, 5)
        VAR3 = (ma5 - Z) / Z

        llv_low_20 = self._llv(low, 20)
        hhv_high_20 = self._hhv(high, 20)
        denominator = hhv_high_20 - llv_low_20
        denominator = denominator.replace(0, np.nan)
        rsv = (close - llv_low_20) / denominator * 100
        VAR4 = self._sma(rsv, 3)

        hhv_high_30 = self._hhv(high, 30)
        llv_low_30 = self._llv(low, 30)
        red_star_signal = (close > Z) & (VAR4.shift(1) < 30) & (VAR4 > VAR4.shift(1)) & (VAR4.shift(1) < VAR4.shift(2))

        green_star_signal = (VAR4.shift(1) < 7) & (VAR4 > VAR4.shift(1)) & (VAR4.shift(1) < VAR4.shift(2)) & (VAR3 < -0.1)

        blue_star_signal = ((VAR4 == 5) | (VAR4 == 5)) & (VAR3 < -0.3)

        result["Z_ma120"] = Z
        result["VAR3"] = VAR3
        result["VAR4"] = VAR4
        result["red_star_bottom"] = red_star_signal
        result["green_star_bottom"] = green_star_signal
        result["blue_star_bottom"] = blue_star_signal

        return result
