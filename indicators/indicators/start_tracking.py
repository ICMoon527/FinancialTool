import numpy as np
import pandas as pd

from indicators.base import BaseIndicator


class StartTracking(BaseIndicator):
    """
    启动追踪指标

    该指标使用主力攻击线和主力控盘线来追踪潜在市场行情的启动。

    公式：
    - 主力攻击线：(MA(CLOSE,3) + MA(CLOSE,6) + MA(CLOSE,9)) / 3
    - 主力控盘线：(MA(CLOSE,5) + MA(CLOSE,10) + MA(CLOSE,15) + MA(CLOSE,20) + MA(CLOSE,25) + MA(CLOSE,30)) / 6
    """

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算简单移动平均（SMA）。

        Args:
            data: 输入序列
            period: SMA周期

        Returns:
            SMA值
        """
        return data.rolling(window=period, min_periods=1).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算启动追踪指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了指标列的DataFrame
        """
        self.validate_input(data)

        close = data["Close"].copy()

        ma3 = self._sma(close, 3)
        ma6 = self._sma(close, 6)
        ma9 = self._sma(close, 9)
        main_attack_line = (ma3 + ma6 + ma9) / 3

        ma5 = self._sma(close, 5)
        ma10 = self._sma(close, 10)
        ma15 = self._sma(close, 15)
        ma20 = self._sma(close, 20)
        ma25 = self._sma(close, 25)
        ma30 = self._sma(close, 30)
        main_control_line = (ma5 + ma10 + ma15 + ma20 + ma25 + ma30) / 6

        result = data.copy()
        result["main_attack_line"] = main_attack_line
        result["main_control_line"] = main_control_line

        return result
