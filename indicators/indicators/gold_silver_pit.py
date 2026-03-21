import numpy as np
import pandas as pd

from ..base import BaseIndicator


class GoldSilverPit(BaseIndicator):
    """
    金银坑指标

    该指标用于识别黄金坑和白银坑等底部形态。

    公式：
    - VAR1：(CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100
    - 黄金坑：VAR1的极端低位

    说明：
    识别黄金坑、白银坑等超跌后的反弹机会。
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
        计算金银坑指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'gold_pit', 'silver_pit'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        llv_low_9 = self._llv(low, 9)
        hhv_high_9 = self._hhv(high, 9)
        denominator = hhv_high_9 - llv_low_9
        denominator = denominator.replace(0, np.nan)
        VAR1 = (close - llv_low_9) / denominator * 100

        VAR2 = self._sma(VAR1, 3)
        VAR3 = self._sma(VAR2, 3)

        gold_pit = (VAR3 < 5) & (VAR3.shift(1) > VAR3) & (VAR3.shift(2) > VAR3.shift(1))
        silver_pit = (VAR3 < 10) & (VAR3 >= 5) & (VAR3.shift(1) > VAR3)

        result["VAR1"] = VAR1
        result["VAR2"] = VAR2
        result["VAR3"] = VAR3
        result["gold_pit"] = gold_pit
        result["silver_pit"] = silver_pit

        return result
