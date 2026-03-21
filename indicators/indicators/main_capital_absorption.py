import numpy as np
import pandas as pd

from indicators.base import BaseIndicator


class MainCapitalAbsorption(BaseIndicator):
    """
    主力吸筹指标

    该指标识别市场中的主力资金吸筹情况，使用一系列计算来检测潜在的机构买入活动。

    公式：
    - VAR2：当前最低价与前一最低价的差值
    - VAR3：标准化波动率比率
    - VAR4：VAR3的指数移动平均
    - VAR5：38周期内的最低价
    - VAR6：38周期内的最高VAR4
    - VAR7：市场状况标记
    - VAR8：最终主力吸筹值
    """

    def _sma(self, data: pd.Series, period: int, weight: int) -> pd.Series:
        """
        计算简单移动平均（SMA）。

        Args:
            data: 输入序列
            period: SMA周期
            weight: SMA权重

        Returns:
            SMA值
        """
        return data.rolling(window=period, min_periods=1).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算指数移动平均（EMA）。

        Args:
            data: 输入序列
            period: EMA周期

        Returns:
            EMA值
        """
        return data.ewm(span=period, adjust=False).mean()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算给定周期内的最低低值（LLV）。

        Args:
            data: 输入序列
            period: 回溯周期

        Returns:
            LLV值
        """
        return data.rolling(window=period, min_periods=1).min()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算给定周期内的最高高值（HHV）。

        Args:
            data: 输入序列
            period: 回溯周期

        Returns:
            HHV值
        """
        return data.rolling(window=period, min_periods=1).max()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算主力吸筹指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'main_capital_absorption'列的DataFrame
        """
        self.validate_input(data)

        low = data["Low"].copy()
        close = data["Close"].copy()

        var2 = low - low.shift(1)

        abs_var2 = var2.abs()
        sma_abs_var2 = self._sma(abs_var2, 3, 1)
        max_var2 = var2.clip(lower=0)
        sma_max_var2 = self._sma(max_var2, 3, 1)

        denominator = sma_max_var2.replace(0, np.nan)
        var3 = (sma_abs_var2 / denominator) * 100
        var3 = var3.fillna(0)

        var4_condition = close * 1.2
        var4 = np.where(var4_condition > 0, var3 * 10, var3 / 10)
        var4 = self._ema(pd.Series(var4, index=data.index), 3)

        var5 = self._llv(low, 38)
        var6 = self._hhv(var4, 38)

        var7 = np.where(self._llv(low, 90) > 0, 1, 0)

        var8_condition = low <= var5
        var8_value = np.where(var8_condition, (var4 + var6 * 2) / 2, 0)
        var8 = self._ema(pd.Series(var8_value, index=data.index), 3) / 618 * var7

        result = data.copy()
        result["main_capital_absorption"] = var8

        return result
