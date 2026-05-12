import numpy as np
import pandas as pd

from indicators.base import BaseIndicator


class MainCapitalDistribution(BaseIndicator):
    """
    主力出货指标（主力吸筹公式的镜像）

    该指标识别市场中的主力资金出货情况（高位派发行为），
    使用主力吸筹公式的镜像结构：
    - 最高价替代最低价
    - HHV替代LLV
    - 检测"创新高时的下跌动能爆发"

    公式：
    - VAR2：当前最高价与前一最高价的差值
    - VAR3：标准化波动率比率（下跌分量占比）
    - VAR4：VAR3的指数移动平均
    - VAR5：38周期内的最高价
    - VAR6：38周期内的最低VAR4
    - VAR7：市场状况标记
    - VAR8：最终主力出货值（负值表示出货信号）
    """

    def __init__(self, filter_threshold: float = 1.01, weakness_threshold: float = 0.30):
        """
        初始化主力出货指标

        Args:
            filter_threshold: 信号过滤阈值，绝对值小于此值的信号将被置零（默认 1.01）
            weakness_threshold: 收盘弱势阈值，上影线占比小于此值不显示出货柱（默认 0.30）
        """
        super().__init__()
        self.filter_threshold = filter_threshold
        self.weakness_threshold = weakness_threshold

    def _sma(self, data: pd.Series, period: int, weight: int) -> pd.Series:
        """计算简单移动平均（SMA）"""
        return data.rolling(window=period, min_periods=1).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均（EMA）"""
        return data.ewm(span=period, adjust=False).mean()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """计算给定周期内的最低低值（LLV）"""
        return data.rolling(window=period, min_periods=1).min()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """计算给定周期内的最高高值（HHV）"""
        return data.rolling(window=period, min_periods=1).max()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算主力出货指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'main_capital_distribution'列的DataFrame
        """
        self.validate_input(data)

        high = data["High"].copy()
        close = data["Close"].copy()

        var2 = high - high.shift(1)

        abs_var2 = var2.abs()
        sma_abs_var2 = self._sma(abs_var2, 3, 1)
        min_var2_abs = var2.clip(upper=0).abs()
        sma_min_var2 = self._sma(min_var2_abs, 3, 1)

        denominator = sma_min_var2.replace(0, np.nan)
        var3 = (sma_abs_var2 / denominator) * 100
        var3 = var3.fillna(0)

        var4_condition = close * 0.8
        var4 = np.where(var4_condition > 0, var3 * 10, var3 / 10)
        var4 = self._ema(pd.Series(var4, index=data.index), 3)

        var5 = self._hhv(high, 38)
        var6 = self._llv(var4, 38)

        var7 = np.where(self._hhv(high, 90) > 0, 1, 0)

        var8_condition = high >= var5
        var8_value = np.where(var8_condition, (var4 + var6 * 2) / 2, 0)
        var8 = self._ema(pd.Series(var8_value, index=data.index), 3) / 618 * var7

        upper_shadow = (high - close) / (high - data["Low"] + 1e-10)
        var8 = np.where(upper_shadow < self.weakness_threshold, 0, var8)
        var8 = np.where(np.abs(var8) < self.filter_threshold, 0, var8)

        result = data.copy()
        result["main_capital_distribution"] = -var8

        return result