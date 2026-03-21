import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TwoEightStyle(BaseIndicator):
    """
    二八风格指标
    监控市场里大盘股与小盘股轮动走强的状态
    红色线代表大盘股走势，黄色线代表小盘股走势
    红色柱代表轮动大盘股走强，黄色柱代表轮动小盘股走强
    红色线和黄色线均在零轴下方运行时代表市场整体走弱，无明显风格偏好
    """

    def __init__(self):
        super().__init__()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Two Eight Style indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 用价格动量作为大盘和小盘风格的代理
        # 沪深300风格代理（慢动量 - 大盘风格）
        price_change_13 = (df['Close'] / df['Close'].shift(13) - 1) * 100
        result['large_cap_momentum'] = self._ema(price_change_13, 2)

        # 平均股价风格代理（快动量 - 小盘风格）
        price_momentum = (df['Close'] / self._sma(df['Close'], 13) - 1) * 100
        result['small_cap_momentum'] = self._ema(price_momentum, 2)

        # CI:=小盘-大盘
        ci = result['small_cap_momentum'] - result['large_cap_momentum']
        result['ci'] = ci

        # 黄色柱: CI>0 AND 小盘>0
        result['small_cap_bar'] = (ci > 0) & (result['small_cap_momentum'] > 0)

        # 红色柱: CI<=0 AND 大盘>0
        result['large_cap_bar'] = (ci <= 0) & (result['large_cap_momentum'] > 0)

        # 谨慎: 小盘<=0 AND 大盘<=0
        result['cautious'] = (result['small_cap_momentum'] <= 0) & (result['large_cap_momentum'] <= 0)

        # 零轴
        result['zero_axis'] = 0

        return result
