import pandas as pd
import numpy as np
from ..base import BaseIndicator


class InfiniteOrbit(BaseIndicator):
    """
    无极轨道指标
    次上轨白细线代表短期压力线，次下轨黄细线代表短期支撑线
    上轨绿粗线代表中期压力线，下轨红粗线代表中期支撑线
    短周期中，股价在次下轨容易获得支撑，在次上轨容易形成压力
    中周期中，股价在下轨容易获得支撑，在上轨容易形成压力
    蓝色K线表示股价触及上轨，处于超买状态
    紫色K线表示股价触及下轨，处于超卖状态
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Infinite Orbit indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # VAR2:=CLOSE*VOL
        var2 = df['Close'] * df['Volume']
        result['var2'] = var2

        # VAR3:=EMA((EMA(VAR2,5)/EMA(VOL,5)+EMA(VAR2,10)/EMA(VOL,10)+EMA(VAR2,20)/EMA(VOL,20)+EMA(VAR2,40)/EMA(VOL,40))/4,15)
        ema_var2_5 = self._ema(var2, 5)
        ema_vol_5 = self._ema(df['Volume'], 5)
        ema_var2_10 = self._ema(var2, 10)
        ema_vol_10 = self._ema(df['Volume'], 10)
        ema_var2_20 = self._ema(var2, 20)
        ema_vol_20 = self._ema(df['Volume'], 20)
        ema_var2_40 = self._ema(var2, 40)
        ema_vol_40 = self._ema(df['Volume'], 40)

        inner_var3 = (
            (ema_var2_5 / (ema_vol_5 + 1e-10)) +
            (ema_var2_10 / (ema_vol_10 + 1e-10)) +
            (ema_var2_20 / (ema_vol_20 + 1e-10)) +
            (ema_var2_40 / (ema_vol_40 + 1e-10))
        ) / 4
        var3 = self._ema(inner_var3, 15)
        result['var3'] = var3

        # 次上轨: 1.07*VAR3
        result['secondary_upper'] = 1.07 * var3

        # 次下轨: VAR3*0.93
        result['secondary_lower'] = var3 * 0.93

        # VAR4:=EMA(CLOSE,10)
        var4 = self._ema(df['Close'], 10)
        result['var4'] = var4

        # 上轨: EMA(VAR4*1.15,5)
        result['upper'] = self._ema(var4 * 1.15, 5)

        # 下轨: EMA(VAR4*0.85,5)
        result['lower'] = self._ema(var4 * 0.85, 5)

        # 触及下轨超卖信号
        result['oversold'] = (df['Low'] <= result['lower']) & (df['Close'] > df['Open'])

        # 触及上轨超买信号
        result['overbought'] = (df['High'] >= result['upper']) & (df['Close'] > df['Open'])

        # 下轨交叉信号: CROSS(CLOSE,下轨)
        cross_lower = (df['Close'] > result['lower']) & (df['Close'].shift(1) <= result['lower'].shift(1))
        result['cross_lower_signal'] = cross_lower

        # 上轨交叉信号: CROSS(上轨,CLOSE)
        cross_upper = (result['upper'] > df['Close']) & (result['upper'].shift(1) <= df['Close'].shift(1))
        result['cross_upper_signal'] = cross_upper

        return result
