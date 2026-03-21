import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BandBullBear(BaseIndicator):
    """
    波段牛熊指标
    红色K线表示股价趋势走牛，应持股待涨
    绿色K线表示股价趋势走熊，应持币观望
    紫色3D K线表示股价连续涨停
    红白细线代表主力攻击线，黄粗线代表主力控盘线
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """Lowest Low Value"""
        return data.rolling(window=period).min()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Band Bull Bear indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 主力攻击线: (MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,9))/3
        ma3 = self._sma(df['Close'], 3)
        ma6 = self._sma(df['Close'], 6)
        ma9 = self._sma(df['Close'], 9)
        result['main_attack_line'] = (ma3 + ma6 + ma9) / 3

        # 主力控盘线: (MA(CLOSE,5)+MA(CLOSE,10)+MA(CLOSE,15)+MA(CLOSE,20)+MA(CLOSE,25)+MA(CLOSE,30))/6
        ma5 = self._sma(df['Close'], 5)
        ma10 = self._sma(df['Close'], 10)
        ma15 = self._sma(df['Close'], 15)
        ma20 = self._sma(df['Close'], 20)
        ma25 = self._sma(df['Close'], 25)
        ma30 = self._sma(df['Close'], 30)
        result['main_control_line'] = (ma5 + ma10 + ma15 + ma20 + ma25 + ma30) / 6

        # 红色主力攻击线条件
        result['attack_line_red'] = (
            (result['main_attack_line'] > result['main_attack_line'].shift(1)) &
            (result['main_attack_line'] > result['main_control_line'])
        )

        # 涨停: (CLOSE/REF(CLOSE,1))-1>=0.090 AND CLOSE=HIGH
        change_pct = df['Close'] / df['Close'].shift(1) - 1
        limit_up = (change_pct >= 0.090) & (df['Close'] == df['High'])
        result['limit_up'] = limit_up

        # A2:=COUNT(涨停,2)=2
        a2 = limit_up.rolling(window=2).sum() == 2
        result['a2'] = a2

        # AAA:=(3*C+H+L+O)/6
        aaa = (3 * df['Close'] + df['High'] + df['Low'] + df['Open']) / 6
        result['aaa'] = aaa

        # VAR1:=EMA(AAA,35)
        var1 = self._ema(aaa, 35)
        result['var1'] = var1

        # VAR2:=(HHV(VAR1,5)+HHV(VAR1,15)+HHV(VAR1,30))/3
        var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
        result['var2'] = var2

        # VAR3:=(LLV(VAR1,5)+LLV(VAR1,15)+LLV(VAR1,30))/3
        var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
        result['var3'] = var3

        # 牛线:=(HHV(VAR2,5)+HHV(VAR2,15)+HHV(VAR2,30))/3
        bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
        result['bull_line'] = bull_line

        # 熊线:=(LLV(VAR3,5)+LLV(VAR3,15)+LLV(VAR3,30))/3
        bear_line = (self._llv(var3, 5) + self._llv(var3, 15) + self._llv(var3, 30)) / 3
        result['bear_line'] = bear_line

        # 红色K线（牛）: AAA>牛线
        result['bull_kline'] = aaa > bull_line

        # 绿色K线（熊）: AAA<熊线
        result['bear_kline'] = aaa < bear_line

        # 连续涨停
        result['continuous_limit_up'] = a2

        return result
