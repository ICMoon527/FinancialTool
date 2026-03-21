import pandas as pd
import numpy as np
from ..base import BaseIndicator


class HotCapitalModel(BaseIndicator):
    """
    游资模型指标
    红白细线代表主力攻击线，黄粗线代表主力控盘线
    红色主力攻击线表示主力攻击线在主力控盘线之上且主力攻击线向上
    紫色3D K线表示股价连续涨停
    红色"游"图标表示当日顶级游资净流入
    绿色"游"图标表示当日顶级游资净流出
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Hot Capital Model indicator

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

        # A3:=COUNT(涨停,3)=2
        a3 = limit_up.rolling(window=3).sum() == 2
        result['a3'] = a3

        # A4:=COUNT(涨停,4)=3
        a4 = limit_up.rolling(window=4).sum() == 3
        result['a4'] = a4

        # A2ZQ:=BARSLAST(A2) - simplified as last occurrence of A2
        a2_last_occurrence = pd.Series(np.nan, index=df.index)
        for i in range(len(df)):
            if a2.iloc[i]:
                a2_last_occurrence.iloc[i] = 0
            elif i > 0 and not pd.isna(a2_last_occurrence.iloc[i-1]):
                a2_last_occurrence.iloc[i] = a2_last_occurrence.iloc[i-1] + 1
        result['a2zq'] = a2_last_occurrence

        # 连续涨停K线 (A2ZQ=0 AND A2>0)
        result['continuous_limit_up'] = (a2_last_occurrence == 0) & (a2)

        # 游资净额: YZ_BUYAMOUNT-YZ_SELLAMOUNTL
        # Note: These are special data fields, we'll use placeholders with Volume/Amount as proxy
        if 'Amount' in df.columns:
            result['hot_capital_net'] = df['Amount'] * (df['Close'] - df['Open']) / df['Close']
        else:
            result['hot_capital_net'] = df['Volume'] * (df['Close'] - df['Open']) / df['Close']

        # 游资净流入/流出信号
        result['hot_capital_inflow'] = result['hot_capital_net'] > 0
        result['hot_capital_outflow'] = result['hot_capital_net'] < 0

        return result
