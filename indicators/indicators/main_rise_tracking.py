import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MainRiseTracking(BaseIndicator):
    """
    主升追踪指标
    红白细线代表主力攻击线，黄粗线代表主力控盘线
    红色主力攻击线表示主力攻击线在主力控盘线之上且主力攻击线向上
    紫色3D效果K线表示股价连续涨停
    黄色星星图标表示龙头主升信号
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Main Rise Tracking indicator

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

        # 连续涨停K线 (A2ZQ=0)
        result['continuous_limit_up'] = (a2_last_occurrence == 0)

        # 成交额1:=AMOUNT>=REF(HHV(AMOUNT,10),1)
        if 'Amount' in df.columns:
            hhv_amount_10 = df['Amount'].rolling(window=10).max()
            result['turnover_condition'] = df['Amount'] >= hhv_amount_10.shift(1)
        else:
            result['turnover_condition'] = True

        # 实体:=CLOSE>LOW
        result['has_body'] = df['Close'] > df['Low']

        # B1:=A2ZQ=0 AND A3=1 AND 成交额1=1 AND 实体=1
        result['b1'] = (
            (a2_last_occurrence == 0) &
            a3 &
            result['turnover_condition'] &
            result['has_body']
        )

        # B2:=A2ZQ=0 AND A4=1 AND 成交额1=1 AND 实体=1 AND (REF(CLOSE,1)=REF(LOW,1) OR REF(成交额1,1)=0)
        result['b2'] = (
            (a2_last_occurrence == 0) &
            a4 &
            result['turnover_condition'] &
            result['has_body'] &
            ((df['Close'].shift(1) == df['Low'].shift(1)) | (~result['turnover_condition'].shift(1).fillna(True).astype(bool)))
        )

        # 龙头主升信号
        result['dragon_rise_signal'] = result['b1'] | result['b2']

        return result
