import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ThreeLineBottom(BaseIndicator):
    """
    三线狙底指标
    出现银字底，适用于多头趋势抄底
    出现金字底，适用于平衡震荡抄底
    出现钻字底，适用于熊市超跌抄底
    红空心彩带表示股价波动向上波动
    绿空心彩带表示股价波动向下波动
    底部紫色基准线表示股价波段低点
    顶部绿色基准线表示股价波段高点
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """Lowest Low Value"""
        return data.rolling(window=period).min()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Three Line Bottom indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # RSV: (((CLOSE - LLV(LOW,9)) / (HHV(HIGH,9) - LLV(LOW,9))) * 100)
        llv_low_9 = self._llv(df['Low'], 9)
        hhv_high_9 = self._hhv(df['High'], 9)
        rsv = ((df['Close'] - llv_low_9) / (hhv_high_9 - llv_low_9 + 1e-10)) * 100
        result['rsv'] = rsv

        # K:=SMA(RSV,3,1)
        k = self._sma(rsv, 3)
        result['k'] = k

        # D:=SMA(K,3,1)
        d = self._sma(k, 3)
        result['d'] = d

        # J:=((3 * K) - (2 * D))
        j = (3 * k) - (2 * d)
        result['j'] = j

        # VARA1:=(((CLOSE - EMA(CLOSE,6)) / EMA(CLOSE,6)) * 100)
        ema6 = self._ema(df['Close'], 6)
        vara1 = ((df['Close'] - ema6) / (ema6 + 1e-10)) * 100
        result['vara1'] = vara1

        # VARA2:=(((CLOSE - EMA(CLOSE,12)) / EMA(CLOSE,12)) * 100)
        ema12 = self._ema(df['Close'], 12)
        vara2 = ((df['Close'] - ema12) / (ema12 + 1e-10)) * 100
        result['vara2'] = vara2

        # VARA3:=(((CLOSE - EMA(CLOSE,24)) / EMA(CLOSE,24)) * 100)
        ema24 = self._ema(df['Close'], 24)
        vara3 = ((df['Close'] - ema24) / (ema24 + 1e-10)) * 100
        result['vara3'] = vara3

        # VARA4:=(((VARA1 + (2 * VARA2)) + (3 * VARA3)) / 6)
        vara4 = ((vara1 + (2 * vara2)) + (3 * vara3)) / 6
        result['vara4'] = vara4

        # VARA5:=MA(VARA4,3)
        vara5 = self._sma(vara4, 3)
        result['vara5'] = vara5

        # 白银底:=CROSS(K,D) AND (VARA5 <= -5)
        kd_cross = (k > d) & (k.shift(1) <= d.shift(1))
        silver_bottom = kd_cross & (vara5 <= -5)
        result['silver_bottom'] = silver_bottom

        # 黄金底:=CROSS(K,D) AND (VARA5 <= -13)
        gold_bottom = kd_cross & (vara5 <= -13)
        result['gold_bottom'] = gold_bottom

        # 钻石底:=CROSS(K,D) AND (VARA5 <= -20)
        diamond_bottom = kd_cross & (vara5 <= -20)
        result['diamond_bottom'] = diamond_bottom

        # D1:=EMA(3*SMA((CLOSE-LLV(LOW,26))/(HHV(HIGH,26)-LLV(LOW,26))*100,5,1)-2*SMA(SMA((CLOSE-LLV(LOW,26))/(HHV(HIGH,26)-LLV(LOW,26))*100,5,1),3,1),5)
        llv_low_26 = self._llv(df['Low'], 26)
        hhv_high_26 = self._hhv(df['High'], 26)
        stoch = ((df['Close'] - llv_low_26) / (hhv_high_26 - llv_low_26 + 1e-10)) * 100
        stoch_sma5 = self._sma(stoch, 5)
        stoch_sma3 = self._sma(stoch_sma5, 3)
        d1 = self._ema(3 * stoch_sma5 - 2 * stoch_sma3, 5)
        result['d1'] = d1

        # D2:=REF(D1,1)
        d2 = d1.shift(1)
        result['d2'] = d2

        # 红空心彩带: D1>=REF(D1,1)
        result['red_ribbon'] = d1 >= d2

        # 绿空心彩带: D1<REF(D1,1)
        result['green_ribbon'] = d1 < d2

        # 底: 10
        result['bottom_line'] = 10

        # 顶: 80
        result['top_line'] = 80

        return result
