import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VolumeAnomaly(BaseIndicator):
    """
    量能异动指标
    黄柱表示当日异动放量
    蓝柱表示当日异动缩量
    实线表示阶段放量
    虚线表示阶段缩量
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
        Calculate the Volume Anomaly indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 成交额亿元: AMOUNT/100000000
        if 'Amount' in df.columns:
            turnover_billion = df['Amount'] / 100000000
        else:
            turnover_billion = df['Volume'] * df['Close'] / 100000000
        result['turnover_billion'] = turnover_billion

        # MA5:=MA(成交额亿元,5)
        ma5 = self._sma(turnover_billion, 5)
        result['ma5'] = ma5

        # MA10:=MA(成交额亿元,10)
        ma10 = self._sma(turnover_billion, 10)
        result['ma10'] = ma10

        # H10:=REF(HHV(成交额亿元,10),1)
        h10 = self._hhv(turnover_billion, 10).shift(1)
        result['h10'] = h10

        # L10:=REF(LLV(成交额亿元,10),1)
        l10 = self._llv(turnover_billion, 10).shift(1)
        result['l10'] = l10

        # 量能异动: IF(成交额亿元>=H10,1,0)
        volume_anomaly = pd.Series(0, index=df.index)
        volume_anomaly = volume_anomaly.where(turnover_billion < h10, 1)
        result['volume_anomaly'] = volume_anomaly

        # M0:=EMA(成交额亿元,10)
        m0 = self._ema(turnover_billion, 10)
        result['m0'] = m0

        # M1:=EMA(M0,2)
        m1 = self._ema(m0, 2)
        result['m1'] = m1

        # IFF: IF(M1>REF(M1,1),M1,DRAWNULL)
        result['m1_up'] = m1 > m1.shift(1)

        # 黄柱: 成交额亿元>=H10
        result['high_volume_anomaly'] = turnover_billion >= h10

        # 蓝柱: 成交额亿元<=L10
        result['low_volume_anomaly'] = turnover_billion <= l10

        return result
