import pandas as pd
import numpy as np
from ..base import BaseIndicator


class FlowingVolumeEnergy(BaseIndicator):
    """
    流动量能指标
    监控增量资金的进出情况
    红色量能柱代表量能增加，青色量能柱代表量能减少
    当量能柱在中期量能线上方时，量能流动比>1，表示增量资金总体流入
    当量能柱在中期量能线下方时，量能流动比<1，表示增量资金总体流出
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
        Calculate the Flowing Volume Energy indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # VAR1:=EMA(VOL,20)/123456
        var1 = self._ema(df['Volume'], 20) / 123456
        result['var1'] = var1

        # VAR2:=REF(VAR1,1)
        var2 = var1.shift(1)
        result['var2'] = var2

        # VAR3:=EMA(VOL,120)/123456
        var3 = self._ema(df['Volume'], 120) / 123456
        result['var3'] = var3

        # 中期量能: VAR3
        result['mid_term_volume'] = var3

        # 量能流动比: IF((VAR1/VAR3)>0,VAR1/VAR3,VAR1/VAR3)
        volume_flow_ratio = var1 / (var3 + 1e-10)
        result['volume_flow_ratio'] = volume_flow_ratio

        # 红色量能柱: VAR1>VAR2
        result['volume_increase'] = var1 > var2

        # 青色量能柱: VAR1<VAR2
        result['volume_decrease'] = var1 < var2

        return result
