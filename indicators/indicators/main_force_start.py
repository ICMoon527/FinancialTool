import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MainForceStart(BaseIndicator):
    """
    主力起动指标
    蓝色柱体为主力低控盘，即控盘程度>=50
    红色柱体为主力中控盘，即控盘程度>=60
    紫色柱体为主力高控盘，即控盘程度>=80
    控盘程度越高越好
    """

    def __init__(self):
        super().__init__()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Main Force Start indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # AAA:=(3*CLOSE+OPEN+HIGH+LOW)/6
        aaa = (3 * df['Close'] + df['Open'] + df['High'] + df['Low']) / 6
        result['aaa'] = aaa

        # MA12:=EMA(AAA,12)
        ma12 = self._ema(aaa, 12)
        result['ma12'] = ma12

        # MA36:=EMA(AAA,36)
        ma36 = self._ema(aaa, 36)
        result['ma36'] = ma36

        # MA108:=EMA(AAA,108)
        ma108 = self._ema(aaa, 108)
        result['ma108'] = ma108

        # MA250:=EMA(AAA,250)
        ma250 = self._ema(aaa, 250)
        result['ma250'] = ma250

        # 控盘度: (EMA(AAA,6)-REF(EMA(AAA,18),1))/REF(EMA(AAA,18),1)*100+55
        ema6 = self._ema(aaa, 6)
        ema18 = self._ema(aaa, 18)
        control_degree = ((ema6 - ema18.shift(1)) / (ema18.shift(1) + 1e-10)) * 100 + 55
        result['control_degree'] = control_degree

        # 低控盘: IF(控盘度>=50,控盘度,0)
        result['low_control'] = control_degree.where(control_degree >= 50, 0)

        # 中控盘: IF(控盘度>=60,控盘度,0)
        result['medium_control'] = control_degree.where(control_degree >= 60, 0)

        # 高控盘: IF(控盘度>=80,控盘度,0)
        result['high_control'] = control_degree.where(control_degree >= 80, 0)

        return result
