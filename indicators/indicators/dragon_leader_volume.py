import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DragonLeaderVolume(BaseIndicator):
    """
    龙头量能指标
    紫色天量柱，表示当日成交量急剧放大
    黄色倍量柱，表示当日成交量显著放大
    蓝色缩量柱，表示当日成交量阶段缩小
    红色假阴真阳量柱，表示当日K线形态为假阴真阳
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Dragon Leader Volume indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 成交额0:=AMOUNT/100000000
        if 'Amount' in df.columns:
            turnover_0 = df['Amount'] / 100000000
        else:
            turnover_0 = df['Volume'] * df['Close'] / 100000000
        result['turnover_0'] = turnover_0

        # 成交额亿元: 成交额0
        result['turnover_billion'] = turnover_0

        # 实时量比: VOL/REF(VOL,1)
        real_time_volume_ratio = df['Volume'] / (df['Volume'].shift(1) + 1e-10)
        result['real_time_volume_ratio'] = real_time_volume_ratio

        # MO5:=MA(成交额0,5)
        mo5 = self._sma(turnover_0, 5)
        result['mo5'] = mo5

        # MO40:=MA(成交额0,40)
        mo40 = self._sma(turnover_0, 40)
        result['mo40'] = mo40

        # MO135:=MA(成交额0,135)
        mo135 = self._sma(turnover_0, 135)
        result['mo135'] = mo135

        # 天量: IF(成交额0/REF(成交额0,1)>3.5,1,0)
        sky_volume = (turnover_0 / (turnover_0.shift(1) + 1e-10)) > 3.5
        result['sky_volume'] = sky_volume

        # 倍量: IF((成交额0/REF(成交额0,1)>=1.5) AND (成交额0/REF(成交额0,1)<=3.5),1,0)
        turnover_ratio = turnover_0 / (turnover_0.shift(1) + 1e-10)
        double_volume = (turnover_ratio >= 1.5) & (turnover_ratio <= 3.5)
        result['double_volume'] = double_volume

        # 六日最高量:=HHV(成交额0,6)
        six_day_highest = self._hhv(turnover_0, 6)
        result['six_day_highest'] = six_day_highest

        # 缩量0:=(成交额0/六日最高量)<=0.6
        shrink_volume_0 = (turnover_0 / (six_day_highest + 1e-10)) <= 0.6
        result['shrink_volume_0'] = shrink_volume_0

        # 过滤:=FILTER(缩量0,5)
        # 简化实现：5天内只取第一次
        shrink_volume = pd.Series(False, index=df.index)
        last_true = -10
        for i in range(len(df)):
            if shrink_volume_0.iloc[i]:
                if i - last_true >= 5:
                    shrink_volume.iloc[i] = True
                    last_true = i
        result['shrink_volume'] = shrink_volume

        # 假阴真阳: IF((CLOSE>REF(CLOSE,1)) AND (CLOSE<OPEN),1,0)
        fake_yin_true_yang = (df['Close'] > df['Close'].shift(1)) & (df['Close'] < df['Open'])
        result['fake_yin_true_yang'] = fake_yin_true_yang

        return result
