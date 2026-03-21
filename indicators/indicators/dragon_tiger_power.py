import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DragonTigerPower(BaseIndicator):
    """
    龙虎动力指标
    把动力运行周期分为增长、衰减、萧条、复苏四个阶段
    红实柱代表动力增长阶段，价格正在上涨，买进或持有
    绿实柱代表动力衰减阶段，价格出现回落或震荡，可适当减仓
    绿虚柱代表动力萧条阶段，价格正在下跌或是将要下跌，宜观望
    红虚柱代表动力复苏阶段，价格出现反弹或震荡的概率增加
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
        Calculate the Dragon Tiger Power indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # TT:=(2*CLOSE+OPEN+HIGH+LOW)
        tt = 2 * df['Close'] + df['Open'] + df['High'] + df['Low']
        result['tt'] = tt

        # 主导动能:=100*(TT/EMA(TT,4)-1)
        ema_tt_4 = self._ema(tt, 4)
        dominant_power = 100 * (tt / (ema_tt_4 + 1e-10) - 1)
        result['dominant_power'] = dominant_power

        # FF:=主导动能
        ff = dominant_power
        result['ff'] = ff

        # 红实柱动力增长: FF>0 AND FF>REF(FF,1)
        power_growth = (ff > 0) & (ff > ff.shift(1))
        result['power_growth'] = power_growth

        # 绿实柱动力衰减: FF>0 AND FF<=REF(FF,1)
        power_decay = (ff > 0) & (ff <= ff.shift(1))
        result['power_decay'] = power_decay

        # 绿虚柱动力萧条: FF<=0 AND FF<=REF(FF,1)
        power_depression = (ff <= 0) & (ff <= ff.shift(1))
        result['power_depression'] = power_depression

        # 红虚柱动力复苏: FF<=0 AND FF>REF(FF,1)
        power_recovery = (ff <= 0) & (ff > ff.shift(1))
        result['power_recovery'] = power_recovery

        # 成交额亿元:=AMOUNT/100000000
        if 'Amount' in df.columns:
            turnover_billion = df['Amount'] / 100000000
        else:
            turnover_billion = df['Volume'] * df['Close'] / 100000000
        result['turnover_billion'] = turnover_billion

        # MA3:=MA(成交额亿元,2)
        ma3 = self._sma(turnover_billion, 2)
        result['ma3'] = ma3

        # MA20:=MA(成交额亿元,20)
        ma20 = self._sma(turnover_billion, 20)
        result['ma20'] = ma20

        # H10:=REF(HHV(成交额亿元,10),1)
        h10 = self._hhv(turnover_billion, 10).shift(1)
        result['h10'] = h10

        # L10:=REF(LLV(成交额亿元,10),1)
        l10 = self._llv(turnover_billion, 10).shift(1)
        result['l10'] = l10

        # 量比:=EMA(MA3/MA20,2)
        volume_ratio = self._ema(ma3 / (ma20 + 1e-10), 2)
        result['volume_ratio'] = volume_ratio

        # 量能增长: FF>0 AND FF>REF(FF,1) AND 量比>=1.20
        volume_growth = power_growth & (volume_ratio >= 1.20)
        result['volume_growth'] = volume_growth

        return result
