import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PositionAllocation(BaseIndicator):
    """
    仓位配置指标
    帮助投资者根据大盘不同状态量化仓位管理
    市场行情高涨时提升仓位最大化赚取利润
    市场行情低迷时降低仓位规避主跌浪风险
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

    def _barslast(self, condition: pd.Series) -> pd.Series:
        """Bars since last occurrence of condition"""
        result = pd.Series(np.nan, index=condition.index)
        last_bar = -1
        for i in range(len(condition)):
            if condition.iloc[i]:
                last_bar = i
                result.iloc[i] = 0
            elif last_bar != -1:
                result.iloc[i] = i - last_bar
        return result

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Position Allocation indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # TR1:=MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
        tr1 = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['Close'].shift(1) - df['High']),
            'lc': abs(df['Close'].shift(1) - df['Low'])
        }).max(axis=1)
        result['tr1'] = tr1

        # ATR:=MA(TR1,10)
        atr = self._sma(tr1, 10)
        result['atr'] = atr

        # MEDIAN:=(HIGH + LOW)/2
        median = (df['High'] + df['Low']) / 2
        result['median'] = median

        # UP:=MEDIAN+ATR*2.2, DN:=MEDIAN-ATR*2.2 (simplified)
        up = median + atr * 2.2
        dn = median - atr * 2.2
        result['up'] = up
        result['dn'] = dn

        # AAA:=(3*CLOSE+HIGH+LOW+OPEN)/6
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

        # 疯熊:=熊线*0.90
        crazy_bear = bear_line * 0.90
        result['crazy_bear'] = crazy_bear

        # 量能流动比:=MA(VOL,20)/MA(VOL,200)
        vol_ma20 = self._sma(df['Volume'], 20)
        vol_ma200 = self._sma(df['Volume'], 200)
        volume_flow_ratio = vol_ma20 / (vol_ma200 + 1e-10)
        result['volume_flow_ratio'] = volume_flow_ratio

        # 简化的仓位建议
        result['position_100'] = (aaa > bull_line) & (volume_flow_ratio >= 1.1)
        result['position_75'] = (aaa > bull_line) & ~result['position_100']
        result['position_25'] = aaa < crazy_bear
        result['position_0'] = (aaa < bear_line) & (aaa >= crazy_bear) & ~result['position_25']
        result['position_50'] = ~(result['position_100'] | result['position_75'] | result['position_25'] | result['position_0'])

        # 四季状态
        result['spring'] = result['position_75'] | result['position_100']
        result['summer'] = result['position_50']
        result['autumn'] = result['position_0']
        result['winter'] = result['position_25']

        return result
