import pandas as pd
import numpy as np
from ..base import BaseIndicator


class StrongDetonation(BaseIndicator):
    """
    强势起爆指标
    黄色线是个股强势趋势线，K线高于黄色线表示个股趋势走强
    红色网状线是大盘走势，K线高于红色网状线表示个股中期跑赢大市
    紫色K线表示个股同时高于黄色线和红色网状线，股价处于强势起爆阶段
    红色K线表示个股高于红色网状线低于黄色线，股价处于强势蓄势阶段
    灰色K线表示个股处于弱势震荡阶段
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
        Calculate the Strong Detonation indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

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

        # A1:= 平均股价CLOSE /EMA(平均股价CLOSE,120)
        # 用个股价格作为代理
        a1 = df['Close'] / self._ema(df['Close'], 120)
        result['a1'] = a1

        # 大盘中线: EMA(EMA(C,120)*A1,2)
        ema120 = self._ema(df['Close'], 120)
        market_midline = self._ema(ema120 * a1, 2)
        result['market_midline'] = market_midline

        # 强势: AAA>=牛线 AND AAA>=大盘中线
        strong = (aaa >= bull_line) & (aaa >= market_midline)
        result['strong'] = strong

        # 次强: AAA<牛线 AND AAA>=大盘中线 AND 牛线>大盘中线
        sub_strong = (aaa < bull_line) & (aaa >= market_midline) & (bull_line > market_midline)
        result['sub_strong'] = sub_strong

        # 弱势: (AAA<=大盘中线 AND AAA<=牛线) OR (AAA>=牛线 AND AAA<大盘中线 AND 牛线<大盘中线)
        weak = ((aaa <= market_midline) & (aaa <= bull_line)) | (
            (aaa >= bull_line) & (aaa < market_midline) & (bull_line < market_midline)
        )
        result['weak'] = weak

        return result
