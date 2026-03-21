import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ThreeLineReversal(BaseIndicator):
    """
    三线扭转指标
    三线从分散趋向贴合，即出现红色向上箭头时，代表行情出现向上反转的趋势
    三线从贴合趋向分散，即出现绿色向下箭头时，代表行情出现向下反转的趋势
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

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Three Line Reversal indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # CC:=(3*C+O+L+H)/6
        cc = (3 * df['Close'] + df['Open'] + df['Low'] + df['High']) / 6
        result['cc'] = cc

        # 灰线: EMA(EMA(CC,20),2)
        gray_line = self._ema(self._ema(cc, 20), 2)
        result['gray_line'] = gray_line

        # 黄线: HHV(灰线,5)
        yellow_line = self._hhv(gray_line, 5)
        result['yellow_line'] = yellow_line

        # 紫线: 灰线-(黄线-灰线)
        purple_line = gray_line - (yellow_line - gray_line)
        result['purple_line'] = purple_line

        # 红色向上箭头: 黄线=紫线 AND REF(黄线,1)>REF(紫线,1)
        up_reversal = (yellow_line == purple_line) & (yellow_line.shift(1) > purple_line.shift(1))
        result['up_reversal'] = up_reversal

        # 绿色向下箭头: 黄线>紫线 AND REF(黄线,1)=REF(紫线,1)
        down_reversal = (yellow_line > purple_line) & (yellow_line.shift(1) == purple_line.shift(1))
        result['down_reversal'] = down_reversal

        return result
