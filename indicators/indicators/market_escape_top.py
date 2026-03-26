import logging
import pandas as pd
import numpy as np
from ..base import BaseIndicator

logger = logging.getLogger(__name__)


class MarketEscapeTop(BaseIndicator):
    """
    大盘逃顶指标
    帮助投资者在大盘见顶时及时逃顶
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Market Escape Top indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)
        logger.warning("[大盘逃顶指标] 注意：本指标使用个股数据模拟大盘分析，非真实大盘数据")

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # RSI指标
        rsi_14 = self._rsi(df['Close'], 14)
        result['rsi_14'] = rsi_14

        # MACD指标
        ema12 = self._ema(df['Close'], 12)
        ema26 = self._ema(df['Close'], 26)
        macd = ema12 - ema26
        signal = self._ema(macd, 9)
        histogram = macd - signal
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = histogram

        # KDJ指标
        low_min = df['Low'].rolling(window=9).min()
        high_max = df['High'].rolling(window=9).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-10) * 100
        k = self._sma(rsv, 3)
        d = self._sma(k, 3)
        j = 3 * k - 2 * d
        result['k'] = k
        result['d'] = d
        result['j'] = j

        # 逃顶信号1: RSI超买
        result['escape_signal_1'] = rsi_14 > 80

        # 逃顶信号2: MACD顶背离
        result['escape_signal_2'] = (macd < macd.shift(1)) & (df['Close'] > df['Close'].shift(1))

        # 逃顶信号3: KDJ超买
        result['escape_signal_3'] = (k > 80) & (d > 80)

        # 综合逃顶信号
        result['escape_top_signal'] = result['escape_signal_1'] | result['escape_signal_2'] | result['escape_signal_3']

        return result
