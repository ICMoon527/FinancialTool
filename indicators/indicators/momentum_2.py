import logging
import pandas as pd
import numpy as np
from ..base import BaseIndicator

logger = logging.getLogger(__name__)


class Momentum2(BaseIndicator):
    """
    动能二号指标
    红色柱状表示资金正在主导价格上涨或即将主导价格上涨
    黄色柱状表示获利资金主导价格
    绿色柱状表示资金主导价格下跌
    蓝色柱状表示空头回补资金主导价格
    """

    def __init__(self):
        super().__init__()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Momentum 2 indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)
        # logger.info("[动能二号指标] 数据验证通过，开始计算指标")

        df = data.copy()
        result = df.copy()

        # TT:=(2*CLOSE+OPEN+HIGH+LOW)
        tt = 2 * df['Close'] + df['Open'] + df['High'] + df['Low']
        result['tt'] = tt

        # 动能价格:=100*(TT/EMA(TT,5)-1)
        ema_tt_5 = self._ema(tt, 5)
        momentum_price = 100 * (tt / (ema_tt_5 + 1e-10) - 1)
        result['momentum_price'] = momentum_price

        # 强动能1:=动能价格>0 AND 动能价格>REF(动能价格,1)
        strong_momentum1 = (momentum_price > 0) & (momentum_price > momentum_price.shift(1))
        result['strong_momentum1'] = strong_momentum1

        # 强动能: IF(强动能1,动能价格,0)
        result['strong_momentum'] = momentum_price.where(strong_momentum1, 0)

        # 中动能1:=动能价格>0 AND 动能价格<REF(动能价格,1)
        medium_momentum1 = (momentum_price > 0) & (momentum_price < momentum_price.shift(1))
        result['medium_momentum1'] = medium_momentum1

        # 中动能: IF(中动能1,动能价格,0)
        result['medium_momentum'] = momentum_price.where(medium_momentum1, 0)

        # 无动能1:=动能价格<0 AND 动能价格<REF(动能价格,1)
        weak_momentum1 = (momentum_price < 0) & (momentum_price < momentum_price.shift(1))
        result['weak_momentum1'] = weak_momentum1

        # 无动能: IF(无动能1,动能价格,0)
        result['no_momentum'] = momentum_price.where(weak_momentum1, 0)

        # 弱动能1:=动能价格<0 AND 动能价格>REF(动能价格,1)
        recovery_momentum1 = (momentum_price < 0) & (momentum_price > momentum_price.shift(1))
        result['recovery_momentum1'] = recovery_momentum1

        # 弱动能: IF(弱动能1,动能价格,0)
        result['recovery_momentum'] = momentum_price.where(recovery_momentum1, 0)

        return result
