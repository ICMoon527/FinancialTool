import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MoneyMakingEffect(BaseIndicator):
    """
    赚钱效应指标
    今日赚钱效应：统计上一个交易日涨幅>=5%的股票在今天的平均涨幅
    红柱表示上涨，绿柱表示下跌
    黄色线表示5日赚钱效应平均值
    紫色线表示20日赚钱效应平均值
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Money Making Effect indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 计算日收益率作为赚钱效应的代理
        daily_return = (df['Close'] / df['Close'].shift(1) - 1) * 100
        
        # 赚钱效应: FT_MAKEMONEYEFFECT proxy
        # 用个股的价格动量作为赚钱效应的近似
        result['money_making_effect'] = daily_return * 5

        # 红柱绿柱
        result['red_bar'] = result['money_making_effect'] >= 0
        result['green_bar'] = result['money_making_effect'] < 0

        # D5日平均: MA(赚钱效应,5)
        result['ma5_effect'] = self._sma(result['money_making_effect'], 5)

        # D20日平均: MA(赚钱效应,20)
        result['ma20_effect'] = self._sma(result['money_making_effect'], 20)

        return result
