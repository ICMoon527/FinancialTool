import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TurnoverHeat(BaseIndicator):
    """
    换手热度指标
    监控个股换手率活跃度
    灰色柱体表示当日换手不活跃
    金色柱体表示当日换手比较活跃
    红色柱体表示当日换手非常活跃
    紫色柱体表示当日换手高度活跃
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Turnover Heat indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 换手率: VOL*100/(FINANCE(7)/100)
        # 使用简化的换手率计算
        turnover_rate = (df['Volume'] / df['Volume'].rolling(window=100).mean()) * 100
        result['turnover_rate'] = turnover_rate

        # 不活跃: 换手率<3
        result['inactive'] = turnover_rate < 3

        # 比较活跃: 换手率>=3 AND 换手率<7
        result['moderately_active'] = (turnover_rate >= 3) & (turnover_rate < 7)

        # 非常活跃: 换手率>=7 AND 换手率<10
        result['very_active'] = (turnover_rate >= 7) & (turnover_rate < 10)

        # 高度活跃: 换手率>=10
        result['highly_active'] = turnover_rate >= 10

        # MA5: MA(换手率,5)
        result['ma5'] = self._sma(turnover_rate, 5)

        # MA10: MA(换手率,10)
        result['ma10'] = self._sma(turnover_rate, 10)

        return result
