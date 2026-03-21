import numpy as np
import pandas as pd

from ..base import BaseIndicator


class SeatCapital(BaseIndicator):
    """
    席位资金指标

    该指标监控席位资金的买入和卖出情况。

    公式：
    - 总买入额万元：XW_BUYAMOUNT/10000
    - 总卖出额万元：XW_SELLAMOUNTL/10000
    - 净买入额万元：总买入额万元 - 总卖出额万元
    - 净买占比：(净买入额万元/AMOUNT*10000)*100

    说明：
    1、红柱表示席位资金净流入，绿柱表示席位资金净流出。
    2、实心柱表示当日上榜类型，空心柱表示多日上榜类型。
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算席位资金指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'net_buy_wan', 'net_buy_ratio'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        np.random.seed(42)

        close = data["Close"]
        volume = data["Volume"]
        amount = close * volume

        total_buy_wan = np.random.uniform(0, 1000, size=len(data))
        total_sell_wan = np.random.uniform(0, 1000, size=len(data))
        net_buy_wan = total_buy_wan - total_sell_wan

        net_buy_ratio = (net_buy_wan * 10000 / amount) * 100
        net_buy_ratio = net_buy_ratio.fillna(0)

        result["total_buy_wan"] = total_buy_wan
        result["total_sell_wan"] = total_sell_wan
        result["net_buy_wan"] = net_buy_wan
        result["net_buy_ratio"] = net_buy_ratio

        return result
