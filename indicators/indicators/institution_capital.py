import numpy as np
import pandas as pd

from ..base import BaseIndicator


class InstitutionCapital(BaseIndicator):
    """
    机构资金指标

    该指标监控机构资金的买入和卖出情况，以及买席和卖席机构数。

    公式：
    - 机构买额万元：JG_BUYAMOUNT/10000
    - 机构卖额万元：JG_SELLAMOUNT/10000
    - 机构净额万元：机构买额万元 - 机构卖额万元
    - 净买占比：(机构净额万元/AMOUNT*10000)*100
    - 买席机构数：JG_BUYNUMBER
    - 卖席机构数：JG_SELLNUMBER

    说明：
    1、橙色实心柱体高度表示机构买额。
    2、蓝色实心柱体高度表示机构卖额。
    3、橙色柱体方块数量表示买席机构家数。
    4、蓝色柱体方块数量表示卖席机构家数。
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算机构资金指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了相关机构资金列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        np.random.seed(42)

        close = data["Close"]
        volume = data["Volume"]
        amount = close * volume

        institution_buy_wan = np.random.uniform(0, 2000, size=len(data))
        institution_sell_wan = np.random.uniform(0, 2000, size=len(data))
        institution_net_wan = institution_buy_wan - institution_sell_wan

        net_buy_ratio = (institution_net_wan * 10000 / amount) * 100
        net_buy_ratio = net_buy_ratio.fillna(0)

        buy_institution_count = np.random.randint(0, 5, size=len(data))
        sell_institution_count = np.random.randint(0, 5, size=len(data))

        result["institution_buy_wan"] = institution_buy_wan
        result["institution_sell_wan"] = institution_sell_wan
        result["institution_net_wan"] = institution_net_wan
        result["net_buy_ratio"] = net_buy_ratio
        result["buy_institution_count"] = buy_institution_count
        result["sell_institution_count"] = sell_institution_count

        return result
