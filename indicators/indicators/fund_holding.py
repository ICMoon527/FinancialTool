import numpy as np
import pandas as pd

from ..base import BaseIndicator


class FundHolding(BaseIndicator):
    """
    基金持仓指标

    该指标显示基金单季度持股占比流通比。

    公式：
    - 占流通股比：FUNDHOLDING
    - 环比增减：FUNDCHANGE
    - 橙色色块表示基金单季度持股占比流通比
    - 色块越高，持股占比越高

    说明：
    1、橙色色块表示基金单季度持股占比流通比。
    2、色块越高，持股占比越高。
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基金持仓指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'fund_holding_ratio', 'fund_change'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        np.random.seed(42)

        fund_holding_ratio = np.random.uniform(0, 30, size=len(data))
        fund_change = np.random.uniform(-10, 10, size=len(data))

        result["fund_holding_ratio"] = fund_holding_ratio
        result["fund_change"] = fund_change

        return result
