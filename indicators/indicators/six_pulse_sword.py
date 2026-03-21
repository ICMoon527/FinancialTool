import numpy as np
import pandas as pd

from ..base import BaseIndicator


class SixPulseSword(BaseIndicator):
    """
    六脉神剑指标

    该指标监控连续阳线的情况。

    公式：
    - GT：BARSLASTCOUNT(C > O)
    - 连续6阳：GT = 6

    说明：
    监控连续阳线的情况，连续6根阳线表示强势。
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算六脉神剑指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            添加了'six_consecutive_yang'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]
        open_ = data["Open"]

        is_yang = close > open_

        consecutive_yang = pd.Series(0, index=data.index)
        count = 0
        for i in range(len(data)):
            if is_yang.iloc[i]:
                count += 1
            else:
                count = 0
            consecutive_yang.iloc[i] = count

        six_consecutive_yang = consecutive_yang == 6

        result["consecutive_yang_count"] = consecutive_yang
        result["six_consecutive_yang"] = six_consecutive_yang

        return result
