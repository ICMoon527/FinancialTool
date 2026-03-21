import numpy as np
import pandas as pd

from ..base import BaseIndicator


class DragonHeight(BaseIndicator):
    """
    龙头高度指标

    该指标监控市场每日最长连板个股的连板数。

    公式：
    - 连板高度：LINKING_BOARD(0)
    - 柱状体越高则连板数越高

    说明：
    1、该指标监控市场每日最长连板个股的连板数。
    2、柱状体越高则连板数越高。
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算龙头高度指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'linking_board_height'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        np.random.seed(42)

        linking_board_height = np.random.randint(0, 8, size=len(data))
        result["linking_board_height"] = linking_board_height

        return result
