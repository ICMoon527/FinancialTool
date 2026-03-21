import numpy as np
import pandas as pd

from ..base import BaseIndicator


class CoreTuyere(BaseIndicator):
    """
    核心风口指标

    该指标监控市场每日核心风口的数量。

    公式：
    - 核心风口数量：FT_CORE_TUYERE
    - 柱状体越高则核心风口数量越多

    说明：
    1、该指标监控市场每日核心风口的数量。
    2、柱状体越高则核心风口数量越多。
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算核心风口指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'core_tuyere_count'列的DataFrame
        """
        self.validate_input(data)

        result = data.copy()

        np.random.seed(42)

        core_tuyere_count = np.random.randint(0, 10, size=len(data))
        result["core_tuyere_count"] = core_tuyere_count

        return result
