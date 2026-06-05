# -*- coding: utf-8 -*-
"""
布林带（Bollinger Bands）指标计算器。

布林带由三条线组成：
- 中轨（Middle）：Close 的 N 日简单移动平均（SMA）
- 上轨（Upper）：中轨 + K × N 日标准差
- 下轨（Lower）：中轨 - K × N 日标准差

默认参数：N=20, K=2.0
"""

import pandas as pd

from indicators.base import BaseIndicator


class Bollinger(BaseIndicator):
    """
    布林带指标。

    参数：
    - n: 移动平均周期（默认 20）
    - k: 标准差倍数（默认 2.0）
    """

    def __init__(self, n: int = 20, k: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.k = k

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带三条线。"""
        self.validate_input(data)

        close = data["Close"]
        result = data.copy()

        # 中轨：N 日简单移动平均
        middle = close.rolling(self.n).mean()

        # N 日滚动标准差
        std = close.rolling(self.n).std()

        # 上轨 / 下轨
        upper = middle + self.k * std
        lower = middle - self.k * std

        result["bollinger_middle"] = middle
        result["bollinger_upper"] = upper
        result["bollinger_lower"] = lower

        return result