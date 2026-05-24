# -*- coding: utf-8 -*-
"""
===================================
EXPMA（指数移动平均线）指标
===================================

职责：
1. 计算快速线和慢速线两条指数移动平均线
2. 支持通过构造参数动态调整周期
"""

import pandas as pd
import numpy as np

from ..base import BaseIndicator


class EXPMA(BaseIndicator):
    """
    EXPMA（Exponential Moving Average），中文称为指数移动平均线。

    EXPMA 是一种加权移动平均线，对近期价格赋予更高的权重。
    本指标同时计算两条不同周期的 EXPMA 线：
    - EXPMA_Fast：快速线（默认 13 周期），对短期价格波动敏感
    - EXPMA_Slow：慢速线（默认 30 周期），反映中长期趋势

    计算公式：
    EXPMA = Close.ewm(span=N, adjust=False).mean()

    使用场景：
    - 快线上穿慢线（金叉）：看涨信号
    - 快线下穿慢线（死叉）：看跌信号
    - 价格在 EXPMA 上方运行：多头趋势
    - 价格在 EXPMA 下方运行：空头趋势

    输入参数：
    - data: DataFrame，必须包含 'Close' 列
    - fast_period: 快速线周期，默认 13
    - slow_period: 慢速线周期，默认 30

    输出参数：
    - EXPMA_Fast: 快速指数移动平均线
    - EXPMA_Slow: 慢速指数移动平均线
    """

    def __init__(self, fast_period: int = 13, slow_period: int = 30):
        """
        初始化 EXPMA 指标参数。

        Args:
            fast_period: 快速线周期，默认 13
            slow_period: 慢速线周期，默认 30
        """
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算 EXPMA 指标。

        Args:
            data: 包含 OHLCV 数据的输入 DataFrame，必须包含 'Close' 列

        Returns:
            添加了 EXPMA 相关列的 DataFrame：
            - EXPMA_Fast: 快速指数移动平均线
            - EXPMA_Slow: 慢速指数移动平均线
        """
        self.validate_input(data)

        result = data.copy()
        close = result["Close"]

        # 使用行业标准的指数平滑公式计算 EMA
        result["EXPMA_Fast"] = close.ewm(span=self.fast_period, adjust=False).mean()
        result["EXPMA_Slow"] = close.ewm(span=self.slow_period, adjust=False).mean()

        return result