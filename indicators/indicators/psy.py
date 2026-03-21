import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PSY(BaseIndicator):
    """
    PSY（Psychological Line）指标，中文称为心理线指标。

    PSY指标是一种衡量市场投资者心理情绪的技术指标，通过计算一定周期内上涨日数占总日数的比例，
    来判断市场的超买超卖状态和投资者的心理预期。

    计算公式：
    PSY = (N日内上涨日数 / N) × 100
    其中：
    - 上涨日：当日收盘价 > 前一日收盘价
    - N：计算周期，通常为10日、12日或20日

    使用场景：
    - PSY在25-75之间为正常区间，市场处于平衡状态
    - PSY > 75 时，处于超买区间，可能回落
    - PSY < 25 时，处于超卖区间，可能反弹
    - PSY > 90 时，严重超买，行情可能反转
    - PSY < 10 时，严重超卖，行情可能反转
    - PSY由低往上穿越25时，为买入信号
    - PSY由高往下跌破75时，为卖出信号
    - PSY在高位连续两次出现峰值时，为卖出信号
    - PSY在低位连续两次出现谷底时，为买入信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认12

    输出参数：
    - PSY: 心理线指标值
    - up_days: 上涨日数
    - overbought: 超买信号
    - oversold: 超卖信号
    - cross_25_up: 上穿25信号
    - cross_75_down: 下穿75信号

    注意事项：
    - PSY是短线指标，信号频繁
    - PSY在震荡市中表现较好
    - 超买超卖阈值需要根据市场特性调整
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 双重确认：等待PSY连续2-3天处于超买/超卖区再行动
    2. 趋势过滤：先判断大趋势，顺势操作
    3. 组合使用：与成交量指标配合使用
    4. 多周期使用：同时使用日线和周线的PSY
    5. 时间过滤：避免在重要数据发布前操作
    6. 平滑处理：使用PSY均线减少噪音
    7. 历史比较：比较当前PSY与历史PSY水平
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, period=12):
        """
        初始化PSY指标

        Parameters
        ----------
        period : int, default 12
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算PSY指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PSY计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        up_day = (df['Close'] > close_prev).astype(int)

        df['up_days'] = up_day.rolling(window=self.period).sum()
        df['PSY'] = df['up_days'] / self.period * 100

        df['overbought'] = df['PSY'] > 75
        df['oversold'] = df['PSY'] < 25

        df['cross_25_up'] = (df['PSY'] > 25) & (df['PSY'].shift(1) <= 25)
        df['cross_75_down'] = (df['PSY'] < 75) & (df['PSY'].shift(1) >= 75)

        return df[['PSY', 'up_days', 'overbought', 'oversold', 'cross_25_up', 'cross_75_down']]
