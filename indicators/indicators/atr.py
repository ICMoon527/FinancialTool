import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ATR(BaseIndicator):
    """
    ATR（Average True Range）指标，中文称为平均真实波幅指标。

    ATR指标由威尔斯·怀尔德（Welles Wilder）于1978年提出，是一种衡量市场波动性的技术指标。
    ATR指标不直接指示价格方向，而是衡量价格波动的剧烈程度，可以用于设置止损、判断市场状态等。

    计算公式：
    1. 计算真实波幅（TR）：
       TR = Max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    2. 计算平均真实波幅（ATR）：
       - 初始ATR = 前N日TR的简单移动平均
       - 后续ATR = (前一日ATR × (N-1) + 当日TR) / N

    使用场景：
    - ATR值越大，市场波动性越大
    - ATR值越小，市场波动性越小
    - ATR可以用于设置止损位，通常设置为1-2倍ATR
    - ATR可以用于判断市场状态，ATR突然增大可能意味着行情即将启动
    - ATR可以用于头寸管理，波动性大的品种减少仓位
    - ATR可以用于趋势确认，价格突破配合ATR放大更可靠

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14

    输出参数：
    - TR: 真实波幅
    - ATR: 平均真实波幅
    - atr_ratio: ATR与价格的比率
    - atr_increasing: ATR上升信号
    - atr_decreasing: ATR下降信号

    注意事项：
    - ATR是绝对值指标，不同价格的股票ATR不可直接比较
    - ATR不指示价格方向，只衡量波动性
    - ATR在价格跳空时会大幅增加
    - 周期参数可以根据市场特性调整，14天是最常用的
    - 建议结合其他指标（如MA、MACD）一起使用

    最佳实践建议：
    1. 相对化处理：使用ATR/价格比率进行不同股票比较
    2. 多重周期：同时使用短期和长期ATR
    3. 止损设置：使用ATR设置动态止损位
    4. 头寸管理：根据ATR调整仓位大小
    5. 突破确认：价格突破时配合ATR放大确认
    6. 趋势判断：ATR上升配合价格趋势更可靠
    7. 波动率过滤：在ATR过低时避免交易
    8. 历史比较：比较当前ATR与历史ATR水平
    """

    def __init__(self, period=14):
        """
        初始化ATR指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算ATR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ATR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))

        df['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        df['ATR'] = df['TR'].rolling(window=self.period).mean()

        for i in range(self.period, len(df)):
            df['ATR'].iloc[i] = (df['ATR'].iloc[i-1] * (self.period - 1) + df['TR'].iloc[i]) / self.period

        df['atr_ratio'] = df['ATR'] / df['Close'] * 100

        df['atr_increasing'] = df['ATR'] > df['ATR'].shift(1)
        df['atr_decreasing'] = df['ATR'] < df['ATR'].shift(1)

        return df[['TR', 'ATR', 'atr_ratio', 'atr_increasing', 'atr_decreasing']]
