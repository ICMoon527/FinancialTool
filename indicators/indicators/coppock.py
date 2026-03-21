import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Coppock(BaseIndicator):
    """
    Coppock（科波克曲线指标），中文称为科波克曲线指标。

    Coppock指标是由埃德温·科波克（Edwin Coppock）提出的，是一种长期动量指标。
    Coppock指标通过计算价格的长期变动率，来判断价格的长期趋势变化。

    计算公式：
    1. 计算14日变动率：ROC14 = (Close - Close_prev_14) / Close_prev_14 × 100
    2. 计算11日变动率：ROC11 = (Close - Close_prev_11) / Close_prev_11 × 100
    3. 计算Coppock = WMA(ROC14 + ROC11, 10)

    使用场景：
    - Coppock由负转正时，为买入信号
    - Coppock由正转负时，为卖出信号
    - Coppock上升时，表示动量增强，为买入信号
    - Coppock下降时，表示动量减弱，为卖出信号
    - Coppock创新高时，表示长期趋势向上
    - Coppock创新低时，表示长期趋势向下
    - Coppock在0以上时，为多头市场
    - Coppock在0以下时，为空头市场

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - roc_period1: 第一个ROC周期，默认14
    - roc_period2: 第二个ROC周期，默认11
    - wma_period: WMA周期，默认10

    输出参数：
    - ROC14: 14日变动率
    - ROC11: 11日变动率
    - Coppock: 科波克曲线指标
    - Coppock_MA: Coppock的移动平均线
    - coppock_positive: Coppock > 0信号
    - coppock_negative: Coppock < 0信号
    - coppock_cross_zero_up: Coppock上穿0线信号
    - coppock_cross_zero_down: Coppock下穿0线信号
    - coppock_rising: Coppock上升信号
    - coppock_falling: Coppock下降信号
    - coppock_new_high: Coppock创新高信号
    - coppock_new_low: Coppock创新低信号

    注意事项：
    - Coppock是长期指标，反应较慢
    - Coppock不适合短线操作
    - Coppock在长期趋势中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、MA）一起使用

    最佳实践建议：
    1. 长期使用：Coppock适合长期投资
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Coppock
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待2-3个交易日确认
    """

    def __init__(self, roc_period1=14, roc_period2=11, wma_period=10, ma_period=10):
        """
        初始化Coppock指标

        Parameters
        ----------
        roc_period1 : int, default 14
            第一个ROC周期
        roc_period2 : int, default 11
            第二个ROC周期
        wma_period : int, default 10
            WMA周期
        ma_period : int, default 10
            移动平均周期
        """
        self.roc_period1 = roc_period1
        self.roc_period2 = roc_period2
        self.wma_period = wma_period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算Coppock指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Coppock计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev1 = df['Close'].shift(self.roc_period1)
        close_prev2 = df['Close'].shift(self.roc_period2)

        df['ROC14'] = (df['Close'] - close_prev1) / close_prev1.replace(0, np.nan) * 100
        df['ROC11'] = (df['Close'] - close_prev2) / close_prev2.replace(0, np.nan) * 100

        roc_sum = df['ROC14'] + df['ROC11']

        weights = np.arange(1, self.wma_period + 1)
        df['Coppock'] = roc_sum.rolling(window=self.wma_period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

        df['Coppock_MA'] = df['Coppock'].rolling(window=self.ma_period).mean()

        df['coppock_positive'] = df['Coppock'] > 0
        df['coppock_negative'] = df['Coppock'] < 0
        df['coppock_cross_zero_up'] = (df['Coppock'] > 0) & (df['Coppock'].shift(1) <= 0)
        df['coppock_cross_zero_down'] = (df['Coppock'] < 0) & (df['Coppock'].shift(1) >= 0)
        df['coppock_rising'] = df['Coppock'] > df['Coppock'].shift(1)
        df['coppock_falling'] = df['Coppock'] < df['Coppock'].shift(1)
        df['coppock_new_high'] = df['Coppock'] == df['Coppock'].rolling(window=self.ma_period).max()
        df['coppock_new_low'] = df['Coppock'] == df['Coppock'].rolling(window=self.ma_period).min()

        return df[['ROC14', 'ROC11', 'Coppock', 'Coppock_MA',
                   'coppock_positive', 'coppock_negative', 'coppock_cross_zero_up', 'coppock_cross_zero_down',
                   'coppock_rising', 'coppock_falling', 'coppock_new_high', 'coppock_new_low']]
