import pandas as pd
import numpy as np
from ..base import BaseIndicator


class AMV(BaseIndicator):
    """
    AMV（成本均线），中文称为成本均线。

    AMV指标是一种量价加权移动平均线，对成交量大的价格赋予更高的权重。
    AMV指标通过成交量加权，来反应市场的平均成本。

    计算公式：
    AMV = sum(Close × Volume, N) / sum(Volume, N)

    使用场景：
    - 价格在AMV上方时，为多头市场
    - 价格在AMV下方时，为空头市场
    - AMV上升时，表示上升趋势
    - AMV下降时，表示下降趋势
    - 短期AMV上穿长期AMV时，为买入信号（金叉）
    - 短期AMV下穿长期AMV时，为卖出信号（死叉）

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period: 计算周期，默认10

    输出参数：
    - AMV: 成本均线
    - price_above_amv: 价格在AMV上方信号
    - price_below_amv: 价格在AMV下方信号
    - amv_rising: AMV上升信号
    - amv_falling: AMV下降信号

    注意事项：
    - AMV是量价加权趋势指标，在震荡市中会产生频繁的假信号
    - AMV在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和AMV同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的AMV
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化AMV指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算AMV指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含AMV计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_volume = df['Close'] * df['Volume']
        sum_close_volume = close_volume.rolling(window=self.period).sum()
        sum_volume = df['Volume'].rolling(window=self.period).sum()
        df['AMV'] = sum_close_volume / sum_volume

        df['price_above_amv'] = df['Close'] > df['AMV']
        df['price_below_amv'] = df['Close'] < df['AMV']
        df['amv_rising'] = df['AMV'] > df['AMV'].shift(1)
        df['amv_falling'] = df['AMV'] < df['AMV'].shift(1)

        return df[['AMV',
                   'price_above_amv', 'price_below_amv',
                   'amv_rising', 'amv_falling']]
