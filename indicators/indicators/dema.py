import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DEMA(BaseIndicator):
    """
    DEMA（Double Exponential Moving Average）指标，中文称为双指数移动平均指标。

    DEMA指标是由帕特里克·穆洛伊（Patrick Mulloy）提出的，是一种基于EMA的趋势指标。
    DEMA指标通过计算双指数移动平均，来减少滞后性，同时保持平滑性。

    计算公式：
    1. 计算EMA1：EMA1 = EMA(Close, N)
    2. 计算EMA2：EMA2 = EMA(EMA1, N)
    3. 计算DEMA：DEMA = 2 × EMA1 - EMA2

    使用场景：
    - DEMA上升时，表示上升趋势，为买入信号
    - DEMA下降时，表示下降趋势，为卖出信号
    - 价格上穿DEMA时，为买入信号
    - 价格下穿DEMA时，为卖出信号
    - DEMA在价格上方时，为阻力位
    - DEMA在价格下方时，为支撑位
    - DEMA比EMA反应更快
    - DEMA比EMA滞后更少

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认10

    输出参数：
    - EMA1: 第一次EMA
    - EMA2: 第二次EMA
    - DEMA: 双指数移动平均
    - close_above_dema: 价格在DEMA上方信号
    - close_below_dema: 价格在DEMA下方信号
    - dema_rising: DEMA上升信号
    - dema_falling: DEMA下降信号
    - dema_cross_up: 价格上穿DEMA信号
    - dema_cross_down: 价格下穿DEMA信号
    - dema_above_ema1: DEMA在EMA1上方信号
    - dema_below_ema1: DEMA在EMA1下方信号

    注意事项：
    - DEMA是趋势指标，在震荡市中会产生频繁的假信号
    - DEMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - DEMA反应较快，适合中短线操作

    最佳实践建议：
    1. 趋势确认：DEMA和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DEMA
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化DEMA指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算DEMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DEMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA1'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['EMA2'] = df['EMA1'].ewm(span=self.period, adjust=False).mean()

        df['DEMA'] = 2 * df['EMA1'] - df['EMA2']

        df['close_above_dema'] = df['Close'] > df['DEMA']
        df['close_below_dema'] = df['Close'] < df['DEMA']
        df['dema_rising'] = df['DEMA'] > df['DEMA'].shift(1)
        df['dema_falling'] = df['DEMA'] < df['DEMA'].shift(1)
        df['dema_cross_up'] = (df['Close'] > df['DEMA']) & (df['Close'].shift(1) <= df['DEMA'].shift(1))
        df['dema_cross_down'] = (df['Close'] < df['DEMA']) & (df['Close'].shift(1) >= df['DEMA'].shift(1))
        df['dema_above_ema1'] = df['DEMA'] > df['EMA1']
        df['dema_below_ema1'] = df['DEMA'] < df['EMA1']

        return df[['EMA1', 'EMA2', 'DEMA',
                   'close_above_dema', 'close_below_dema', 'dema_rising', 'dema_falling',
                   'dema_cross_up', 'dema_cross_down', 'dema_above_ema1', 'dema_below_ema1']]
