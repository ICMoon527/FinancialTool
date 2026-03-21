import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TEMA(BaseIndicator):
    """
    TEMA（Triple Exponential Moving Average）指标，中文称为三重指数移动平均指标。

    TEMA指标是由帕特里克·穆洛伊（Patrick Mulloy）提出的，是一种基于EMA的趋势指标。
    TEMA指标通过计算三重指数移动平均，来减少滞后性，同时保持平滑性。

    计算公式：
    1. 计算EMA1：EMA1 = EMA(Close, N)
    2. 计算EMA2：EMA2 = EMA(EMA1, N)
    3. 计算EMA3：EMA3 = EMA(EMA2, N)
    4. 计算TEMA：TEMA = 3 × EMA1 - 3 × EMA2 + EMA3

    使用场景：
    - TEMA上升时，表示上升趋势，为买入信号
    - TEMA下降时，表示下降趋势，为卖出信号
    - 价格上穿TEMA时，为买入信号
    - 价格下穿TEMA时，为卖出信号
    - TEMA在价格上方时，为阻力位
    - TEMA在价格下方时，为支撑位
    - TEMA比EMA反应更快
    - TEMA比EMA滞后更少

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认14

    输出参数：
    - EMA1: 第一次EMA
    - EMA2: 第二次EMA
    - EMA3: 第三次EMA
    - TEMA: 三重指数移动平均
    - close_above_tema: 价格在TEMA上方信号
    - close_below_tema: 价格在TEMA下方信号
    - tema_rising: TEMA上升信号
    - tema_falling: TEMA下降信号
    - tema_cross_up: 价格上穿TEMA信号
    - tema_cross_down: 价格下穿TEMA信号
    - tema_above_ema1: TEMA在EMA1上方信号
    - tema_below_ema1: TEMA在EMA1下方信号

    注意事项：
    - TEMA是趋势指标，在震荡市中会产生频繁的假信号
    - TEMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - TEMA反应较快，适合中短线操作

    最佳实践建议：
    1. 趋势确认：TEMA和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的TEMA
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14):
        """
        初始化TEMA指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算TEMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含TEMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA1'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['EMA2'] = df['EMA1'].ewm(span=self.period, adjust=False).mean()
        df['EMA3'] = df['EMA2'].ewm(span=self.period, adjust=False).mean()

        df['TEMA'] = 3 * df['EMA1'] - 3 * df['EMA2'] + df['EMA3']

        df['close_above_tema'] = df['Close'] > df['TEMA']
        df['close_below_tema'] = df['Close'] < df['TEMA']
        df['tema_rising'] = df['TEMA'] > df['TEMA'].shift(1)
        df['tema_falling'] = df['TEMA'] < df['TEMA'].shift(1)
        df['tema_cross_up'] = (df['Close'] > df['TEMA']) & (df['Close'].shift(1) <= df['TEMA'].shift(1))
        df['tema_cross_down'] = (df['Close'] < df['TEMA']) & (df['Close'].shift(1) >= df['TEMA'].shift(1))
        df['tema_above_ema1'] = df['TEMA'] > df['EMA1']
        df['tema_below_ema1'] = df['TEMA'] < df['EMA1']

        return df[['EMA1', 'EMA2', 'EMA3', 'TEMA',
                   'close_above_tema', 'close_below_tema', 'tema_rising', 'tema_falling',
                   'tema_cross_up', 'tema_cross_down', 'tema_above_ema1', 'tema_below_ema1']]
