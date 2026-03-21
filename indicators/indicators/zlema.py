import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ZLEMA(BaseIndicator):
    """
    ZLEMA（Zero Lag Exponential Moving Average），中文称为零滞后指数移动平均。

    ZLEMA指标是由John Ehlers提出的，是一种零滞后的指数移动平均。
    ZLEMA指标通过预先移动价格，来减少移动平均线的滞后性。

    计算公式：
    1. 计算Lag：Lag = (Period - 1) / 2
    2. 计算EMA_data：EMA_data = 2 × Close - Close.shift(Lag)
    3. 计算ZLEMA：ZLEMA = EMA(EMA_data, Period)

    使用场景：
    - ZLEMA上升时，表示上升趋势
    - ZLEMA下降时，表示下降趋势
    - 价格在ZLEMA上方时，为多头市场
    - 价格在ZLEMA下方时，为空头市场
    - 价格上穿ZLEMA时，为买入信号
    - 价格下穿ZLEMA时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认14

    输出参数：
    - EMA_Data: EMA数据
    - ZLEMA: 零滞后指数移动平均
    - price_above_zlema: 价格在ZLEMA上方信号
    - price_below_zlema: 价格在ZLEMA下方信号
    - zlema_rising: ZLEMA上升信号
    - zlema_falling: ZLEMA下降信号
    - price_cross_zlema_up: 价格上穿ZLEMA信号
    - price_cross_zlema_down: 价格下穿ZLEMA信号

    注意事项：
    - ZLEMA是零滞后移动平均，在震荡市中会产生频繁的假信号
    - ZLEMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和ZLEMA同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ZLEMA
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14):
        """
        初始化ZLEMA指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算ZLEMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ZLEMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        lag = int((self.period - 1) / 2)
        df['EMA_Data'] = 2 * df['Close'] - df['Close'].shift(lag)
        df['ZLEMA'] = df['EMA_Data'].ewm(span=self.period, adjust=False).mean()

        df['price_above_zlema'] = df['Close'] > df['ZLEMA']
        df['price_below_zlema'] = df['Close'] < df['ZLEMA']
        df['zlema_rising'] = df['ZLEMA'] > df['ZLEMA'].shift(1)
        df['zlema_falling'] = df['ZLEMA'] < df['ZLEMA'].shift(1)
        df['price_cross_zlema_up'] = (df['Close'] > df['ZLEMA']) & (df['Close'].shift(1) <= df['ZLEMA'].shift(1))
        df['price_cross_zlema_down'] = (df['Close'] < df['ZLEMA']) & (df['Close'].shift(1) >= df['ZLEMA'].shift(1))

        return df[['EMA_Data', 'ZLEMA',
                   'price_above_zlema', 'price_below_zlema',
                   'zlema_rising', 'zlema_falling',
                   'price_cross_zlema_up', 'price_cross_zlema_down']]
