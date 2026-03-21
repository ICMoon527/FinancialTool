import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CHO(BaseIndicator):
    """
    CHO（佳庆指标），中文称为佳庆指标。

    CHO指标是由Marc Chaikin提出的，是一种量价指标。
    CHO指标结合了累积派发线和移动平均线的特点，来判断资金的流动。

    计算公式：
    1. 计算MFM（资金流向乘数）：MFM = ((Close - Low) - (High - Close)) / (High - Low)
    2. 计算MFV（资金流向量）：MFV = MFM × Volume
    3. 计算ADL（累积派发线）：ADL = sum(MFV)
    4. 计算CHO：CHO = EMA(ADL, F) - EMA(ADL, S)

    使用场景：
    - CHO > 0 时，表示资金流入，为多头信号
    - CHO < 0 时，表示资金流出，为空头信号
    - CHO上升时，表示资金流入增加，为买入信号
    - CHO下降时，表示资金流出增加，为卖出信号
    - CHO由负转正时，为买入信号
    - CHO由正转负时，为卖出信号
    - CHO与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列
    - fast_period: 快速EMA周期，默认3
    - slow_period: 慢速EMA周期，默认10

    输出参数：
    - MFM: 资金流向乘数
    - MFV: 资金流向量
    - ADL: 累积派发线
    - EMA_Fast: 快速EMA
    - EMA_Slow: 慢速EMA
    - CHO: 佳庆指标
    - CHO_MA: CHO的移动平均线
    - cho_positive: CHO > 0信号
    - cho_negative: CHO < 0信号
    - cho_rising: CHO上升信号
    - cho_falling: CHO下降信号
    - cho_cross_zero_up: CHO上穿0线信号
    - cho_cross_zero_down: CHO下穿0线信号
    - cho_above_ma: CHO在MA上方信号
    - cho_below_ma: CHO在MA下方信号

    注意事项：
    - CHO是量价指标，反应资金流动
    - CHO在震荡市中会产生频繁的假信号
    - CHO在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：CHO和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的CHO
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=3, slow_period=10, ma_period=10):
        """
        初始化CHO指标

        Parameters
        ----------
        fast_period : int, default 3
            快速EMA周期
        slow_period : int, default 10
            慢速EMA周期
        ma_period : int, default 10
            移动平均周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算CHO指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CHO计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low.replace(0, np.nan)
        df['MFM'] = df['MFM'].fillna(0)
        df['MFV'] = df['MFM'] * df['Volume']
        df['ADL'] = df['MFV'].cumsum()

        df['EMA_Fast'] = df['ADL'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['ADL'].ewm(span=self.slow_period, adjust=False).mean()
        df['CHO'] = df['EMA_Fast'] - df['EMA_Slow']
        df['CHO_MA'] = df['CHO'].rolling(window=self.ma_period).mean()

        df['cho_positive'] = df['CHO'] > 0
        df['cho_negative'] = df['CHO'] < 0
        df['cho_rising'] = df['CHO'] > df['CHO'].shift(1)
        df['cho_falling'] = df['CHO'] < df['CHO'].shift(1)
        df['cho_cross_zero_up'] = (df['CHO'] > 0) & (df['CHO'].shift(1) <= 0)
        df['cho_cross_zero_down'] = (df['CHO'] < 0) & (df['CHO'].shift(1) >= 0)
        df['cho_above_ma'] = df['CHO'] > df['CHO_MA']
        df['cho_below_ma'] = df['CHO'] < df['CHO_MA']

        return df[['MFM', 'MFV', 'ADL', 'EMA_Fast', 'EMA_Slow', 'CHO', 'CHO_MA',
                   'cho_positive', 'cho_negative',
                   'cho_rising', 'cho_falling',
                   'cho_cross_zero_up', 'cho_cross_zero_down',
                   'cho_above_ma', 'cho_below_ma']]
