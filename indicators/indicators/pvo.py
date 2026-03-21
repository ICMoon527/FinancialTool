import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PVO(BaseIndicator):
    """
    PVO（Percentage Volume Oscillator），中文称为成交量百分比震荡器。

    PVO指标是一种基于成交量的动量指标，类似于MACD但使用成交量而不是价格。
    PVO指标通过计算成交量的EMA差值百分比，来判断成交量的变化和趋势。

    计算公式：
    1. 计算EMA_Fast：EMA_Fast = EMA(Volume, F)
    2. 计算EMA_Slow：EMA_Slow = EMA(Volume, S)
    3. 计算PVO：PVO = (EMA_Fast - EMA_Slow) / EMA_Slow × 100
    4. 计算Signal：Signal = EMA(PVO, P)
    5. 计算Histogram：Histogram = PVO - Signal

    使用场景：
    - PVO > 0 时，表示短期成交量大于长期成交量，为多头信号
    - PVO < 0 时，表示短期成交量小于长期成交量，为空头信号
    - PVO上升时，表示成交量增加，为买入信号
    - PVO下降时，表示成交量减少，为卖出信号
    - PVO上穿Signal时，为买入信号（金叉）
    - PVO下穿Signal时，为卖出信号（死叉）
    - Histogram > 0 时，表示动量增加
    - Histogram < 0 时，表示动量减少

    输入参数：
    - data: DataFrame，必须包含'Volume'列
    - fast_period: 快速周期，默认12
    - slow_period: 慢速周期，默认26
    - signal_period: 信号周期，默认9

    输出参数：
    - EMA_Fast: 快速EMA
    - EMA_Slow: 慢速EMA
    - PVO: 成交量百分比震荡器
    - Signal: 信号线
    - Histogram: 柱状图
    - pvo_positive: PVO > 0信号
    - pvo_negative: PVO < 0信号
    - pvo_rising: PVO上升信号
    - pvo_falling: PVO下降信号
    - pvo_above_signal: PVO在Signal上方信号
    - pvo_below_signal: PVO在Signal下方信号
    - pvo_cross_signal_up: PVO上穿Signal信号
    - pvo_cross_signal_down: PVO下穿Signal信号
    - histogram_positive: Histogram > 0信号
    - histogram_negative: Histogram < 0信号

    注意事项：
    - PVO是成交量动量指标，在震荡市中会产生频繁的假信号
    - PVO在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：PVO和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的PVO
    4. 量价配合：结合价格变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化PVO指标

        Parameters
        ----------
        fast_period : int, default 12
            快速周期
        slow_period : int, default 26
            慢速周期
        signal_period : int, default 9
            信号周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算PVO指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PVO计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA_Fast'] = df['Volume'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['Volume'].ewm(span=self.slow_period, adjust=False).mean()
        df['PVO'] = ((df['EMA_Fast'] - df['EMA_Slow']) / df['EMA_Slow']) * 100
        df['Signal'] = df['PVO'].ewm(span=self.signal_period, adjust=False).mean()
        df['Histogram'] = df['PVO'] - df['Signal']

        df['pvo_positive'] = df['PVO'] > 0
        df['pvo_negative'] = df['PVO'] < 0
        df['pvo_rising'] = df['PVO'] > df['PVO'].shift(1)
        df['pvo_falling'] = df['PVO'] < df['PVO'].shift(1)
        df['pvo_above_signal'] = df['PVO'] > df['Signal']
        df['pvo_below_signal'] = df['PVO'] < df['Signal']
        df['pvo_cross_signal_up'] = (df['PVO'] > df['Signal']) & (df['PVO'].shift(1) <= df['Signal'].shift(1))
        df['pvo_cross_signal_down'] = (df['PVO'] < df['Signal']) & (df['PVO'].shift(1) >= df['Signal'].shift(1))
        df['histogram_positive'] = df['Histogram'] > 0
        df['histogram_negative'] = df['Histogram'] < 0

        return df[['EMA_Fast', 'EMA_Slow', 'PVO', 'Signal', 'Histogram',
                   'pvo_positive', 'pvo_negative', 'pvo_rising', 'pvo_falling',
                   'pvo_above_signal', 'pvo_below_signal',
                   'pvo_cross_signal_up', 'pvo_cross_signal_down',
                   'histogram_positive', 'histogram_negative']]
