import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SWI(BaseIndicator):
    """
    SWI（Schaff Trend Cycle），中文称为沙夫趋势周期。

    SWI指标是由Doug Schaff提出的，是一种结合了MACD和随机指标特点的趋势指标。
    SWI指标通过计算MACD的随机变化，来判断价格的趋势和超买超卖状态。

    计算公式：
    1. 计算MACD：MACD = EMA(Close, 12) - EMA(Close, 26)
    2. 计算Signal：Signal = EMA(MACD, 9)
    3. 计算Histogram：Histogram = MACD - Signal
    4. 计算Stochastic Histogram：stoch_hist = (Histogram - Lowest_Histogram(N)) / (Highest_Histogram(N) - Lowest_Histogram(N))
    5. 计算Smooth1：Smooth1 = EMA(stoch_hist, M)
    6. 计算Smooth2：Smooth2 = EMA(Smooth1, M)

    使用场景：
    - SWI > 0.75 时，为超买信号，可能回落
    - SWI < 0.25 时，为超卖信号，可能反弹
    - SWI上升时，表示多头力量增强，为买入信号
    - SWI下降时，表示空头力量增强，为卖出信号
    - SWI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号周期，默认9
    - stoch_period: 随机周期，默认10
    - smooth_period: 平滑周期，默认3

    输出参数：
    - MACD: MACD线
    - Signal: 信号线
    - Histogram: 柱状图
    - Stoch_Hist: 随机柱状图
    - Smooth1: 一次平滑
    - Smooth2: 二次平滑
    - SWI: 沙夫趋势周期
    - swi_overbought: SWI > 0.75信号
    - swi_oversold: SWI < 0.25信号
    - swi_rising: SWI上升信号
    - swi_falling: SWI下降信号
    - smooth1_above_smooth2: Smooth1在Smooth2上方信号
    - smooth1_below_smooth2: Smooth1在Smooth2下方信号
    - smooth1_cross_smooth2_up: Smooth1上穿Smooth2信号
    - smooth1_cross_smooth2_down: Smooth1下穿Smooth2信号

    注意事项：
    - SWI是趋势指标，在震荡市中会产生频繁的假信号
    - SWI在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：SWI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的SWI
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9, stoch_period=10, smooth_period=3):
        """
        初始化SWI指标

        Parameters
        ----------
        fast_period : int, default 12
            快速EMA周期
        slow_period : int, default 26
            慢速EMA周期
        signal_period : int, default 9
            信号周期
        stoch_period : int, default 10
            随机周期
        smooth_period : int, default 3
            平滑周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.stoch_period = stoch_period
        self.smooth_period = smooth_period

    def calculate(self, data):
        """
        计算SWI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SWI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        ema_fast = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['Signal'] = df['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']

        hist_min = df['Histogram'].rolling(window=self.stoch_period).min()
        hist_max = df['Histogram'].rolling(window=self.stoch_period).max()
        df['Stoch_Hist'] = (df['Histogram'] - hist_min) / (hist_max - hist_min)
        df['Smooth1'] = df['Stoch_Hist'].ewm(span=self.smooth_period, adjust=False).mean()
        df['Smooth2'] = df['Smooth1'].ewm(span=self.smooth_period, adjust=False).mean()
        df['SWI'] = df['Smooth2']

        df['swi_overbought'] = df['SWI'] > 0.75
        df['swi_oversold'] = df['SWI'] < 0.25
        df['swi_rising'] = df['SWI'] > df['SWI'].shift(1)
        df['swi_falling'] = df['SWI'] < df['SWI'].shift(1)
        df['smooth1_above_smooth2'] = df['Smooth1'] > df['Smooth2']
        df['smooth1_below_smooth2'] = df['Smooth1'] < df['Smooth2']
        df['smooth1_cross_smooth2_up'] = (df['Smooth1'] > df['Smooth2']) & (df['Smooth1'].shift(1) <= df['Smooth2'].shift(1))
        df['smooth1_cross_smooth2_down'] = (df['Smooth1'] < df['Smooth2']) & (df['Smooth1'].shift(1) >= df['Smooth2'].shift(1))

        return df[['MACD', 'Signal', 'Histogram', 'Stoch_Hist', 'Smooth1', 'Smooth2', 'SWI',
                   'swi_overbought', 'swi_oversold', 'swi_rising', 'swi_falling',
                   'smooth1_above_smooth2', 'smooth1_below_smooth2',
                   'smooth1_cross_smooth2_up', 'smooth1_cross_smooth2_down']]
