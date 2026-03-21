import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TSI(BaseIndicator):
    """
    TSI（True Strength Index），中文称为真实强度指标。

    TSI指标是由William Blau提出的，是一种动量指标。
    TSI指标通过计算价格变动的双重平滑，来判断价格的超买超卖状态和趋势变化。

    计算公式：
    1. 计算Momentum：Momentum = Close - Close_prev
    2. 计算AbsMomentum：AbsMomentum = |Momentum|
    3. 计算DoubleSmoothedMomentum：DoubleSmoothedMomentum = EMA(EMA(Momentum, N), M)
    4. 计算DoubleSmoothedAbsMomentum：DoubleSmoothedAbsMomentum = EMA(EMA(AbsMomentum, N), M)
    5. 计算TSI：TSI = (DoubleSmoothedMomentum / DoubleSmoothedAbsMomentum) × 100
    6. 计算Signal：Signal = EMA(TSI, P)

    使用场景：
    - TSI > 0 时，表示多头市场
    - TSI < 0 时，表示空头市场
    - TSI上升时，表示多头力量增强，为买入信号
    - TSI下降时，表示空头力量增强，为卖出信号
    - TSI上穿Signal时，为买入信号（金叉）
    - TSI下穿Signal时，为卖出信号（死叉）
    - TSI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - long_period: 长期周期，默认25
    - short_period: 短期周期，默认13
    - signal_period: 信号周期，默认13

    输出参数：
    - Momentum: 动量
    - AbsMomentum: 绝对动量
    - DoubleSmoothedMomentum: 双重平滑动量
    - DoubleSmoothedAbsMomentum: 双重平滑绝对动量
    - TSI: 真实强度指标
    - Signal: 信号线
    - tsi_positive: TSI > 0信号
    - tsi_negative: TSI < 0信号
    - tsi_rising: TSI上升信号
    - tsi_falling: TSI下降信号
    - tsi_above_signal: TSI在Signal上方信号
    - tsi_below_signal: TSI在Signal下方信号
    - tsi_cross_signal_up: TSI上穿Signal信号
    - tsi_cross_signal_down: TSI下穿Signal信号

    注意事项：
    - TSI是动量指标，在震荡市中会产生频繁的假信号
    - TSI在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：TSI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的TSI
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, long_period=25, short_period=13, signal_period=13):
        """
        初始化TSI指标

        Parameters
        ----------
        long_period : int, default 25
            长期周期
        short_period : int, default 13
            短期周期
        signal_period : int, default 13
            信号周期
        """
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算TSI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含TSI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['Momentum'] = df['Close'].diff()
        df['AbsMomentum'] = abs(df['Momentum'])
        df['DoubleSmoothedMomentum'] = df['Momentum'].ewm(span=self.long_period, adjust=False).mean().ewm(span=self.short_period, adjust=False).mean()
        df['DoubleSmoothedAbsMomentum'] = df['AbsMomentum'].ewm(span=self.long_period, adjust=False).mean().ewm(span=self.short_period, adjust=False).mean()
        df['TSI'] = (df['DoubleSmoothedMomentum'] / df['DoubleSmoothedAbsMomentum']) * 100
        df['Signal'] = df['TSI'].ewm(span=self.signal_period, adjust=False).mean()

        df['tsi_positive'] = df['TSI'] > 0
        df['tsi_negative'] = df['TSI'] < 0
        df['tsi_rising'] = df['TSI'] > df['TSI'].shift(1)
        df['tsi_falling'] = df['TSI'] < df['TSI'].shift(1)
        df['tsi_above_signal'] = df['TSI'] > df['Signal']
        df['tsi_below_signal'] = df['TSI'] < df['Signal']
        df['tsi_cross_signal_up'] = (df['TSI'] > df['Signal']) & (df['TSI'].shift(1) <= df['Signal'].shift(1))
        df['tsi_cross_signal_down'] = (df['TSI'] < df['Signal']) & (df['TSI'].shift(1) >= df['Signal'].shift(1))

        return df[['Momentum', 'AbsMomentum', 'DoubleSmoothedMomentum', 'DoubleSmoothedAbsMomentum', 'TSI', 'Signal',
                   'tsi_positive', 'tsi_negative', 'tsi_rising', 'tsi_falling',
                   'tsi_above_signal', 'tsi_below_signal',
                   'tsi_cross_signal_up', 'tsi_cross_signal_down']]
