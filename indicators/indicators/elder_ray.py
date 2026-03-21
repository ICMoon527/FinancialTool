import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ElderRay(BaseIndicator):
    """
    Elder Ray（埃尔德射线指标），中文称为埃尔德射线指标。

    Elder Ray指标是由亚历山大·埃尔德（Alexander Elder）提出的，是一种趋势指标。
    Elder Ray指标通过计算价格与指数移动平均线的差值，来判断价格的趋势和买卖力量。

    计算公式：
    1. 计算EMA：EMA = EMA(Close, N)
    2. 计算Bull Power（多头力量）：Bull_Power = High - EMA
    3. 计算Bear Power（空头力量）：Bear_Power = Low - EMA

    使用场景：
    - Bull Power > 0 且上升时，表示多头力量增强，为买入信号
    - Bear Power < 0 且下降时，表示空头力量增强，为卖出信号
    - Bull Power创新高时，表示多头力量强劲
    - Bear Power创新低时，表示空头力量强劲
    - Bull Power由负转正时，为买入信号
    - Bear Power由正转负时，为卖出信号
    - EMA上升时，表示上升趋势
    - EMA下降时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认13

    输出参数：
    - EMA: 指数移动平均线
    - Bull_Power: 多头力量
    - Bear_Power: 空头力量
    - bull_power_positive: Bull Power > 0信号
    - bear_power_negative: Bear Power < 0信号
    - bull_power_rising: Bull Power上升信号
    - bear_power_falling: Bear Power下降信号
    - bull_power_new_high: Bull Power创新高信号
    - bear_power_new_low: Bear Power创新低信号
    - bull_power_cross_zero_up: Bull Power上穿0线信号
    - bear_power_cross_zero_down: Bear Power下穿0线信号
    - ema_rising: EMA上升信号
    - ema_falling: EMA下降信号

    注意事项：
    - Elder Ray是趋势指标，在震荡市中会产生频繁的假信号
    - Elder Ray在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - Elder Ray反应较快，适合中短线操作

    最佳实践建议：
    1. 趋势确认：Bull Power和EMA同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Elder Ray
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=13):
        """
        初始化Elder Ray指标

        Parameters
        ----------
        period : int, default 13
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算Elder Ray指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Elder Ray计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['Bull_Power'] = df['High'] - df['EMA']
        df['Bear_Power'] = df['Low'] - df['EMA']

        df['bull_power_positive'] = df['Bull_Power'] > 0
        df['bear_power_negative'] = df['Bear_Power'] < 0
        df['bull_power_rising'] = df['Bull_Power'] > df['Bull_Power'].shift(1)
        df['bear_power_falling'] = df['Bear_Power'] < df['Bear_Power'].shift(1)
        df['bull_power_new_high'] = df['Bull_Power'] == df['Bull_Power'].rolling(window=self.period).max()
        df['bear_power_new_low'] = df['Bear_Power'] == df['Bear_Power'].rolling(window=self.period).min()
        df['bull_power_cross_zero_up'] = (df['Bull_Power'] > 0) & (df['Bull_Power'].shift(1) <= 0)
        df['bear_power_cross_zero_down'] = (df['Bear_Power'] < 0) & (df['Bear_Power'].shift(1) >= 0)
        df['ema_rising'] = df['EMA'] > df['EMA'].shift(1)
        df['ema_falling'] = df['EMA'] < df['EMA'].shift(1)

        return df[['EMA', 'Bull_Power', 'Bear_Power',
                   'bull_power_positive', 'bear_power_negative',
                   'bull_power_rising', 'bear_power_falling',
                   'bull_power_new_high', 'bear_power_new_low',
                   'bull_power_cross_zero_up', 'bear_power_cross_zero_down',
                   'ema_rising', 'ema_falling']]
