import pandas as pd
import numpy as np
from ..base import BaseIndicator


class WilliamsR(BaseIndicator):
    """
    Williams %R（威廉指标），中文称为威廉指标。

    Williams %R指标是由Larry Williams提出的，是一种震荡指标。
    Williams %R指标通过计算当前价格在一段时间内的位置，来判断价格的超买超卖状态。

    计算公式：
    Williams %R = (Highest_High - Close) / (Highest_High - Lowest_Low) × (-100)

    使用场景：
    - Williams %R > -20 时，为超买信号，可能回落
    - Williams %R < -80 时，为超卖信号，可能反弹
    - Williams %R从超买区回落时，为卖出信号
    - Williams %R从超卖区反弹时，为买入信号
    - Williams %R与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14

    输出参数：
    - Williams_R: 威廉指标
    - Williams_R_MA: 威廉指标的移动平均线
    - williams_overbought: Williams %R > -20信号
    - williams_oversold: Williams %R < -80信号
    - williams_rising: Williams %R上升信号
    - williams_falling: Williams %R下降信号
    - williams_cross_overbought_down: Williams %R下穿-20信号
    - williams_cross_oversold_up: Williams %R上穿-80信号
    - williams_above_ma: Williams %R在MA上方信号
    - williams_below_ma: Williams %R在MA下方信号

    注意事项：
    - Williams %R是震荡指标，在单边市中会产生频繁的假信号
    - Williams %R在震荡市中表现最好
    - Williams %R反应较快，适合中短线操作
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、BOLL）一起使用

    最佳实践建议：
    1. 趋势确认：Williams %R和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Williams %R
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14, ma_period=10):
        """
        初始化Williams %R指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算Williams %R指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Williams %R计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        highest_high = df['High'].rolling(window=self.period).max()
        lowest_low = df['Low'].rolling(window=self.period).min()
        df['Williams_R'] = (highest_high - df['Close']) / (highest_high - lowest_low) * (-100)
        df['Williams_R_MA'] = df['Williams_R'].rolling(window=self.ma_period).mean()

        df['williams_overbought'] = df['Williams_R'] > -20
        df['williams_oversold'] = df['Williams_R'] < -80
        df['williams_rising'] = df['Williams_R'] > df['Williams_R'].shift(1)
        df['williams_falling'] = df['Williams_R'] < df['Williams_R'].shift(1)
        df['williams_cross_overbought_down'] = (df['Williams_R'] < -20) & (df['Williams_R'].shift(1) >= -20)
        df['williams_cross_oversold_up'] = (df['Williams_R'] > -80) & (df['Williams_R'].shift(1) <= -80)
        df['williams_above_ma'] = df['Williams_R'] > df['Williams_R_MA']
        df['williams_below_ma'] = df['Williams_R'] < df['Williams_R_MA']

        return df[['Williams_R', 'Williams_R_MA',
                   'williams_overbought', 'williams_oversold',
                   'williams_rising', 'williams_falling',
                   'williams_cross_overbought_down', 'williams_cross_oversold_up',
                   'williams_above_ma', 'williams_below_ma']]
