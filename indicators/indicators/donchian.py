import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Donchian(BaseIndicator):
    """
    Donchian Channel（唐奇安通道），中文称为唐奇安通道。

    Donchian Channel指标是由Richard Donchian提出的，是一种基于最高价和最低价的通道指标。
    Donchian Channel指标通过计算一段时间内的最高价和最低价，来判断价格的趋势和支撑阻力位。

    计算公式：
    1. 计算Upper（上轨）：Upper = Highest_High(N)
    2. 计算Lower（下轨）：Lower = Lowest_Low(N)
    3. 计算Middle（中轨）：Middle = (Upper + Lower) / 2

    使用场景：
    - 价格突破上轨时，为买入信号
    - 价格跌破下轨时，为卖出信号
    - 价格在中轨上方时，为多头市场
    - 价格在中轨下方时，为空头市场
    - 通道向上时，表示上升趋势
    - 通道向下时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认20

    输出参数：
    - Upper: 上轨
    - Lower: 下轨
    - Middle: 中轨
    - close_above_upper: 价格突破上轨信号
    - close_below_lower: 价格跌破下轨信号
    - close_above_middle: 价格在中轨上方信号
    - close_below_middle: 价格在中轨下方信号
    - middle_rising: 中轨上升信号
    - middle_falling: 中轨下降信号
    - channel_width: 通道宽度
    - channel_expanding: 通道扩大信号
    - channel_shrinking: 通道缩小信号

    注意事项：
    - Donchian Channel是趋势跟踪指标，在震荡市中会产生频繁的假信号
    - Donchian Channel在单边市中表现最好
    - Donchian Channel反应较慢，不适合短线操作
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和中轨同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Donchian Channel
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20):
        """
        初始化Donchian Channel指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算Donchian Channel指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Donchian Channel计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['Upper'] = df['High'].rolling(window=self.period).max()
        df['Lower'] = df['Low'].rolling(window=self.period).min()
        df['Middle'] = (df['Upper'] + df['Lower']) / 2

        df['close_above_upper'] = df['Close'] > df['Upper']
        df['close_below_lower'] = df['Close'] < df['Lower']
        df['close_above_middle'] = df['Close'] > df['Middle']
        df['close_below_middle'] = df['Close'] < df['Middle']
        df['middle_rising'] = df['Middle'] > df['Middle'].shift(1)
        df['middle_falling'] = df['Middle'] < df['Middle'].shift(1)
        df['channel_width'] = df['Upper'] - df['Lower']
        df['channel_expanding'] = df['channel_width'] > df['channel_width'].shift(1)
        df['channel_shrinking'] = df['channel_width'] < df['channel_width'].shift(1)

        return df[['Upper', 'Lower', 'Middle',
                   'close_above_upper', 'close_below_lower',
                   'close_above_middle', 'close_below_middle',
                   'middle_rising', 'middle_falling',
                   'channel_width', 'channel_expanding', 'channel_shrinking']]
