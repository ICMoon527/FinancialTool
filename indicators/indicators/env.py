import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ENV(BaseIndicator):
    """
    ENV（Envelope）指标，中文称为通道指标或包络线指标。

    ENV指标是一种基于移动平均线的指标，通过计算移动平均线上下一定百分比的通道，
    来判断价格的超买超卖状态和趋势变化。

    计算公式：
    1. 计算MA（移动平均线）：MA = SMA(Close, N)
    2. 计算上轨：Upper = MA × (1 + K/100)
    3. 计算下轨：Lower = MA × (1 - K/100)

    使用场景：
    - 价格突破上轨时，为超买信号，可能回落
    - 价格跌破下轨时，为超卖信号，可能反弹
    - 价格在上轨上方时，为强势市场
    - 价格在下轨下方时，为弱势市场
    - 价格在中轨上方时，为多头市场
    - 价格在中轨下方时，为空头市场
    - 通道向上时，为上升趋势
    - 通道向下时，为下降趋势

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20
    - percent: 通道百分比，默认10

    输出参数：
    - MA: 移动平均线（中轨）
    - Upper: 上轨
    - Lower: 下轨
    - close_above_upper: 价格突破上轨信号
    - close_below_lower: 价格跌破下轨信号
    - close_above_ma: 价格在中轨上方信号
    - close_below_ma: 价格在中轨下方信号
    - ma_rising: 中轨上升信号
    - ma_falling: 中轨下降信号
    - channel_width: 通道宽度
    - channel_expanding: 通道扩大信号
    - channel_shrinking: 通道缩小信号

    注意事项：
    - ENV是趋势指标，在震荡市中会产生频繁的假信号
    - ENV在单边市中表现最好
    - 周期参数和百分比可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - ENV反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：价格和中轨同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ENV
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, percent=10):
        """
        初始化ENV指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        percent : int, default 10
            通道百分比
        """
        self.period = period
        self.percent = percent

    def calculate(self, data):
        """
        计算ENV指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ENV计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA'] = df['Close'].rolling(window=self.period).mean()
        df['Upper'] = df['MA'] * (1 + self.percent / 100)
        df['Lower'] = df['MA'] * (1 - self.percent / 100)

        df['close_above_upper'] = df['Close'] > df['Upper']
        df['close_below_lower'] = df['Close'] < df['Lower']
        df['close_above_ma'] = df['Close'] > df['MA']
        df['close_below_ma'] = df['Close'] < df['MA']
        df['ma_rising'] = df['MA'] > df['MA'].shift(1)
        df['ma_falling'] = df['MA'] < df['MA'].shift(1)
        df['channel_width'] = df['Upper'] - df['Lower']
        df['channel_expanding'] = df['channel_width'] > df['channel_width'].shift(1)
        df['channel_shrinking'] = df['channel_width'] < df['channel_width'].shift(1)

        return df[['MA', 'Upper', 'Lower', 'close_above_upper', 'close_below_lower', 
                   'close_above_ma', 'close_below_ma', 'ma_rising', 'ma_falling',
                   'channel_width', 'channel_expanding', 'channel_shrinking']]
