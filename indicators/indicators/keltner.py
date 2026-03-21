import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Keltner(BaseIndicator):
    """
    Keltner Channel（凯特纳通道指标），中文称为凯特纳通道指标。

    Keltner Channel指标是由切斯特·凯特纳（Chester Keltner）提出的，是一种基于ATR的通道指标。
    Keltner Channel指标通过计算移动平均线和ATR的倍数，来判断价格的超买超卖状态和趋势变化。

    计算公式：
    1. 计算EMA：EMA = EMA(Close, N)
    2. 计算ATR：ATR = ATR(N)
    3. 计算Upper（上轨）：Upper = EMA + K × ATR
    4. 计算Lower（下轨）：Lower = EMA - K × ATR

    使用场景：
    - 价格突破上轨时，为超买信号，可能回落
    - 价格跌破下轨时，为超卖信号，可能反弹
    - 价格在中轨上方时，为多头市场
    - 价格在中轨下方时，为空头市场
    - 通道向上时，表示上升趋势
    - 通道向下时，表示下降趋势
    - 通道扩大时，表示波动性增加
    - 通道缩小时，表示波动性减少

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认20
    - atr_period: ATR周期，默认10
    - multiplier: ATR倍数，默认2

    输出参数：
    - EMA: 指数移动平均线（中轨）
    - ATR: 平均真实波幅
    - Upper: 上轨
    - Lower: 下轨
    - close_above_upper: 价格突破上轨信号
    - close_below_lower: 价格跌破下轨信号
    - close_above_ema: 价格在中轨上方信号
    - close_below_ema: 价格在中轨下方信号
    - ema_rising: 中轨上升信号
    - ema_falling: 中轨下降信号
    - channel_width: 通道宽度
    - channel_expanding: 通道扩大信号
    - channel_shrinking: 通道缩小信号

    注意事项：
    - Keltner Channel是趋势指标，在震荡市中会产生频繁的假信号
    - Keltner Channel在单边市中表现最好
    - 周期参数和倍数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - Keltner Channel反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：价格和中轨同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Keltner Channel
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, atr_period=10, multiplier=2):
        """
        初始化Keltner Channel指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        atr_period : int, default 10
            ATR周期
        multiplier : int, default 2
            ATR倍数
        """
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate(self, data):
        """
        计算Keltner Channel指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Keltner Channel计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA'] = df['Close'].ewm(span=self.period, adjust=False).mean()

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=self.atr_period).mean()

        df['Upper'] = df['EMA'] + self.multiplier * df['ATR']
        df['Lower'] = df['EMA'] - self.multiplier * df['ATR']

        df['close_above_upper'] = df['Close'] > df['Upper']
        df['close_below_lower'] = df['Close'] < df['Lower']
        df['close_above_ema'] = df['Close'] > df['EMA']
        df['close_below_ema'] = df['Close'] < df['EMA']
        df['ema_rising'] = df['EMA'] > df['EMA'].shift(1)
        df['ema_falling'] = df['EMA'] < df['EMA'].shift(1)
        df['channel_width'] = df['Upper'] - df['Lower']
        df['channel_expanding'] = df['channel_width'] > df['channel_width'].shift(1)
        df['channel_shrinking'] = df['channel_width'] < df['channel_width'].shift(1)

        return df[['EMA', 'ATR', 'Upper', 'Lower',
                   'close_above_upper', 'close_below_lower',
                   'close_above_ema', 'close_below_ema',
                   'ema_rising', 'ema_falling',
                   'channel_width', 'channel_expanding', 'channel_shrinking']]
