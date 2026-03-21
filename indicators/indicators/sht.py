import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SHT(BaseIndicator):
    """
    SHT（蛇系指标），中文称为蛇系指标。

    SHT指标是一种短期动量指标，通过计算价格的短期变化，来判断价格的超买超卖状态。
    SHT指标结合了移动平均和动量，来反应市场的短期情绪。

    计算公式：
    1. 计算SMA1：SMA1 = SMA(Close, N)
    2. 计算SMA2：SMA2 = SMA(SMA1, M)
    3. 计算SHT：SHT = (SMA1 - SMA2) × 100

    使用场景：
    - SHT > 0 时，表示多头市场
    - SHT < 0 时，表示空头市场
    - SHT上升时，表示多头力量增强，为买入信号
    - SHT下降时，表示空头力量增强，为卖出信号
    - SHT由负转正时，为买入信号
    - SHT由正转负时，为卖出信号
    - SHT与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - short_period: 短期周期，默认5
    - long_period: 长期周期，默认10

    输出参数：
    - SMA1: 短期移动平均线
    - SMA2: 长期移动平均线
    - SHT: 蛇系指标
    - SHT_MA: SHT的移动平均线
    - sht_positive: SHT > 0信号
    - sht_negative: SHT < 0信号
    - sht_rising: SHT上升信号
    - sht_falling: SHT下降信号
    - sht_cross_zero_up: SHT上穿0线信号
    - sht_cross_zero_down: SHT下穿0线信号
    - sht_above_ma: SHT在MA上方信号
    - sht_below_ma: SHT在MA下方信号

    注意事项：
    - SHT是短期动量指标，在震荡市中会产生频繁的假信号
    - SHT在单边市中表现最好
    - SHT反应较快，适合中短线操作
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：SHT和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的SHT
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, short_period=5, long_period=10, ma_period=10):
        """
        初始化SHT指标

        Parameters
        ----------
        short_period : int, default 5
            短期周期
        long_period : int, default 10
            长期周期
        ma_period : int, default 10
            移动平均周期
        """
        self.short_period = short_period
        self.long_period = long_period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算SHT指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SHT计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['SMA1'] = df['Close'].rolling(window=self.short_period).mean()
        df['SMA2'] = df['SMA1'].rolling(window=self.long_period).mean()
        df['SHT'] = (df['SMA1'] - df['SMA2']) * 100
        df['SHT_MA'] = df['SHT'].rolling(window=self.ma_period).mean()

        df['sht_positive'] = df['SHT'] > 0
        df['sht_negative'] = df['SHT'] < 0
        df['sht_rising'] = df['SHT'] > df['SHT'].shift(1)
        df['sht_falling'] = df['SHT'] < df['SHT'].shift(1)
        df['sht_cross_zero_up'] = (df['SHT'] > 0) & (df['SHT'].shift(1) <= 0)
        df['sht_cross_zero_down'] = (df['SHT'] < 0) & (df['SHT'].shift(1) >= 0)
        df['sht_above_ma'] = df['SHT'] > df['SHT_MA']
        df['sht_below_ma'] = df['SHT'] < df['SHT_MA']

        return df[['SMA1', 'SMA2', 'SHT', 'SHT_MA',
                   'sht_positive', 'sht_negative',
                   'sht_rising', 'sht_falling',
                   'sht_cross_zero_up', 'sht_cross_zero_down',
                   'sht_above_ma', 'sht_below_ma']]
