import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ForceIndex(BaseIndicator):
    """
    Force Index（强力指数指标），中文称为强力指数指标。

    Force Index指标是由亚历山大·埃尔德（Alexander Elder）提出的，是一种量价指标。
    Force Index指标通过计算价格变动与成交量的乘积，来衡量市场的买卖力量和动量变化。

    计算公式：
    1. 计算价格变动：Price_Change = Close - Close_prev
    2. 计算Raw Force Index：Raw_FI = Price_Change × Volume
    3. 计算Force Index：FI = EMA(Raw_FI, N)

    使用场景：
    - FI > 0 时，表示买方力量较强，为多头市场
    - FI < 0 时，表示卖方力量较强，为空头市场
    - FI上升时，表示买方力量增强，为买入信号
    - FI下降时，表示卖方力量增强，为卖出信号
    - FI创新高时，表示买方力量强劲
    - FI创新低时，表示卖方力量强劲
    - FI与价格同步上升时，趋势健康
    - FI与价格同步下降时，趋势健康
    - FI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period: 计算周期，默认13

    输出参数：
    - Price_Change: 价格变动
    - Raw_FI: 原始强力指数
    - FI: 强力指数
    - FI_MA: 强力指数的移动平均线
    - fi_positive: FI > 0信号
    - fi_negative: FI < 0信号
    - fi_rising: FI上升信号
    - fi_falling: FI下降信号
    - fi_new_high: FI创新高信号
    - fi_new_low: FI创新低信号
    - fi_above_ma: FI在MA上方信号
    - fi_below_ma: FI在MA下方信号
    - fi_cross_zero_up: FI上穿0线信号
    - fi_cross_zero_down: FI下穿0线信号

    注意事项：
    - Force Index是量价指标，反应买卖力量
    - Force Index在震荡市中会产生频繁的假信号
    - Force Index在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：FI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Force Index
    4. 背离判断：注意FI与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=13, ma_period=10):
        """
        初始化Force Index指标

        Parameters
        ----------
        period : int, default 13
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算Force Index指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Force Index计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - close_prev
        df['Raw_FI'] = df['Price_Change'] * df['Volume']
        df['FI'] = df['Raw_FI'].ewm(span=self.period, adjust=False).mean()
        df['FI_MA'] = df['FI'].rolling(window=self.ma_period).mean()

        df['fi_positive'] = df['FI'] > 0
        df['fi_negative'] = df['FI'] < 0
        df['fi_rising'] = df['FI'] > df['FI'].shift(1)
        df['fi_falling'] = df['FI'] < df['FI'].shift(1)
        df['fi_new_high'] = df['FI'] == df['FI'].rolling(window=self.ma_period).max()
        df['fi_new_low'] = df['FI'] == df['FI'].rolling(window=self.ma_period).min()
        df['fi_above_ma'] = df['FI'] > df['FI_MA']
        df['fi_below_ma'] = df['FI'] < df['FI_MA']
        df['fi_cross_zero_up'] = (df['FI'] > 0) & (df['FI'].shift(1) <= 0)
        df['fi_cross_zero_down'] = (df['FI'] < 0) & (df['FI'].shift(1) >= 0)

        return df[['Price_Change', 'Raw_FI', 'FI', 'FI_MA',
                   'fi_positive', 'fi_negative', 'fi_rising', 'fi_falling',
                   'fi_new_high', 'fi_new_low', 'fi_above_ma', 'fi_below_ma',
                   'fi_cross_zero_up', 'fi_cross_zero_down']]
