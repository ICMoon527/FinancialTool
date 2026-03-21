import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EhlerFisher(BaseIndicator):
    """
    Ehlers Fisher Transform（埃勒斯费舍尔变换），中文称为埃勒斯费舍尔变换。

    Ehlers Fisher Transform指标是由John Ehlers提出的，是一种统计变换指标。
    Ehlers Fisher Transform指标通过将价格转换为高斯分布，来判断价格的超买超卖状态。

    计算公式：
    1. 计算Value：Value = (High + Low) / 2
    2. 计算Normalized：Normalized = (Value - Lowest_Value(N)) / (Highest_Value(N) - Lowest_Value(N)) × 2 - 1
    3. 计算Fisher：Fisher = 0.5 × ln((1 + Normalized) / (1 - Normalized))

    使用场景：
    - Fisher > 0 时，表示多头市场
    - Fisher < 0 时，表示空头市场
    - Fisher上升时，表示多头力量增强，为买入信号
    - Fisher下降时，表示空头力量增强，为卖出信号
    - Fisher与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'列
    - period: 计算周期，默认10

    输出参数：
    - Value: 价格中间值
    - Normalized: 标准化值
    - Fisher: 费舍尔变换
    - Fisher_MA: 费舍尔变换的移动平均线
    - fisher_positive: Fisher > 0信号
    - fisher_negative: Fisher < 0信号
    - fisher_rising: Fisher上升信号
    - fisher_falling: Fisher下降信号
    - fisher_above_ma: Fisher在MA上方信号
    - fisher_below_ma: Fisher在MA下方信号
    - fisher_cross_zero_up: Fisher上穿0线信号
    - fisher_cross_zero_down: Fisher下穿0线信号

    注意事项：
    - Ehlers Fisher Transform是统计变换指标，在震荡市中会产生频繁的假信号
    - Ehlers Fisher Transform在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：Fisher和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Ehlers Fisher Transform
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period=10):
        """
        初始化Ehlers Fisher Transform指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算Ehlers Fisher Transform指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Ehlers Fisher Transform计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['Value'] = (df['High'] + df['Low']) / 2
        value_min = df['Value'].rolling(window=self.period).min()
        value_max = df['Value'].rolling(window=self.period).max()
        df['Normalized'] = ((df['Value'] - value_min) / (value_max - value_min)) * 2 - 1
        df['Normalized'] = df['Normalized'].clip(-0.999, 0.999)
        df['Fisher'] = 0.5 * np.log((1 + df['Normalized']) / (1 - df['Normalized']))
        df['Fisher_MA'] = df['Fisher'].rolling(window=self.ma_period).mean()

        df['fisher_positive'] = df['Fisher'] > 0
        df['fisher_negative'] = df['Fisher'] < 0
        df['fisher_rising'] = df['Fisher'] > df['Fisher'].shift(1)
        df['fisher_falling'] = df['Fisher'] < df['Fisher'].shift(1)
        df['fisher_above_ma'] = df['Fisher'] > df['Fisher_MA']
        df['fisher_below_ma'] = df['Fisher'] < df['Fisher_MA']
        df['fisher_cross_zero_up'] = (df['Fisher'] > 0) & (df['Fisher'].shift(1) <= 0)
        df['fisher_cross_zero_down'] = (df['Fisher'] < 0) & (df['Fisher'].shift(1) >= 0)

        return df[['Value', 'Normalized', 'Fisher', 'Fisher_MA',
                   'fisher_positive', 'fisher_negative', 'fisher_rising', 'fisher_falling',
                   'fisher_above_ma', 'fisher_below_ma',
                   'fisher_cross_zero_up', 'fisher_cross_zero_down']]
