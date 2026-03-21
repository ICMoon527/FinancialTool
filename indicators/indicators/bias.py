import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BIAS(BaseIndicator):
    """
    BIAS指标，中文称为乖离率指标或偏离率指标。

    BIAS指标是一种基于移动平均线的技术指标，通过计算当日收盘价与移动平均线之间的偏离程度，
    来判断价格是否超买或超卖。BIAS指标反映了价格偏离其趋势的程度。

    计算公式：
    BIAS = (Close - MA) / MA × 100
    其中：
    - Close 为当日收盘价
    - MA 为N日移动平均线
    - N 为计算周期，通常为5日、10日、20日或30日

    使用场景：
    - BIAS > 0 时，价格在移动平均线上方，为多头市场
    - BIAS < 0 时，价格在移动平均线下方，为空头市场
    - BIAS过大时（如>5），处于超买区间，可能回落，为卖出信号
    - BIAS过小时（如<-5），处于超卖区间，可能反弹，为买入信号
    - BIAS由下往上穿越0线时，为买入信号
    - BIAS由上往下穿越0线时，为卖出信号
    - BIAS向下跌破上升趋势线时，为卖出信号
    - BIAS向上突破下降趋势线时，为买入信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认10
    - overbought_threshold: 超买阈值，默认5
    - oversold_threshold: 超卖阈值，默认-5

    输出参数：
    - BIAS: 乖离率指标值
    - MA: 移动平均线
    - overbought: 超买信号
    - oversold: 超卖信号
    - cross_zero_up: 上穿0线信号
    - cross_zero_down: 下穿0线信号

    注意事项：
    - BIAS的超买超卖阈值需要根据不同股票和市场调整
    - 强势股的BIAS可以容忍更高的正值，弱势股可以容忍更低的负值
    - 周期参数可以根据市场特性调整
    - 在单边市中，BIAS可能长期处于超买或超卖区
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 动态阈值：根据历史数据设置动态的超买超卖阈值
    2. 多周期使用：同时使用短期、中期和长期BIAS
    3. 趋势过滤：先判断大趋势，顺势操作
    4. 背离确认：结合价格和BIAS的背离信号
    5. 组合使用：与MA、MACD等趋势指标配合使用
    6. 量价分析：结合成交量变化验证信号
    7. 止损设置：设置严格止损，控制风险
    8. 统计分析：分析历史BIAS的分布情况
    """

    def __init__(self, period=10, overbought_threshold=5, oversold_threshold=-5):
        """
        初始化BIAS指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        overbought_threshold : float, default 5
            超买阈值
        oversold_threshold : float, default -5
            超卖阈值
        """
        self.period = period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def calculate(self, data):
        """
        计算BIAS指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BIAS计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA'] = df['Close'].rolling(window=self.period).mean()
        df['BIAS'] = (df['Close'] - df['MA']) / df['MA'] * 100

        df['overbought'] = df['BIAS'] > self.overbought_threshold
        df['oversold'] = df['BIAS'] < self.oversold_threshold

        df['cross_zero_up'] = (df['BIAS'] > 0) & (df['BIAS'].shift(1) <= 0)
        df['cross_zero_down'] = (df['BIAS'] < 0) & (df['BIAS'].shift(1) >= 0)

        return df[['BIAS', 'MA', 'overbought', 'oversold', 'cross_zero_up', 'cross_zero_down']]
