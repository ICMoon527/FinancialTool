import pandas as pd
import numpy as np
from ..base import BaseIndicator


class QACDExtended(BaseIndicator):
    """
    QACDExtended（快异同平均扩展），中文称为快异同平均扩展。

    QACDExtended指标是QACD指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    QACDExtended指标结合了快速和慢速移动平均，来反应价格的趋势。

    计算公式：
    1. 计算DIF：DIF = EMA(Close, N) - EMA(Close, M)
    2. 计算MACD：MACD = EMA(DIF, P)
    3. 计算QACD_OSC：QACD_OSC = (DIF - MACD) × 2

    使用场景：
    - DIF > 0 时，表示多头市场
    - DIF < 0 时，表示空头市场
    - DIF上升时，表示上升趋势
    - DIF下降时，表示下降趋势
    - DIF上穿MACD时，为买入信号（金叉）
    - DIF下穿MACD时，为卖出信号（死叉）
    - QACD_OSC > 0 时，表示量价配合良好
    - QACD_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认10
    - slow_period: 慢速EMA周期，默认30
    - signal_period: 信号线周期，默认10

    输出参数：
    - DIF: 差离值
    - MACD: 信号线
    - QACD_OSC: QACD柱
    - dif_positive: DIF > 0信号
    - dif_negative: DIF < 0信号
    - dif_rising: DIF上升信号
    - dif_falling: DIF下降信号
    - dif_above_macd: DIF在MACD上方信号
    - dif_below_macd: DIF在MACD下方信号
    - qacd_hist_positive: QACD柱 > 0信号
    - qacd_hist_negative: QACD柱 < 0信号
    - qacd_hist_rising: QACD柱上升信号
    - qacd_hist_falling: QACD柱下降信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - QACDExtended是趋势指标，在单边市中表现最好
    - QACDExtended在震荡市中会产生频繁的假信号
    - QACDExtended反应较快，适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：DIF和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的QACDExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=10, slow_period=30, signal_period=10):
        """
        初始化QACDExtended指标

        Parameters
        ----------
        fast_period : int, default 10
            快速EMA周期
        slow_period : int, default 30
            慢速EMA周期
        signal_period : int, default 10
            信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算QACDExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含QACDExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['DIF'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean() - df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        df['MACD'] = df['DIF'].ewm(span=self.signal_period, adjust=False).mean()
        df['QACD_OSC'] = (df['DIF'] - df['MACD']) * 2

        df['dif_positive'] = df['DIF'] > 0
        df['dif_negative'] = df['DIF'] < 0
        df['dif_rising'] = df['DIF'] > df['DIF'].shift(1)
        df['dif_falling'] = df['DIF'] < df['DIF'].shift(1)
        df['dif_above_macd'] = df['DIF'] > df['MACD']
        df['dif_below_macd'] = df['DIF'] < df['MACD']
        df['qacd_hist_positive'] = df['QACD_OSC'] > 0
        df['qacd_hist_negative'] = df['QACD_OSC'] < 0
        df['qacd_hist_rising'] = df['QACD_OSC'] > df['QACD_OSC'].shift(1)
        df['qacd_hist_falling'] = df['QACD_OSC'] < df['QACD_OSC'].shift(1)
        df['golden_cross'] = (df['DIF'] > df['MACD']) & (df['DIF'].shift(1) <= df['MACD'].shift(1))
        df['death_cross'] = (df['DIF'] < df['MACD']) & (df['DIF'].shift(1) >= df['MACD'].shift(1))

        return df[['DIF', 'MACD', 'QACD_OSC',
                   'dif_positive', 'dif_negative',
                   'dif_rising', 'dif_falling',
                   'dif_above_macd', 'dif_below_macd',
                   'qacd_hist_positive', 'qacd_hist_negative',
                   'qacd_hist_rising', 'qacd_hist_falling',
                   'golden_cross', 'death_cross']]
