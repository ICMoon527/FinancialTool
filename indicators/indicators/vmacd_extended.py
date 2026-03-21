import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VMACDExtended(BaseIndicator):
    """
    VMACDExtended（量指数平滑异同移动平均扩展），中文称为量指数平滑异同移动平均扩展。

    VMACDExtended指标是VMACD指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    VMACDExtended指标结合了成交量的移动平均，来反应市场的资金流向。

    计算公式：
    1. 计算EMA1：EMA1 = EMA(Volume, N)
    2. 计算EMA2：EMA2 = EMA(Volume, M)
    3. 计算DIF：DIF = EMA1 - EMA2
    4. 计算DEA：DEA = EMA(DIF, P)
    5. 计算MACD柱：MACD_OSC = (DIF - DEA) × 2

    使用场景：
    - DIF > 0 时，表示多头市场
    - DIF < 0 时，表示空头市场
    - DIF上升时，表示量能放大
    - DIF下降时，表示量能萎缩
    - DIF上穿DEA时，为买入信号（金叉）
    - DIF下穿DEA时，为卖出信号（死叉）
    - MACD柱 > 0 时，表示量能放大
    - MACD柱 < 0 时，表示量能萎缩
    - MACD柱上升时，表示量能放大加速
    - MACD柱下降时，表示量能放大减速

    输入参数：
    - data: DataFrame，必须包含'Volume'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号线周期，默认9

    输出参数：
    - VEMA_Fast: 快速指数移动平均线
    - VEMA_Slow: 慢速指数移动平均线
    - DIF: 差离值
    - DEA: 信号线
    - MACD_OSC: MACD柱
    - dif_positive: DIF > 0信号
    - dif_negative: DIF < 0信号
    - dif_rising: DIF上升信号
    - dif_falling: DIF下降信号
    - dif_above_dea: DIF在DEA上方信号
    - dif_below_dea: DIF在DEA下方信号
    - macd_hist_positive: MACD柱 > 0信号
    - macd_hist_negative: MACD柱 < 0信号
    - macd_hist_rising: MACD柱上升信号
    - macd_hist_falling: MACD柱下降信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - VMACDExtended是量能指标，反应资金流动
    - VMACDExtended在震荡市中会产生频繁的假信号
    - VMACDExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VMACD和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的VMACDExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化VMACDExtended指标

        Parameters
        ----------
        fast_period : int, default 12
            快速EMA周期
        slow_period : int, default 26
            慢速EMA周期
        signal_period : int, default 9
            信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算VMACDExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VMACDExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['VEMA_Fast'] = df['Volume'].ewm(span=self.fast_period, adjust=False).mean()
        df['VEMA_Slow'] = df['Volume'].ewm(span=self.slow_period, adjust=False).mean()
        df['DIF'] = df['VEMA_Fast'] - df['VEMA_Slow']
        df['DEA'] = df['DIF'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_OSC'] = (df['DIF'] - df['DEA']) * 2

        df['dif_positive'] = df['DIF'] > 0
        df['dif_negative'] = df['DIF'] < 0
        df['dif_rising'] = df['DIF'] > df['DIF'].shift(1)
        df['dif_falling'] = df['DIF'] < df['DIF'].shift(1)
        df['dif_above_dea'] = df['DIF'] > df['DEA']
        df['dif_below_dea'] = df['DIF'] < df['DEA']
        df['macd_hist_positive'] = df['MACD_OSC'] > 0
        df['macd_hist_negative'] = df['MACD_OSC'] < 0
        df['macd_hist_rising'] = df['MACD_OSC'] > df['MACD_OSC'].shift(1)
        df['macd_hist_falling'] = df['MACD_OSC'] < df['MACD_OSC'].shift(1)
        df['golden_cross'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
        df['death_cross'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

        return df[['VEMA_Fast', 'VEMA_Slow', 'DIF', 'DEA', 'MACD_OSC',
                   'dif_positive', 'dif_negative',
                   'dif_rising', 'dif_falling',
                   'dif_above_dea', 'dif_below_dea',
                   'macd_hist_positive', 'macd_hist_negative',
                   'macd_hist_rising', 'macd_hist_falling',
                   'golden_cross', 'death_cross']]
