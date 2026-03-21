import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MACDExtended(BaseIndicator):
    """
    MACDExtended（MACD扩展），中文称为MACD扩展。

    MACDExtended指标是MACD指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    MACDExtended指标结合了快速EMA、慢速EMA和信号线，来反应价格的趋势和动量。

    计算公式：
    1. 计算EMA12：EMA12 = EMA(Close, 12)
    2. 计算EMA26：EMA26 = EMA(Close, 26)
    3. 计算DIF：DIF = EMA12 - EMA26
    4. 计算DEA：DEA = EMA(DIF, 9)
    5. 计算MACD：MACD = 2 × (DIF - DEA)
    6. 计算MACD_HIST：MACD_HIST = DIF - DEA
    7. 计算MACD_OSC：MACD_OSC = (DIF / DEA) × 100

    使用场景：
    - DIF上穿DEA时，为买入信号（金叉）
    - DIF下穿DEA时，为卖出信号（死叉）
    - MACD > 0 时，表示多头市场
    - MACD < 0 时，表示空头市场
    - MACD上升时，表示上升趋势
    - MACD下降时，表示下降趋势
    - MACD与价格同步时，趋势健康
    - MACD与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号线周期，默认9

    输出参数：
    - EMA12: 快速指数移动平均线
    - EMA26: 慢速指数移动平均线
    - DIF: 差离值
    - DEA: 信号线
    - MACD: MACD柱
    - MACD_HIST: MACD柱状图
    - MACD_OSC: MACD震荡指标
    - macd_positive: MACD > 0信号
    - macd_negative: MACD < 0信号
    - macd_rising: MACD上升信号
    - macd_falling: MACD下降信号
    - dif_above_dea: DIF在DEA上方信号
    - dif_below_dea: DIF在DEA下方信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - MACDExtended是趋势指标，在震荡市中会产生频繁的假信号
    - MACDExtended在单边市中表现最好
    - MACDExtended反应较慢，不适合短线操作
    - 建议结合其他指标（如RSI、KDJ）一起使用

    最佳实践建议：
    1. 趋势确认：价格和MACD同时向上更可靠
    2. 组合使用：与RSI等摆动指标配合使用
    3. 多周期使用：同时使用日线和周线的MACDExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化MACDExtended指标

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
        计算MACDExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MACDExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA12'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['DIF'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_HIST'] = df['DIF'] - df['DEA']
        df['MACD'] = df['MACD_HIST'] * 2
        df['MACD_OSC'] = (df['DIF'] / df['DEA'].replace(0, np.nan)) * 100
        df['MACD_OSC'] = df['MACD_OSC'].fillna(0)

        df['macd_positive'] = df['MACD'] > 0
        df['macd_negative'] = df['MACD'] < 0
        df['macd_rising'] = df['MACD'] > df['MACD'].shift(1)
        df['macd_falling'] = df['MACD'] < df['MACD'].shift(1)
        df['dif_above_dea'] = df['DIF'] > df['DEA']
        df['dif_below_dea'] = df['DIF'] < df['DEA']
        df['golden_cross'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
        df['death_cross'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

        return df[['EMA12', 'EMA26', 'DIF', 'DEA', 'MACD', 'MACD_HIST', 'MACD_OSC',
                   'macd_positive', 'macd_negative',
                   'macd_rising', 'macd_falling',
                   'dif_above_dea', 'dif_below_dea',
                   'golden_cross', 'death_cross']]
