import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DKXExtended(BaseIndicator):
    """
    DKXExtended（多空线扩展），中文称为多空线扩展。

    DKXExtended指标是DKX指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    DKXExtended指标结合了多个移动平均线，来反应价格的趋势。

    计算公式：
    1. 计算MID：MID = (3 × Close + High + Low + Open) / 6
    2. 计算DKX：DKX = SMA((MID × 2 + MID_1 + MID_2 + MID_3) / 5, N)
    3. 计算MADKX：MADKX = SMA(DKX, M)
    4. 计算DKX_OSC：DKX_OSC = (DKX - MADKX) × 100

    使用场景：
    - DKX > MADKX 时，表示多头市场
    - DKX < MADKX 时，表示空头市场
    - DKX上升时，表示上升趋势
    - DKX下降时，表示下降趋势
    - DKX上穿MADKX时，为买入信号（金叉）
    - DKX下穿MADKX时，为卖出信号（死叉）
    - DKX_OSC > 0 时，表示量价配合良好
    - DKX_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Open'、'High'、'Low'、'Close'列
    - period1: 第一计算周期，默认10
    - period2: 第二计算周期，默认5

    输出参数：
    - MID: 中值
    - DKX: 多空线
    - MADKX: 多空线移动平均
    - DKX_OSC: 多空线震荡指标
    - dkx_above_madkx: DKX在MADKX上方信号
    - dkx_below_madkx: DKX在MADKX下方信号
    - dkx_rising: DKX上升信号
    - dkx_falling: DKX下降信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - DKXExtended是趋势指标，在单边市中表现最好
    - DKXExtended在震荡市中会产生频繁的假信号
    - DKXExtended反应较慢，不适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：DKX和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DKXExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=10, period2=5):
        """
        初始化DKXExtended指标

        Parameters
        ----------
        period1 : int, default 10
            第一计算周期
        period2 : int, default 5
            第二计算周期
        """
        self.period1 = period1
        self.period2 = period2

    def calculate(self, data):
        """
        计算DKXExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Open'、'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DKXExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MID'] = (3 * df['Close'] + df['High'] + df['Low'] + df['Open']) / 6
        mid_sum = df['MID'] * 2 + df['MID'].shift(1) + df['MID'].shift(2) + df['MID'].shift(3)
        df['DKX'] = (mid_sum / 5).rolling(window=self.period1).mean()
        df['MADKX'] = df['DKX'].rolling(window=self.period2).mean()
        df['DKX_OSC'] = (df['DKX'] - df['MADKX']) * 100

        df['dkx_above_madkx'] = df['DKX'] > df['MADKX']
        df['dkx_below_madkx'] = df['DKX'] < df['MADKX']
        df['dkx_rising'] = df['DKX'] > df['DKX'].shift(1)
        df['dkx_falling'] = df['DKX'] < df['DKX'].shift(1)
        df['golden_cross'] = (df['DKX'] > df['MADKX']) & (df['DKX'].shift(1) <= df['MADKX'].shift(1))
        df['death_cross'] = (df['DKX'] < df['MADKX']) & (df['DKX'].shift(1) >= df['MADKX'].shift(1))

        return df[['MID', 'DKX', 'MADKX', 'DKX_OSC',
                   'dkx_above_madkx', 'dkx_below_madkx',
                   'dkx_rising', 'dkx_falling',
                   'golden_cross', 'death_cross']]
