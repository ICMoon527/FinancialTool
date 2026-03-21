import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DKX(BaseIndicator):
    """
    DKX（多空线指标，中文称为多空线指标。

    DKX指标是一种结合了移动平均线的趋势指标，通过计算短期和长期移动平均线的关系，
    来判断价格趋势的变化。DKX指标由两条线组成：DKX（多空线）和MADKX（多空线的移动平均线）。

    计算公式：
    1. 计算MA1 = (3 × Close + 2 × High + Low + Open) / 7
    2. 计算MA2 = MA1的N日移动平均
    3. DKX = MA2的M日移动平均
    4. MADKX = DKX的P日移动平均

    使用场景：
    - DKX在MADKX上方时，为多头市场，为买入信号
    - DKX在MADKX下方时，为空头市场，为卖出信号
    - DKX上穿MADKX时，为买入信号
    - DKX下穿MADKX时，为卖出信号
    - DKX和MADKX同时向上时，趋势健康
    - DKX和MADKX同时向下时，趋势健康

    输入参数：
    - data: DataFrame，必须包含'Open'、'High'、'Low'、'Close'列
    - period: DKX周期，默认10
    - ma_period: MADKX周期，默认6

    输出参数：
    - MA1: 加权移动平均线
    - MA2: MA1的移动平均线
    - DKX: 多空线
    - MADKX: 多空线的移动平均线
    - dkx_above_madkx: DKX在MADKX上方信号
    - dkx_below_madkx: DKX在MADKX下方信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - DKX是趋势指标，在震荡市中会产生频繁的假信号
    - DKX在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - DKX反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：DKX和MADKX同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DKX
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：金叉死叉后等待2-3个交易日确认
    """

    def __init__(self, period=10, ma_period=6):
        """
        初始化DKX指标

        Parameters
        ----------
        period : int, default 10
            DKX周期
        ma_period : int, default 6
            MADKX周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算DKX指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Open'、'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DKX计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA1'] = (3 * df['Close'] + 2 * df['High'] + df['Low'] + df['Open']) / 7
        df['MA2'] = df['MA1'].rolling(window=self.period).mean()
        df['DKX'] = df['MA2'].rolling(window=self.period).mean()
        df['MADKX'] = df['DKX'].rolling(window=self.ma_period).mean()

        df['dkx_above_madkx'] = df['DKX'] > df['MADKX']
        df['dkx_below_madkx'] = df['DKX'] < df['MADKX']
        df['golden_cross'] = (df['DKX'] > df['MADKX']) & (df['DKX'].shift(1) <= df['MADKX'].shift(1))
        df['death_cross'] = (df['DKX'] < df['MADKX']) & (df['DKX'].shift(1) >= df['MADKX'].shift(1))

        return df[['MA1', 'MA2', 'DKX', 'MADKX', 'dkx_above_madkx', 'dkx_below_madkx', 
                   'golden_cross', 'death_cross']]
