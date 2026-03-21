import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MASSExtended(BaseIndicator):
    """
    MASSExtended（梅斯线扩展），中文称为梅斯线扩展。

    MASSExtended指标是MASS指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    MASSExtended指标结合了最高价和最低价，来反应市场的波动性。

    计算公式：
    1. 计算AHL：AHL = High - Low
    2. 计算EMA1：EMA1 = EMA(AHL, N)
    3. 计算EMA2：EMA2 = EMA(EMA1, N)
    4. 计算Ratio：Ratio = EMA1 / EMA2
    5. 计算MASS：MASS = sum(Ratio, M)
    6. 计算MASS_MA1：MASS_MA1 = EMA(MASS, P)
    7. 计算MASS_MA2：MASS_MA2 = EMA(MASS_MA1, Q)
    8. 计算MASS_OSC：MASS_OSC = (MASS_MA1 - MASS_MA2) × 100

    使用场景：
    - MASS > 27 时，表示超买，为卖出信号
    - MASS < 25 时，表示超卖，为买入信号
    - MASS上升时，表示波动性增加
    - MASS下降时，表示波动性减少
    - MASS上穿25时，为买入信号
    - MASS下穿27时，为卖出信号
    - MASS_OSC > 0 时，表示量价配合良好
    - MASS_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'列
    - period1: 第一计算周期，默认10
    - period2: 第二计算周期，默认25
    - ma_period1: 第一移动平均周期，默认6
    - ma_period2: 第二移动平均周期，默认6

    输出参数：
    - AHL: 最高价与最低价的差值
    - EMA1: AHL的第一指数移动平均线
    - EMA2: AHL的第二指数移动平均线
    - Ratio: EMA1与EMA2的比值
    - MASS: 梅斯线
    - MASS_MA1: MASS的第一移动平均线
    - MASS_MA2: MASS的第二移动平均线
    - MASS_OSC: MASS的震荡指标
    - mass_overbought: MASS > 27超买信号
    - mass_oversold: MASS < 25超卖信号
    - mass_rising: MASS上升信号
    - mass_falling: MASS下降信号
    - mass_ma1_above_ma2: MASS_MA1在MASS_MA2上方信号
    - mass_ma1_below_ma2: MASS_MA1在MASS_MA2下方信号
    - mass_osc_positive: MASS_OSC > 0信号
    - mass_osc_negative: MASS_OSC < 0信号
    - mass_cross_up_25: MASS上穿25信号
    - mass_cross_down_27: MASS下穿27信号

    注意事项：
    - MASSExtended是波动性指标，反应市场波动
    - MASSExtended在震荡市中会产生频繁的假信号
    - MASSExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：MASS和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的MASSExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=10, period2=25, ma_period1=6, ma_period2=6):
        """
        初始化MASSExtended指标

        Parameters
        ----------
        period1 : int, default 10
            第一计算周期
        period2 : int, default 25
            第二计算周期
        ma_period1 : int, default 6
            第一移动平均周期
        ma_period2 : int, default 6
            第二移动平均周期
        """
        self.period1 = period1
        self.period2 = period2
        self.ma_period1 = ma_period1
        self.ma_period2 = ma_period2

    def calculate(self, data):
        """
        计算MASSExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MASSExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['AHL'] = df['High'] - df['Low']
        df['EMA1'] = df['AHL'].ewm(span=self.period1, adjust=False).mean()
        df['EMA2'] = df['EMA1'].ewm(span=self.period1, adjust=False).mean()
        df['Ratio'] = df['EMA1'] / df['EMA2'].replace(0, np.nan)
        df['Ratio'] = df['Ratio'].fillna(1)
        df['MASS'] = df['Ratio'].rolling(window=self.period2).sum()
        df['MASS_MA1'] = df['MASS'].ewm(span=self.ma_period1, adjust=False).mean()
        df['MASS_MA2'] = df['MASS_MA1'].ewm(span=self.ma_period2, adjust=False).mean()
        df['MASS_OSC'] = (df['MASS_MA1'] - df['MASS_MA2']) * 100

        df['mass_overbought'] = df['MASS'] > 27
        df['mass_oversold'] = df['MASS'] < 25
        df['mass_rising'] = df['MASS'] > df['MASS'].shift(1)
        df['mass_falling'] = df['MASS'] < df['MASS'].shift(1)
        df['mass_ma1_above_ma2'] = df['MASS_MA1'] > df['MASS_MA2']
        df['mass_ma1_below_ma2'] = df['MASS_MA1'] < df['MASS_MA2']
        df['mass_osc_positive'] = df['MASS_OSC'] > 0
        df['mass_osc_negative'] = df['MASS_OSC'] < 0
        df['mass_cross_up_25'] = (df['MASS'] > 25) & (df['MASS'].shift(1) <= 25)
        df['mass_cross_down_27'] = (df['MASS'] < 27) & (df['MASS'].shift(1) >= 27)

        return df[['AHL', 'EMA1', 'EMA2', 'Ratio', 'MASS', 'MASS_MA1', 'MASS_MA2', 'MASS_OSC',
                   'mass_overbought', 'mass_oversold',
                   'mass_rising', 'mass_falling',
                   'mass_ma1_above_ma2', 'mass_ma1_below_ma2',
                   'mass_osc_positive', 'mass_osc_negative',
                   'mass_cross_up_25', 'mass_cross_down_27']]
