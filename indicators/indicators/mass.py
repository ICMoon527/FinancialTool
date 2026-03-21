import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MASS(BaseIndicator):
    """
    MASS（Mass Index）指标，中文称为梅斯线指标。

    MASS指标是由唐纳德·多西（Donald Dorsey）提出的，是一种基于高低波幅的震荡指标。
    MASS指标通过计算高低波幅的移动平均比值，来衡量价格的震荡程度和趋势变化。

    计算公式：
    1. 计算HL = High - Low
    2. 计算HL_MA1 = MA(HL, N)
    3. 计算HL_MA2 = MA(HL_MA1, N)
    4. 计算Ratio = HL_MA1 / HL_MA2
    5. 计算MASS = Σ(Ratio, M)

    使用场景：
    - MASS < 25 时，表示震荡程度较低，可能出现趋势行情，为买入信号
    - MASS > 27 时，表示震荡程度较高，可能出现反转，为卖出信号
    - MASS由下往上穿越25时，为买入信号
    - MASS由上往下跌破27时，为卖出信号
    - MASS在25-27之间时，表示震荡市
    - MASS创新低时，表示震荡程度极低，可能出现大趋势

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'列
    - ma_period: 移动平均周期，默认9
    - sum_period: 累加周期，默认25

    输出参数：
    - HL: 高低波幅
    - HL_MA1: HL的第一次移动平均
    - HL_MA2: HL_MA1的第二次移动平均
    - Ratio: 比值
    - MASS: 梅斯线指标
    - mass_low: MASS < 25信号
    - mass_high: MASS > 27信号
    - mass_cross_25_up: MASS上穿25信号
    - mass_cross_27_down: MASS下穿27信号
    - mass_new_low: MASS创新低信号
    - mass_new_high: MASS创新高信号

    注意事项：
    - MASS是震荡指标，反应价格震荡程度
    - MASS在震荡市中表现较好
    - MASS在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 震荡判断：MASS < 25 表示震荡程度较低
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的MASS
    4. 趋势过滤：先判断大趋势，顺势操作
    5. 止损设置：设置严格止损，控制风险
    6. 灵活调整：根据市场波动性调整参数
    7. 等待确认：信号出现后等待1-2个交易日确认
    8. 历史对比：对比历史MASS的高低点
    """

    def __init__(self, ma_period=9, sum_period=25):
        """
        初始化MASS指标

        Parameters
        ----------
        ma_period : int, default 9
            移动平均周期
        sum_period : int, default 25
            累加周期
        """
        self.ma_period = ma_period
        self.sum_period = sum_period

    def calculate(self, data):
        """
        计算MASS指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MASS计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['HL'] = df['High'] - df['Low']
        df['HL_MA1'] = df['HL'].rolling(window=self.ma_period).mean()
        df['HL_MA2'] = df['HL_MA1'].rolling(window=self.ma_period).mean()
        df['Ratio'] = df['HL_MA1'] / df['HL_MA2'].replace(0, np.nan)
        df['MASS'] = df['Ratio'].rolling(window=self.sum_period).sum()

        df['mass_low'] = df['MASS'] < 25
        df['mass_high'] = df['MASS'] > 27
        df['mass_cross_25_up'] = (df['MASS'] > 25) & (df['MASS'].shift(1) <= 25)
        df['mass_cross_27_down'] = (df['MASS'] < 27) & (df['MASS'].shift(1) >= 27)
        df['mass_new_low'] = df['MASS'] == df['MASS'].rolling(window=self.sum_period).min()
        df['mass_new_high'] = df['MASS'] == df['MASS'].rolling(window=self.sum_period).max()

        return df[['HL', 'HL_MA1', 'HL_MA2', 'Ratio', 'MASS', 'mass_low', 'mass_high',
                   'mass_cross_25_up', 'mass_cross_27_down', 'mass_new_low', 'mass_new_high']]
