import pandas as pd
import numpy as np
from ..base import BaseIndicator


class LWRExtended(BaseIndicator):
    """
    LWRExtended（威廉指标扩展），中文称为威廉指标扩展。

    LWRExtended指标是LWR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    LWRExtended指标结合了最高价、最低价和收盘价，来反应价格的超买超卖。

    计算公式：
    1. 计算HHV：HHV = max(High, N)
    2. 计算LLV：LLV = min(Low, N)
    3. 计算R%：R% = (HHV - Close) / (HHV - LLV) × 100
    4. 计算LWR：LWR = 100 - R%
    5. 计算LWR_MA1：LWR_MA1 = EMA(LWR, M)
    6. 计算LWR_MA2：LWR_MA2 = EMA(LWR_MA1, P)
    7. 计算LWR_OSC：LWR_OSC = (LWR_MA1 - LWR_MA2) × 100

    使用场景：
    - LWR > 80 时，表示超卖，为买入信号
    - LWR < 20 时，表示超买，为卖出信号
    - LWR在50以上时，表示强势
    - LWR在50以下时，表示弱势
    - LWR从低位上升穿破20时，为买入信号
    - LWR从高位下降穿破80时，为卖出信号
    - LWR_OSC > 0 时，表示量价配合良好
    - LWR_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认10
    - ma_period1: 第一移动平均周期，默认6
    - ma_period2: 第二移动平均周期，默认6

    输出参数：
    - HHV: 最高价的最高值
    - LLV: 最低价的最低值
    - R: 威廉指标
    - LWR: LWR指标
    - LWR_MA1: LWR的第一移动平均线
    - LWR_MA2: LWR的第二移动平均线
    - LWR_OSC: LWR的震荡指标
    - lwr_overbought: LWR > 80超买信号
    - lwr_oversold: LWR < 20超卖信号
    - lwr_above_50: LWR > 50信号
    - lwr_below_50: LWR < 50信号
    - lwr_rising: LWR上升信号
    - lwr_falling: LWR下降信号
    - lwr_cross_up_20: LWR上穿20信号
    - lwr_cross_down_80: LWR下穿80信号

    注意事项：
    - LWRExtended是超买超卖指标，在震荡市中表现最好
    - LWRExtended在单边市中会产生频繁的假信号
    - LWRExtended反应较快，适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：LWR和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的LWRExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period1=6, ma_period2=6):
        """
        初始化LWRExtended指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        ma_period1 : int, default 6
            第一移动平均周期
        ma_period2 : int, default 6
            第二移动平均周期
        """
        self.period = period
        self.ma_period1 = ma_period1
        self.ma_period2 = ma_period2

    def calculate(self, data):
        """
        计算LWRExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含LWRExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['HHV'] = df['High'].rolling(window=self.period).max()
        df['LLV'] = df['Low'].rolling(window=self.period).min()
        df['R'] = ((df['HHV'] - df['Close']) / (df['HHV'] - df['LLV']).replace(0, np.nan)) * 100
        df['R'] = df['R'].fillna(50)
        df['LWR'] = 100 - df['R']
        df['LWR_MA1'] = df['LWR'].ewm(span=self.ma_period1, adjust=False).mean()
        df['LWR_MA2'] = df['LWR_MA1'].ewm(span=self.ma_period2, adjust=False).mean()
        df['LWR_OSC'] = (df['LWR_MA1'] - df['LWR_MA2']) * 100

        df['lwr_overbought'] = df['LWR'] > 80
        df['lwr_oversold'] = df['LWR'] < 20
        df['lwr_above_50'] = df['LWR'] > 50
        df['lwr_below_50'] = df['LWR'] < 50
        df['lwr_rising'] = df['LWR'] > df['LWR'].shift(1)
        df['lwr_falling'] = df['LWR'] < df['LWR'].shift(1)
        df['lwr_cross_up_20'] = (df['LWR'] > 20) & (df['LWR'].shift(1) <= 20)
        df['lwr_cross_down_80'] = (df['LWR'] < 80) & (df['LWR'].shift(1) >= 80)

        return df[['HHV', 'LLV', 'R', 'LWR', 'LWR_MA1', 'LWR_MA2', 'LWR_OSC',
                   'lwr_overbought', 'lwr_oversold',
                   'lwr_above_50', 'lwr_below_50',
                   'lwr_rising', 'lwr_falling',
                   'lwr_cross_up_20', 'lwr_cross_down_80']]
