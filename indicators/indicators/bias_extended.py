import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BIASExtended(BaseIndicator):
    """
    BIASExtended（乖离率扩展），中文称为乖离率扩展。

    BIASExtended指标是BIAS指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    BIASExtended指标结合了价格与移动平均线的偏离程度，来反应价格的超买超卖状态。

    计算公式：
    1. 计算MA：MA = SMA(Close, N)
    2. 计算BIAS：BIAS = ((Close - MA) / MA) × 100
    3. 计算BIAS_MA1：BIAS_MA1 = EMA(BIAS, N)
    4. 计算BIAS_MA2：BIAS_MA2 = EMA(BIAS_MA1, M)
    5. 计算BIAS_OSC：BIAS_OSC = (BIAS_MA1 - BIAS_MA2) × 100

    使用场景：
    - BIAS > 0 时，表示价格在MA上方
    - BIAS < 0 时，表示价格在MA下方
    - BIAS > 10 时，表示超买，为卖出信号
    - BIAS < -10 时，表示超卖，为买入信号
    - BIAS_OSC > 0 时，表示量价配合良好
    - BIAS_OSC < 0 时，表示量价背离
    - BIAS上升时，表示上升趋势
    - BIAS下降时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - ma_period: MA周期，默认10
    - period1: 第一计算周期，默认5
    - period2: 第二计算周期，默认10

    输出参数：
    - MA: 移动平均线
    - BIAS: 乖离率
    - BIAS_MA1: BIAS的第一移动平均线
    - BIAS_MA2: BIAS的第二移动平均线
    - BIAS_OSC: BIAS的震荡指标
    - bias_positive: BIAS > 0信号
    - bias_negative: BIAS < 0信号
    - bias_overbought: BIAS超买信号（>10）
    - bias_oversold: BIAS超卖信号（<-10）
    - bias_rising: BIAS上升信号
    - bias_falling: BIAS下降信号
    - bias_osc_positive: BIAS_OSC > 0信号
    - bias_osc_negative: BIAS_OSC < 0信号

    注意事项：
    - BIASExtended是摆动指标，在震荡市中表现最好
    - BIASExtended在单边市中会产生钝化现象
    - BIASExtended反应较快，适合中短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和BIAS同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的BIASExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10, period1=5, period2=10):
        """
        初始化BIASExtended指标

        Parameters
        ----------
        ma_period : int, default 10
            MA周期
        period1 : int, default 5
            第一计算周期
        period2 : int, default 10
            第二计算周期
        """
        self.ma_period = ma_period
        self.period1 = period1
        self.period2 = period2

    def calculate(self, data):
        """
        计算BIASExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BIASExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA'] = df['Close'].rolling(window=self.ma_period).mean()
        df['BIAS'] = ((df['Close'] - df['MA']) / df['MA'].replace(0, np.nan)) * 100
        df['BIAS'] = df['BIAS'].fillna(0)
        df['BIAS_MA1'] = df['BIAS'].ewm(span=self.period1, adjust=False).mean()
        df['BIAS_MA2'] = df['BIAS_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['BIAS_OSC'] = (df['BIAS_MA1'] - df['BIAS_MA2']) * 100

        df['bias_positive'] = df['BIAS'] > 0
        df['bias_negative'] = df['BIAS'] < 0
        df['bias_overbought'] = df['BIAS'] > 10
        df['bias_oversold'] = df['BIAS'] < -10
        df['bias_rising'] = df['BIAS'] > df['BIAS'].shift(1)
        df['bias_falling'] = df['BIAS'] < df['BIAS'].shift(1)
        df['bias_osc_positive'] = df['BIAS_OSC'] > 0
        df['bias_osc_negative'] = df['BIAS_OSC'] < 0

        return df[['MA', 'BIAS', 'BIAS_MA1', 'BIAS_MA2', 'BIAS_OSC',
                   'bias_positive', 'bias_negative',
                   'bias_overbought', 'bias_oversold',
                   'bias_rising', 'bias_falling',
                   'bias_osc_positive', 'bias_osc_negative']]
