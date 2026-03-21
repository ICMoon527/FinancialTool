import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TAPIExtended(BaseIndicator):
    """
    TAPIExtended（加权指数成交值扩展），中文称为加权指数成交值扩展。

    TAPIExtended指标是TAPI指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    TAPIExtended指标结合了成交量和价格，来反应市场的量价关系。

    计算公式：
    1. 计算TAPI：TAPI = Volume / Close
    2. 计算TAPI_MA1：TAPI_MA1 = EMA(TAPI, N)
    3. 计算TAPI_MA2：TAPI_MA2 = EMA(TAPI_MA1, M)
    4. 计算TAPI_OSC：TAPI_OSC = (TAPI_MA1 - TAPI_MA2) × 100

    使用场景：
    - TAPI上升时，表示量价配合良好，为买入信号
    - TAPI下降时，表示量价背离，为卖出信号
    - TAPI与价格同步时，趋势健康
    - TAPI与价格背离时，可能反转
    - TAPI_MA1 > TAPI_MA2 时，表示量价配合良好
    - TAPI_MA1 < TAPI_MA2 时，表示量价背离
    - TAPI_OSC > 0 时，表示量价配合良好
    - TAPI_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period1: 第一计算周期，默认10
    - period2: 第二计算周期，默认5

    输出参数：
    - TAPI: 加权指数成交值
    - TAPI_MA1: TAPI的第一移动平均线
    - TAPI_MA2: TAPI的第二移动平均线
    - TAPI_OSC: TAPI的震荡指标
    - tapi_rising: TAPI上升信号
    - tapi_falling: TAPI下降信号
    - tapi_ma1_above_ma2: TAPI_MA1在TAPI_MA2上方信号
    - tapi_ma1_below_ma2: TAPI_MA1在TAPI_MA2下方信号
    - tapi_osc_positive: TAPI_OSC > 0信号
    - tapi_osc_negative: TAPI_OSC < 0信号

    注意事项：
    - TAPIExtended是量价指标，反应资金流动
    - TAPIExtended在震荡市中会产生频繁的假信号
    - TAPIExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：TAPI和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的TAPIExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=10, period2=5):
        """
        初始化TAPIExtended指标

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
        计算TAPIExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含TAPIExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TAPI'] = df['Volume'] / df['Close'].replace(0, np.nan)
        df['TAPI'] = df['TAPI'].fillna(0)
        df['TAPI_MA1'] = df['TAPI'].ewm(span=self.period1, adjust=False).mean()
        df['TAPI_MA2'] = df['TAPI_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['TAPI_OSC'] = (df['TAPI_MA1'] - df['TAPI_MA2']) * 100

        df['tapi_rising'] = df['TAPI'] > df['TAPI'].shift(1)
        df['tapi_falling'] = df['TAPI'] < df['TAPI'].shift(1)
        df['tapi_ma1_above_ma2'] = df['TAPI_MA1'] > df['TAPI_MA2']
        df['tapi_ma1_below_ma2'] = df['TAPI_MA1'] < df['TAPI_MA2']
        df['tapi_osc_positive'] = df['TAPI_OSC'] > 0
        df['tapi_osc_negative'] = df['TAPI_OSC'] < 0

        return df[['TAPI', 'TAPI_MA1', 'TAPI_MA2', 'TAPI_OSC',
                   'tapi_rising', 'tapi_falling',
                   'tapi_ma1_above_ma2', 'tapi_ma1_below_ma2',
                   'tapi_osc_positive', 'tapi_osc_negative']]
