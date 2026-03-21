import pandas as pd
import numpy as np
from ..base import BaseIndicator


class WRExtended(BaseIndicator):
    """
    WRExtended（威廉指标扩展），中文称为威廉指标扩展。

    WRExtended指标是WR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    WRExtended指标结合了最高价、最低价和收盘价，来反应价格的超买超卖状态。

    计算公式：
    1. 计算WR：WR = -100 × (HHV(High, N) - Close) / (HHV(High, N) - LLV(Low, N))
    2. 计算WR_MA1：WR_MA1 = EMA(WR, N)
    3. 计算WR_MA2：WR_MA2 = EMA(WR_MA1, M)
    4. 计算WR_OSC：WR_OSC = (WR_MA1 - WR_MA2) × 100

    使用场景：
    - WR > -20 时，表示超买，为卖出信号
    - WR < -80 时，表示超卖，为买入信号
    - WR上升时，表示价格上涨，为买入信号
    - WR下降时，表示价格下跌，为卖出信号
    - WR_OSC > 0 时，表示量价配合良好
    - WR_OSC < 0 时，表示量价背离
    - WR与价格同步时，趋势健康
    - WR与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14
    - period1: 第一计算周期，默认5
    - period2: 第二计算周期，默认10

    输出参数：
    - WR: 威廉指标
    - WR_MA1: WR的第一移动平均线
    - WR_MA2: WR的第二移动平均线
    - WR_OSC: WR的震荡指标
    - wr_overbought: WR超买信号（>-20）
    - wr_oversold: WR超卖信号（<-80）
    - wr_rising: WR上升信号
    - wr_falling: WR下降信号
    - wr_osc_positive: WR_OSC > 0信号
    - wr_osc_negative: WR_OSC < 0信号
    - wr_osc_rising: WR_OSC上升信号
    - wr_osc_falling: WR_OSC下降信号

    注意事项：
    - WRExtended是摆动指标，在震荡市中表现最好
    - WRExtended在单边市中会产生钝化现象
    - WRExtended反应较快，适合中短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和WR同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的WRExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14, period1=5, period2=10):
        """
        初始化WRExtended指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        period1 : int, default 5
            第一计算周期
        period2 : int, default 10
            第二计算周期
        """
        self.period = period
        self.period1 = period1
        self.period2 = period2

    def calculate(self, data):
        """
        计算WRExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含WRExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_max = df['High'].rolling(window=self.period).max()
        low_min = df['Low'].rolling(window=self.period).min()
        df['WR'] = -100 * (high_max - df['Close']) / (high_max - low_min).replace(0, np.nan)
        df['WR'] = df['WR'].fillna(-50)
        df['WR_MA1'] = df['WR'].ewm(span=self.period1, adjust=False).mean()
        df['WR_MA2'] = df['WR_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['WR_OSC'] = (df['WR_MA1'] - df['WR_MA2']) * 100

        df['wr_overbought'] = df['WR'] > -20
        df['wr_oversold'] = df['WR'] < -80
        df['wr_rising'] = df['WR'] > df['WR'].shift(1)
        df['wr_falling'] = df['WR'] < df['WR'].shift(1)
        df['wr_osc_positive'] = df['WR_OSC'] > 0
        df['wr_osc_negative'] = df['WR_OSC'] < 0
        df['wr_osc_rising'] = df['WR_OSC'] > df['WR_OSC'].shift(1)
        df['wr_osc_falling'] = df['WR_OSC'] < df['WR_OSC'].shift(1)

        return df[['WR', 'WR_MA1', 'WR_MA2', 'WR_OSC',
                   'wr_overbought', 'wr_oversold',
                   'wr_rising', 'wr_falling',
                   'wr_osc_positive', 'wr_osc_negative',
                   'wr_osc_rising', 'wr_osc_falling']]
