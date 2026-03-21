import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VRExtended(BaseIndicator):
    """
    VRExtended（成交量变异率扩展），中文称为成交量变异率扩展。

    VRExtended指标是VR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    VRExtended指标结合了成交量和价格变化，来反应市场的资金流向。

    计算公式：
    1. 计算AVS：AVS = sum(Volume if Close > Close_prev, N)
    2. 计算BVS：BVS = sum(Volume if Close < Close_prev, N)
    3. 计算CVS：CVS = sum(Volume if Close == Close_prev, N)
    4. 计算VR：VR = ((AVS + 0.5 × CVS) / (BVS + 0.5 × CVS)) × 100
    5. 计算VR_MA1：VR_MA1 = EMA(VR, N)
    6. 计算VR_MA2：VR_MA2 = EMA(VR_MA1, M)
    7. 计算VR_OSC：VR_OSC = (VR_MA1 - VR_MA2) × 100

    使用场景：
    - VR > 100 时，表示多头市场
    - VR < 100 时，表示空头市场
    - VR > 150 时，表示超买，为卖出信号
    - VR < 70 时，表示超卖，为买入信号
    - VR上升时，表示量价配合改善，为买入信号
    - VR下降时，表示量价配合恶化，为卖出信号
    - VR_OSC > 0 时，表示量价配合良好
    - VR_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period: 计算周期，默认26
    - period1: 第一计算周期，默认5
    - period2: 第二计算周期，默认10

    输出参数：
    - AVS: 上涨成交量
    - BVS: 下跌成交量
    - CVS: 平盘成交量
    - VR: 成交量变异率
    - VR_MA1: VR的第一移动平均线
    - VR_MA2: VR的第二移动平均线
    - VR_OSC: VR的震荡指标
    - vr_above_100: VR > 100信号
    - vr_below_100: VR < 100信号
    - vr_overbought: VR超买信号（>150）
    - vr_oversold: VR超卖信号（<70）
    - vr_rising: VR上升信号
    - vr_falling: VR下降信号
    - vr_osc_positive: VR_OSC > 0信号
    - vr_osc_negative: VR_OSC < 0信号

    注意事项：
    - VRExtended是量价指标，反应资金流动
    - VRExtended在震荡市中会产生频繁的假信号
    - VRExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VR和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的VRExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=26, period1=5, period2=10):
        """
        初始化VRExtended指标

        Parameters
        ----------
        period : int, default 26
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
        计算VRExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VRExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        price_change = df['Close'].diff()
        df['AVS'] = np.where(price_change > 0, df['Volume'], 0)
        df['BVS'] = np.where(price_change < 0, df['Volume'], 0)
        df['CVS'] = np.where(price_change == 0, df['Volume'], 0)

        avs_sum = df['AVS'].rolling(window=self.period).sum()
        bvs_sum = df['BVS'].rolling(window=self.period).sum()
        cvs_sum = df['CVS'].rolling(window=self.period).sum()
        df['VR'] = ((avs_sum + 0.5 * cvs_sum) / (bvs_sum + 0.5 * cvs_sum).replace(0, np.nan)) * 100
        df['VR'] = df['VR'].fillna(100)
        df['VR_MA1'] = df['VR'].ewm(span=self.period1, adjust=False).mean()
        df['VR_MA2'] = df['VR_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['VR_OSC'] = (df['VR_MA1'] - df['VR_MA2']) * 100

        df['vr_above_100'] = df['VR'] > 100
        df['vr_below_100'] = df['VR'] < 100
        df['vr_overbought'] = df['VR'] > 150
        df['vr_oversold'] = df['VR'] < 70
        df['vr_rising'] = df['VR'] > df['VR'].shift(1)
        df['vr_falling'] = df['VR'] < df['VR'].shift(1)
        df['vr_osc_positive'] = df['VR_OSC'] > 0
        df['vr_osc_negative'] = df['VR_OSC'] < 0

        return df[['AVS', 'BVS', 'CVS', 'VR', 'VR_MA1', 'VR_MA2', 'VR_OSC',
                   'vr_above_100', 'vr_below_100',
                   'vr_overbought', 'vr_oversold',
                   'vr_rising', 'vr_falling',
                   'vr_osc_positive', 'vr_osc_negative']]
