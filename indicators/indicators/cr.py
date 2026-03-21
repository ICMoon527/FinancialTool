import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CR(BaseIndicator):
    """
    CR（能量指标），中文称为能量指标。

    CR指标是由Larry Williams提出的，是一种能量指标。
    CR指标通过计算中间价的动量，来判断价格的超买超卖状态和能量变化。

    计算公式：
    1. 计算MID（中间价）：MID = (High + Low) / 2
    2. 计算MID_prev：MID_prev = MID.shift(1)
    3. 计算P1：P1 = max(High - MID_prev, 0)
    4. 计算P2：P2 = max(MID_prev - Low, 0)
    5. 计算P3：P3 = max(High - Low, 0)
    6. 计算P4：P4 = max(MID_prev - Close_prev, 0)
    7. 计算SUM_P1：SUM_P1 = sum(P1, N)
    8. 计算SUM_P2：SUM_P2 = sum(P2, N)
    9. 计算SUM_P3：SUM_P3 = sum(P3, N)
    10. 计算SUM_P4：SUM_P4 = sum(P4, N)
    11. 计算CR：CR = (SUM_P1 + SUM_P2) / (SUM_P3 + SUM_P4) × 100

    使用场景：
    - CR > 300 时，为超买信号，可能回落
    - CR < 40 时，为超卖信号，可能反弹
    - CR上升时，表示能量增加，为买入信号
    - CR下降时，表示能量减少，为卖出信号
    - CR与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认26

    输出参数：
    - MID: 中间价
    - P1: P1值
    - P2: P2值
    - P3: P3值
    - P4: P4值
    - CR: 能量指标
    - CR_MA1: CR的MA1
    - CR_MA2: CR的MA2
    - CR_MA3: CR的MA3
    - CR_MA4: CR的MA4
    - cr_overbought: CR > 300信号
    - cr_oversold: CR < 40信号
    - cr_rising: CR上升信号
    - cr_falling: CR下降信号
    - cr_above_ma1: CR在MA1上方信号
    - cr_below_ma1: CR在MA1下方信号

    注意事项：
    - CR是能量指标，在震荡市中会产生频繁的假信号
    - CR在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：CR和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的CR
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=26, ma1_period=5, ma2_period=10, ma3_period=20, ma4_period=30):
        """
        初始化CR指标

        Parameters
        ----------
        period : int, default 26
            计算周期
        ma1_period : int, default 5
            MA1周期
        ma2_period : int, default 10
            MA2周期
        ma3_period : int, default 20
            MA3周期
        ma4_period : int, default 30
            MA4周期
        """
        self.period = period
        self.ma1_period = ma1_period
        self.ma2_period = ma2_period
        self.ma3_period = ma3_period
        self.ma4_period = ma4_period

    def calculate(self, data):
        """
        计算CR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MID'] = (df['High'] + df['Low']) / 2
        mid_prev = df['MID'].shift(1)
        close_prev = df['Close'].shift(1)

        df['P1'] = np.maximum(df['High'] - mid_prev, 0)
        df['P2'] = np.maximum(mid_prev - df['Low'], 0)
        df['P3'] = np.maximum(df['High'] - df['Low'], 0)
        df['P4'] = np.maximum(mid_prev - close_prev, 0)

        sum_p1 = df['P1'].rolling(window=self.period).sum()
        sum_p2 = df['P2'].rolling(window=self.period).sum()
        sum_p3 = df['P3'].rolling(window=self.period).sum()
        sum_p4 = df['P4'].rolling(window=self.period).sum()

        df['CR'] = ((sum_p1 + sum_p2) / (sum_p3 + sum_p4)) * 100
        df['CR_MA1'] = df['CR'].rolling(window=self.ma1_period).mean()
        df['CR_MA2'] = df['CR'].rolling(window=self.ma2_period).mean()
        df['CR_MA3'] = df['CR'].rolling(window=self.ma3_period).mean()
        df['CR_MA4'] = df['CR'].rolling(window=self.ma4_period).mean()

        df['cr_overbought'] = df['CR'] > 300
        df['cr_oversold'] = df['CR'] < 40
        df['cr_rising'] = df['CR'] > df['CR'].shift(1)
        df['cr_falling'] = df['CR'] < df['CR'].shift(1)
        df['cr_above_ma1'] = df['CR'] > df['CR_MA1']
        df['cr_below_ma1'] = df['CR'] < df['CR_MA1']

        return df[['MID', 'P1', 'P2', 'P3', 'P4', 'CR', 'CR_MA1', 'CR_MA2', 'CR_MA3', 'CR_MA4',
                   'cr_overbought', 'cr_oversold', 'cr_rising', 'cr_falling',
                   'cr_above_ma1', 'cr_below_ma1']]
