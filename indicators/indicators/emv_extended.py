import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EMVExtended(BaseIndicator):
    """
    EMVExtended（简易波动指标扩展），中文称为简易波动指标扩展。

    EMVExtended指标是EMV指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    EMVExtended指标结合了成交量和价格波动，来反应市场的资金流向。

    计算公式：
    1. 计算MidPoint：MidPoint = (High + Low) / 2 - (High_prev + Low_prev) / 2
    2. 计算BoxRatio：BoxRatio = Volume / ((High - Low) × 1000000)
    3. 计算EMV：EMV = MidPoint / BoxRatio
    4. 计算EMV_MA1：EMV_MA1 = EMA(EMV, N)
    5. 计算EMV_MA2：EMV_MA2 = EMA(EMV_MA1, M)
    6. 计算EMV_OSC：EMV_OSC = (EMV_MA1 - EMV_MA2) × 100

    使用场景：
    - EMV > 0 时，表示多头市场
    - EMV < 0 时，表示空头市场
    - EMV上升时，表示资金流入，为买入信号
    - EMV下降时，表示资金流出，为卖出信号
    - EMV与价格同步时，趋势健康
    - EMV与价格背离时，可能反转
    - EMV_OSC > 0 时，表示量价配合良好
    - EMV_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Volume'列
    - period1: 第一计算周期，默认14
    - period2: 第二计算周期，默认9

    输出参数：
    - MidPoint: 中点变动
    - BoxRatio: 成交量比例
    - EMV: 简易波动指标
    - EMV_MA1: EMV的第一移动平均线
    - EMV_MA2: EMV的第二移动平均线
    - EMV_OSC: EMV的震荡指标
    - emv_positive: EMV > 0信号
    - emv_negative: EMV < 0信号
    - emv_rising: EMV上升信号
    - emv_falling: EMV下降信号
    - emv_above_ma: EMV在MA上方信号
    - emv_below_ma: EMV在MA下方信号

    注意事项：
    - EMVExtended是量价指标，反应资金流动
    - EMVExtended在震荡市中会产生频繁的假信号
    - EMVExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：EMV和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的EMVExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=14, period2=9):
        """
        初始化EMVExtended指标

        Parameters
        ----------
        period1 : int, default 14
            第一计算周期
        period2 : int, default 9
            第二计算周期
        """
        self.period1 = period1
        self.period2 = period2

    def calculate(self, data):
        """
        计算EMVExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含EMVExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        mid_point = ((df['High'] + df['Low']) / 2) - ((prev_high + prev_low) / 2)
        box_ratio = df['Volume'] / ((df['High'] - df['Low']).replace(0, np.nan) * 1000000)
        box_ratio = box_ratio.replace(0, np.nan)
        df['MidPoint'] = mid_point
        df['BoxRatio'] = box_ratio
        df['EMV'] = mid_point / box_ratio.replace(0, np.nan)
        df['EMV'] = df['EMV'].fillna(0)
        df['EMV_MA1'] = df['EMV'].ewm(span=self.period1, adjust=False).mean()
        df['EMV_MA2'] = df['EMV_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['EMV_OSC'] = (df['EMV_MA1'] - df['EMV_MA2']) * 100

        df['emv_positive'] = df['EMV'] > 0
        df['emv_negative'] = df['EMV'] < 0
        df['emv_rising'] = df['EMV'] > df['EMV'].shift(1)
        df['emv_falling'] = df['EMV'] < df['EMV'].shift(1)
        df['emv_above_ma'] = df['EMV'] > df['EMV_MA1']
        df['emv_below_ma'] = df['EMV'] < df['EMV_MA1']

        return df[['MidPoint', 'BoxRatio', 'EMV', 'EMV_MA1', 'EMV_MA2', 'EMV_OSC',
                   'emv_positive', 'emv_negative',
                   'emv_rising', 'emv_falling',
                   'emv_above_ma', 'emv_below_ma']]
