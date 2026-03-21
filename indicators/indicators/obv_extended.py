import pandas as pd
import numpy as np
from ..base import BaseIndicator


class OBVExtended(BaseIndicator):
    """
    OBVExtended（能量潮扩展），中文称为能量潮扩展。

    OBVExtended指标是OBV指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    OBVExtended指标结合了成交量和价格变化，来反应市场的资金流向。

    计算公式：
    1. 计算OBV：OBV = sum(Volume if Close > Close_prev else -Volume if Close < Close_prev else 0)
    2. 计算OBV_MA1：OBV_MA1 = EMA(OBV, N)
    3. 计算OBV_MA2：OBV_MA2 = EMA(OBV_MA1, M)
    4. 计算OBV_OSC：OBV_OSC = (OBV_MA1 - OBV_MA2) × 100

    使用场景：
    - OBV上升时，表示资金流入，为买入信号
    - OBV下降时，表示资金流出，为卖出信号
    - OBV_OSC > 0 时，表示量价配合良好
    - OBV_OSC < 0 时，表示量价背离
    - OBV_OSC上升时，表示量价配合改善，为买入信号
    - OBV_OSC下降时，表示量价配合恶化，为卖出信号
    - OBV与价格同步时，趋势健康
    - OBV与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'和'Volume'列
    - period1: 第一计算周期，默认5
    - period2: 第二计算周期，默认10

    输出参数：
    - OBV: 能量潮
    - OBV_MA1: OBV的第一移动平均线
    - OBV_MA2: OBV的第二移动平均线
    - OBV_OSC: OBV的震荡指标
    - obv_osc_positive: OBV_OSC > 0信号
    - obv_osc_negative: OBV_OSC < 0信号
    - obv_osc_rising: OBV_OSC上升信号
    - obv_osc_falling: OBV_OSC下降信号
    - obv_osc_cross_zero_up: OBV_OSC上穿0线信号
    - obv_osc_cross_zero_down: OBV_OSC下穿0线信号

    注意事项：
    - OBVExtended是量价指标，反应资金流动
    - OBVExtended在震荡市中会产生频繁的假信号
    - OBVExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：OBV和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的OBVExtended
    4. 量价配合：结合价格变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=5, period2=10, ma_period=10):
        """
        初始化OBVExtended指标

        Parameters
        ----------
        period1 : int, default 5
            第一计算周期
        period2 : int, default 10
            第二计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period1 = period1
        self.period2 = period2
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算OBVExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'和'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含OBVExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        price_change = df['Close'].diff()
        obv = pd.Series(0, index=df.index)
        obv[price_change > 0] = df['Volume'][price_change > 0]
        obv[price_change < 0] = -df['Volume'][price_change < 0]
        obv[price_change == 0] = 0
        df['OBV'] = obv.cumsum()
        df['OBV_MA1'] = df['OBV'].ewm(span=self.period1, adjust=False).mean()
        df['OBV_MA2'] = df['OBV_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['OBV_OSC'] = (df['OBV_MA1'] - df['OBV_MA2']) * 100

        df['obv_osc_positive'] = df['OBV_OSC'] > 0
        df['obv_osc_negative'] = df['OBV_OSC'] < 0
        df['obv_osc_rising'] = df['OBV_OSC'] > df['OBV_OSC'].shift(1)
        df['obv_osc_falling'] = df['OBV_OSC'] < df['OBV_OSC'].shift(1)
        df['obv_osc_cross_zero_up'] = (df['OBV_OSC'] > 0) & (df['OBV_OSC'].shift(1) <= 0)
        df['obv_osc_cross_zero_down'] = (df['OBV_OSC'] < 0) & (df['OBV_OSC'].shift(1) >= 0)

        return df[['OBV', 'OBV_MA1', 'OBV_MA2', 'OBV_OSC',
                   'obv_osc_positive', 'obv_osc_negative',
                   'obv_osc_rising', 'obv_osc_falling',
                   'obv_osc_cross_zero_up', 'obv_osc_cross_zero_down']]
