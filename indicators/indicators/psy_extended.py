import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PSYExtended(BaseIndicator):
    """
    PSYExtended（心理线扩展），中文称为心理线扩展。

    PSYExtended指标是PSY指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    PSYExtended指标结合了上涨天数和总天数，来反应市场的人气。

    计算公式：
    1. 计算PSY：PSY = (上涨天数 / 总天数) × 100
    2. 计算PSY_MA1：PSY_MA1 = EMA(PSY, N)
    3. 计算PSY_MA2：PSY_MA2 = EMA(PSY_MA1, M)
    4. 计算PSY_OSC：PSY_OSC = (PSY_MA1 - PSY_MA2) × 100

    使用场景：
    - PSY > 75 时，表示超买，为卖出信号
    - PSY < 25 时，表示超卖，为买入信号
    - PSY > 50 时，表示多头市场
    - PSY < 50 时，表示空头市场
    - PSY_OSC > 0 时，表示量价配合良好
    - PSY_OSC < 0 时，表示量价背离
    - PSY上升时，表示人气上升
    - PSY下降时，表示人气下降

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认12
    - period1: 第一计算周期，默认5
    - period2: 第二计算周期，默认10

    输出参数：
    - PSY: 心理线
    - PSY_MA1: PSY的第一移动平均线
    - PSY_MA2: PSY的第二移动平均线
    - PSY_OSC: PSY的震荡指标
    - psy_above_75: PSY > 75信号
    - psy_below_25: PSY < 25信号
    - psy_above_50: PSY > 50信号
    - psy_below_50: PSY < 50信号
    - psy_rising: PSY上升信号
    - psy_falling: PSY下降信号
    - psy_osc_positive: PSY_OSC > 0信号
    - psy_osc_negative: PSY_OSC < 0信号

    注意事项：
    - PSYExtended是人气指标，反应市场情绪
    - PSYExtended在震荡市中会产生频繁的假信号
    - PSYExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：PSY和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的PSYExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=12, period1=5, period2=10):
        """
        初始化PSYExtended指标

        Parameters
        ----------
        period : int, default 12
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
        计算PSYExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PSYExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        price_change = df['Close'].diff()
        up_days = (price_change > 0).astype(int)
        df['PSY'] = (up_days.rolling(window=self.period).sum() / self.period) * 100
        df['PSY_MA1'] = df['PSY'].ewm(span=self.period1, adjust=False).mean()
        df['PSY_MA2'] = df['PSY_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['PSY_OSC'] = (df['PSY_MA1'] - df['PSY_MA2']) * 100

        df['psy_above_75'] = df['PSY'] > 75
        df['psy_below_25'] = df['PSY'] < 25
        df['psy_above_50'] = df['PSY'] > 50
        df['psy_below_50'] = df['PSY'] < 50
        df['psy_rising'] = df['PSY'] > df['PSY'].shift(1)
        df['psy_falling'] = df['PSY'] < df['PSY'].shift(1)
        df['psy_osc_positive'] = df['PSY_OSC'] > 0
        df['psy_osc_negative'] = df['PSY_OSC'] < 0

        return df[['PSY', 'PSY_MA1', 'PSY_MA2', 'PSY_OSC',
                   'psy_above_75', 'psy_below_25',
                   'psy_above_50', 'psy_below_50',
                   'psy_rising', 'psy_falling',
                   'psy_osc_positive', 'psy_osc_negative']]
