import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ROCExtended(BaseIndicator):
    """
    ROCExtended（变动率扩展），中文称为变动率扩展。

    ROCExtended指标是ROC指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    ROCExtended指标结合了价格变化率，来反应价格的动量变化。

    计算公式：
    1. 计算ROC：ROC = ((Close - Close_prev_N) / Close_prev_N) × 100
    2. 计算ROC_MA1：ROC_MA1 = EMA(ROC, N)
    3. 计算ROC_MA2：ROC_MA2 = EMA(ROC_MA1, M)
    4. 计算ROC_OSC：ROC_OSC = (ROC_MA1 - ROC_MA2) × 100

    使用场景：
    - ROC > 0 时，表示上升趋势
    - ROC < 0 时，表示下降趋势
    - ROC上升时，表示动量增强，为买入信号
    - ROC下降时，表示动量减弱，为卖出信号
    - ROC_OSC > 0 时，表示量价配合良好
    - ROC_OSC < 0 时，表示量价背离
    - ROC与价格同步时，趋势健康
    - ROC与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认12
    - period1: 第一计算周期，默认5
    - period2: 第二计算周期，默认10

    输出参数：
    - ROC: 变动率
    - ROC_MA1: ROC的第一移动平均线
    - ROC_MA2: ROC的第二移动平均线
    - ROC_OSC: ROC的震荡指标
    - roc_positive: ROC > 0信号
    - roc_negative: ROC < 0信号
    - roc_rising: ROC上升信号
    - roc_falling: ROC下降信号
    - roc_osc_positive: ROC_OSC > 0信号
    - roc_osc_negative: ROC_OSC < 0信号
    - roc_osc_rising: ROC_OSC上升信号
    - roc_osc_falling: ROC_OSC下降信号

    注意事项：
    - ROCExtended是动量指标，在单边市中表现最好
    - ROCExtended在震荡市中会产生频繁的假信号
    - ROCExtended反应较快，适合中短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ROC和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ROCExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=12, period1=5, period2=10):
        """
        初始化ROCExtended指标

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
        计算ROCExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ROCExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['ROC'] = ((df['Close'] - df['Close'].shift(self.period)) / df['Close'].shift(self.period).replace(0, np.nan)) * 100
        df['ROC'] = df['ROC'].fillna(0)
        df['ROC_MA1'] = df['ROC'].ewm(span=self.period1, adjust=False).mean()
        df['ROC_MA2'] = df['ROC_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['ROC_OSC'] = (df['ROC_MA1'] - df['ROC_MA2']) * 100

        df['roc_positive'] = df['ROC'] > 0
        df['roc_negative'] = df['ROC'] < 0
        df['roc_rising'] = df['ROC'] > df['ROC'].shift(1)
        df['roc_falling'] = df['ROC'] < df['ROC'].shift(1)
        df['roc_osc_positive'] = df['ROC_OSC'] > 0
        df['roc_osc_negative'] = df['ROC_OSC'] < 0
        df['roc_osc_rising'] = df['ROC_OSC'] > df['ROC_OSC'].shift(1)
        df['roc_osc_falling'] = df['ROC_OSC'] < df['ROC_OSC'].shift(1)

        return df[['ROC', 'ROC_MA1', 'ROC_MA2', 'ROC_OSC',
                   'roc_positive', 'roc_negative',
                   'roc_rising', 'roc_falling',
                   'roc_osc_positive', 'roc_osc_negative',
                   'roc_osc_rising', 'roc_osc_falling']]
