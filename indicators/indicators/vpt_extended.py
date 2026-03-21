import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VPTExtended(BaseIndicator):
    """
    VPTExtended（量价趋势扩展），中文称为量价趋势扩展。

    VPTExtended指标是VPT指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    VPTExtended指标结合了成交量和价格，来反应市场的量价趋势。

    计算公式：
    1. 计算VPT：VPT = cumsum(Volume × (Close - Close_prev) / Close_prev)
    2. 计算VPT_MA1：VPT_MA1 = EMA(VPT, N)
    3. 计算VPT_MA2：VPT_MA2 = EMA(VPT_MA1, M)
    4. 计算VPT_OSC：VPT_OSC = (VPT_MA1 - VPT_MA2) × 100

    使用场景：
    - VPT上升时，表示量价配合良好，为买入信号
    - VPT下降时，表示量价背离，为卖出信号
    - VPT与价格同步时，趋势健康
    - VPT与价格背离时，可能反转
    - VPT_MA1 > VPT_MA2 时，表示量价配合良好
    - VPT_MA1 < VPT_MA2 时，表示量价背离
    - VPT_OSC > 0 时，表示量价配合良好
    - VPT_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period1: 第一计算周期，默认10
    - period2: 第二计算周期，默认5

    输出参数：
    - VPT: 量价趋势
    - VPT_MA1: VPT的第一移动平均线
    - VPT_MA2: VPT的第二移动平均线
    - VPT_OSC: VPT的震荡指标
    - vpt_rising: VPT上升信号
    - vpt_falling: VPT下降信号
    - vpt_ma1_above_ma2: VPT_MA1在VPT_MA2上方信号
    - vpt_ma1_below_ma2: VPT_MA1在VPT_MA2下方信号
    - vpt_osc_positive: VPT_OSC > 0信号
    - vpt_osc_negative: VPT_OSC < 0信号

    注意事项：
    - VPTExtended是量价指标，反应资金流动
    - VPTExtended在震荡市中会产生频繁的假信号
    - VPTExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VPT和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的VPTExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=10, period2=5):
        """
        初始化VPTExtended指标

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
        计算VPTExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VPTExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_close = df['Close'].shift(1)
        vpt_change = df['Volume'] * (df['Close'] - prev_close) / prev_close.replace(0, np.nan)
        vpt_change = vpt_change.fillna(0)
        df['VPT'] = vpt_change.cumsum()
        df['VPT_MA1'] = df['VPT'].ewm(span=self.period1, adjust=False).mean()
        df['VPT_MA2'] = df['VPT_MA1'].ewm(span=self.period2, adjust=False).mean()
        df['VPT_OSC'] = (df['VPT_MA1'] - df['VPT_MA2']) * 100

        df['vpt_rising'] = df['VPT'] > df['VPT'].shift(1)
        df['vpt_falling'] = df['VPT'] < df['VPT'].shift(1)
        df['vpt_ma1_above_ma2'] = df['VPT_MA1'] > df['VPT_MA2']
        df['vpt_ma1_below_ma2'] = df['VPT_MA1'] < df['VPT_MA2']
        df['vpt_osc_positive'] = df['VPT_OSC'] > 0
        df['vpt_osc_negative'] = df['VPT_OSC'] < 0

        return df[['VPT', 'VPT_MA1', 'VPT_MA2', 'VPT_OSC',
                   'vpt_rising', 'vpt_falling',
                   'vpt_ma1_above_ma2', 'vpt_ma1_below_ma2',
                   'vpt_osc_positive', 'vpt_osc_negative']]
