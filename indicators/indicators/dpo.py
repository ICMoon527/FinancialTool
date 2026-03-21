import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DPO(BaseIndicator):
    """
    DPO（Detrended Price Oscillator），中文称为去趋势价格震荡器。

    DPO指标是通过消除价格趋势来识别价格周期的震荡指标。
    DPO指标通过计算当前价格与过去移动平均线的差值，来判断价格的周期变化。

    计算公式：
    1. 计算SMA：SMA = SMA(Close, N)
    2. 计算Shifted_SMA：Shifted_SMA = SMA.shift(N/2 + 1)
    3. 计算DPO：DPO = Close - Shifted_SMA

    使用场景：
    - DPO > 0 时，表示价格高于过去的平均水平，为多头信号
    - DPO < 0 时，表示价格低于过去的平均水平，为空头信号
    - DPO上升时，表示价格上升，为买入信号
    - DPO下降时，表示价格下降，为卖出信号
    - DPO由负转正时，为买入信号
    - DPO由正转负时，为卖出信号
    - DPO与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20

    输出参数：
    - SMA: 简单移动平均线
    - Shifted_SMA: 移动的移动平均线
    - DPO: 去趋势价格震荡器
    - DPO_MA: DPO的移动平均线
    - dpo_positive: DPO > 0信号
    - dpo_negative: DPO < 0信号
    - dpo_rising: DPO上升信号
    - dpo_falling: DPO下降信号
    - dpo_cross_zero_up: DPO上穿0线信号
    - dpo_cross_zero_down: DPO下穿0线信号
    - dpo_above_ma: DPO在MA上方信号
    - dpo_below_ma: DPO在MA下方信号

    注意事项：
    - DPO是去趋势震荡指标，在震荡市中表现最好
    - DPO在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 周期判断：DPO的周期与市场周期同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DPO
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, ma_period=10):
        """
        初始化DPO指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算DPO指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DPO计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        shift_period = int(self.period / 2 + 1)
        df['SMA'] = df['Close'].rolling(window=self.period).mean()
        df['Shifted_SMA'] = df['SMA'].shift(shift_period)
        df['DPO'] = df['Close'] - df['Shifted_SMA']
        df['DPO_MA'] = df['DPO'].rolling(window=self.ma_period).mean()

        df['dpo_positive'] = df['DPO'] > 0
        df['dpo_negative'] = df['DPO'] < 0
        df['dpo_rising'] = df['DPO'] > df['DPO'].shift(1)
        df['dpo_falling'] = df['DPO'] < df['DPO'].shift(1)
        df['dpo_cross_zero_up'] = (df['DPO'] > 0) & (df['DPO'].shift(1) <= 0)
        df['dpo_cross_zero_down'] = (df['DPO'] < 0) & (df['DPO'].shift(1) >= 0)
        df['dpo_above_ma'] = df['DPO'] > df['DPO_MA']
        df['dpo_below_ma'] = df['DPO'] < df['DPO_MA']

        return df[['SMA', 'Shifted_SMA', 'DPO', 'DPO_MA',
                   'dpo_positive', 'dpo_negative',
                   'dpo_rising', 'dpo_falling',
                   'dpo_cross_zero_up', 'dpo_cross_zero_down',
                   'dpo_above_ma', 'dpo_below_ma']]
