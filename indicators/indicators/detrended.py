import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Detrended(BaseIndicator):
    """
    Detrended Price Oscillator（去趋势价格震荡指标），中文称为去趋势价格震荡指标。

    去趋势价格震荡指标（DPO）是一种震荡指标，通过计算价格与移动平均线的差值，
    来判断价格的超买超卖状态和趋势变化。

    计算公式：
    1. 计算移动平均线：MA = SMA(Close, N)
    2. 计算DPO：DPO = Close_prev_(N/2+1) - MA

    使用场景：
    - DPO > 0 时，表示价格在趋势之上，为多头市场
    - DPO < 0 时，表示价格在趋势之下，为空头市场
    - DPO由负转正时，为买入信号
    - DPO由正转负时，为卖出信号
    - DPO创新高时，表示价格强势
    - DPO创新低时，表示价格弱势
    - DPO上升时，表示动量增强
    - DPO下降时，表示动量减弱

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20

    输出参数：
    - MA: 移动平均线
    - DPO: 去趋势价格震荡指标
    - DPO_MA: DPO的移动平均线
    - dpo_positive: DPO > 0信号
    - dpo_negative: DPO < 0信号
    - dpo_cross_zero_up: DPO上穿0线信号
    - dpo_cross_zero_down: DPO下穿0线信号
    - dpo_rising: DPO上升信号
    - dpo_falling: DPO下降信号
    - dpo_new_high: DPO创新高信号
    - dpo_new_low: DPO创新低信号
    - dpo_above_ma: DPO在MA上方信号
    - dpo_below_ma: DPO在MA下方信号

    注意事项：
    - DPO是震荡指标，在震荡市中表现较好
    - DPO在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 超买超卖：DPO < 0 买入和DPO > 0 卖出更可靠
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

        df['MA'] = df['Close'].rolling(window=self.period).mean()

        shift_period = int(self.period / 2) + 1
        close_shifted = df['Close'].shift(shift_period)

        df['DPO'] = close_shifted - df['MA']
        df['DPO_MA'] = df['DPO'].rolling(window=self.ma_period).mean()

        df['dpo_positive'] = df['DPO'] > 0
        df['dpo_negative'] = df['DPO'] < 0
        df['dpo_cross_zero_up'] = (df['DPO'] > 0) & (df['DPO'].shift(1) <= 0)
        df['dpo_cross_zero_down'] = (df['DPO'] < 0) & (df['DPO'].shift(1) >= 0)
        df['dpo_rising'] = df['DPO'] > df['DPO'].shift(1)
        df['dpo_falling'] = df['DPO'] < df['DPO'].shift(1)
        df['dpo_new_high'] = df['DPO'] == df['DPO'].rolling(window=self.ma_period).max()
        df['dpo_new_low'] = df['DPO'] == df['DPO'].rolling(window=self.ma_period).min()
        df['dpo_above_ma'] = df['DPO'] > df['DPO_MA']
        df['dpo_below_ma'] = df['DPO'] < df['DPO_MA']

        return df[['MA', 'DPO', 'DPO_MA', 'dpo_positive', 'dpo_negative',
                   'dpo_cross_zero_up', 'dpo_cross_zero_down', 'dpo_rising', 'dpo_falling',
                   'dpo_new_high', 'dpo_new_low', 'dpo_above_ma', 'dpo_below_ma']]
