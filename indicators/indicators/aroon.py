import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Aroon(BaseIndicator):
    """
    Aroon（阿隆指标），中文称为阿隆指标。

    Aroon指标是由图莎尔·钱德（Tushar Chande）提出的，是一种趋势指标。
    Aroon指标通过计算价格在N日内创最高价和最低价的时间，来判断价格的趋势变化。

    计算公式：
    1. 计算Aroon Up = (N - 距离最高价的天数) / N × 100
    2. 计算Aroon Down = (N - 距离最低价的天数) / N × 100
    3. 计算Aroon Oscillator = Aroon Up - Aroon Down

    使用场景：
    - Aroon Up > 70 且 Aroon Down < 30 时，表示上升趋势强劲
    - Aroon Down > 70 且 Aroon Up < 30 时，表示下降趋势强劲
    - Aroon Oscillator > 0 时，表示上升趋势
    - Aroon Oscillator < 0 时，表示下降趋势
    - Aroon Up由下往上穿越50时，为买入信号
    - Aroon Down由下往上穿越50时，为卖出信号
    - Aroon Up和Aroon Down都在50以下时，表示震荡市

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'列
    - period: 计算周期，默认25

    输出参数：
    - Aroon_Up: 阿隆上升线
    - Aroon_Down: 阿隆下降线
    - Aroon_Oscillator: 阿隆震荡指标
    - aroon_up_strong: Aroon Up > 70信号
    - aroon_down_strong: Aroon Down > 70信号
    - aroon_osc_positive: Aroon Oscillator > 0信号
    - aroon_osc_negative: Aroon Oscillator < 0信号
    - aroon_up_cross_50: Aroon Up上穿50信号
    - aroon_down_cross_50: Aroon Down上穿50信号
    - uptrend_signal: 上升趋势信号
    - downtrend_signal: 下降趋势信号

    注意事项：
    - Aroon是趋势指标，在震荡市中会产生频繁的假信号
    - Aroon在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - Aroon反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：Aroon Up > 70且Aroon Down < 30更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Aroon
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待2-3个交易日确认
    """

    def __init__(self, period=25):
        """
        初始化Aroon指标

        Parameters
        ----------
        period : int, default 25
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算Aroon指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Aroon计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        days_since_high = []
        days_since_low = []

        for i in range(len(df)):
            start_idx = max(0, i - self.period + 1)
            end_idx = i + 1
            window_high = df['High'].iloc[start_idx:end_idx]
            window_low = df['Low'].iloc[start_idx:end_idx]

            high_idx = window_high.idxmax()
            low_idx = window_low.idxmin()

            days_since_high.append(i - high_idx)
            days_since_low.append(i - low_idx)

        df['days_since_high'] = days_since_high
        df['days_since_low'] = days_since_low

        df['Aroon_Up'] = (self.period - df['days_since_high']) / self.period * 100
        df['Aroon_Down'] = (self.period - df['days_since_low']) / self.period * 100
        df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']

        df['aroon_up_strong'] = df['Aroon_Up'] > 70
        df['aroon_down_strong'] = df['Aroon_Down'] > 70
        df['aroon_osc_positive'] = df['Aroon_Oscillator'] > 0
        df['aroon_osc_negative'] = df['Aroon_Oscillator'] < 0
        df['aroon_up_cross_50'] = (df['Aroon_Up'] > 50) & (df['Aroon_Up'].shift(1) <= 50)
        df['aroon_down_cross_50'] = (df['Aroon_Down'] > 50) & (df['Aroon_Down'].shift(1) <= 50)
        df['uptrend_signal'] = (df['Aroon_Up'] > 70) & (df['Aroon_Down'] < 30)
        df['downtrend_signal'] = (df['Aroon_Down'] > 70) & (df['Aroon_Up'] < 30)

        return df[['Aroon_Up', 'Aroon_Down', 'Aroon_Oscillator',
                   'aroon_up_strong', 'aroon_down_strong',
                   'aroon_osc_positive', 'aroon_osc_negative',
                   'aroon_up_cross_50', 'aroon_down_cross_50',
                   'uptrend_signal', 'downtrend_signal']]
