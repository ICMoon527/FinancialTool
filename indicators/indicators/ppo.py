import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PPO(BaseIndicator):
    """
    PPO（Price Percentage Oscillator）指标，中文称为价格摆动指标。

    PPO指标是一种基于移动平均线的震荡指标，通过计算两条移动平均线的百分比差值，
    来判断价格的动量变化和趋势变化。

    计算公式：
    1. 计算快速EMA：EMA_Fast = EMA(Close, Fast_Period)
    2. 计算慢速EMA：EMA_Slow = EMA(Close, Slow_Period)
    3. 计算PPO：PPO = (EMA_Fast - EMA_Slow) / EMA_Slow × 100
    4. 计算信号线：Signal = EMA(PPO, Signal_Period)
    5. 计算柱状图：Histogram = PPO - Signal

    使用场景：
    - PPO > 0 时，表示上升趋势，为多头市场
    - PPO < 0 时，表示下降趋势，为空头市场
    - PPO上穿信号线时，为买入信号
    - PPO下穿信号线时，为卖出信号
    - 柱状图由负转正时，为买入信号
    - 柱状图由正转负时，为卖出信号
    - PPO创新高时，表示动量强劲
    - PPO创新低时，表示动量疲软

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号线周期，默认9

    输出参数：
    - EMA_Fast: 快速EMA
    - EMA_Slow: 慢速EMA
    - PPO: 价格摆动指标
    - Signal: 信号线
    - Histogram: 柱状图
    - ppo_positive: PPO > 0信号
    - ppo_negative: PPO < 0信号
    - ppo_cross_signal_up: PPO上穿信号线信号
    - ppo_cross_signal_down: PPO下穿信号线信号
    - histogram_positive: 柱状图 > 0信号
    - histogram_negative: 柱状图 < 0信号
    - histogram_cross_zero_up: 柱状图上穿0线信号
    - histogram_cross_zero_down: 柱状图下穿0线信号
    - ppo_new_high: PPO创新高信号
    - ppo_new_low: PPO创新低信号

    注意事项：
    - PPO是震荡指标，在震荡市中会产生频繁的假信号
    - PPO在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - PPO反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：PPO > 0且柱状图 > 0更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的PPO
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化PPO指标

        Parameters
        ----------
        fast_period : int, default 12
            快速EMA周期
        slow_period : int, default 26
            慢速EMA周期
        signal_period : int, default 9
            信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算PPO指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PPO计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA_Fast'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()

        df['PPO'] = (df['EMA_Fast'] - df['EMA_Slow']) / df['EMA_Slow'].replace(0, np.nan) * 100
        df['Signal'] = df['PPO'].ewm(span=self.signal_period, adjust=False).mean()
        df['Histogram'] = df['PPO'] - df['Signal']

        df['ppo_positive'] = df['PPO'] > 0
        df['ppo_negative'] = df['PPO'] < 0
        df['ppo_cross_signal_up'] = (df['PPO'] > df['Signal']) & (df['PPO'].shift(1) <= df['Signal'].shift(1))
        df['ppo_cross_signal_down'] = (df['PPO'] < df['Signal']) & (df['PPO'].shift(1) >= df['Signal'].shift(1))
        df['histogram_positive'] = df['Histogram'] > 0
        df['histogram_negative'] = df['Histogram'] < 0
        df['histogram_cross_zero_up'] = (df['Histogram'] > 0) & (df['Histogram'].shift(1) <= 0)
        df['histogram_cross_zero_down'] = (df['Histogram'] < 0) & (df['Histogram'].shift(1) >= 0)
        df['ppo_new_high'] = df['PPO'] == df['PPO'].rolling(window=self.signal_period).max()
        df['ppo_new_low'] = df['PPO'] == df['PPO'].rolling(window=self.signal_period).min()

        return df[['EMA_Fast', 'EMA_Slow', 'PPO', 'Signal', 'Histogram',
                   'ppo_positive', 'ppo_negative', 'ppo_cross_signal_up', 'ppo_cross_signal_down',
                   'histogram_positive', 'histogram_negative', 'histogram_cross_zero_up', 'histogram_cross_zero_down',
                   'ppo_new_high', 'ppo_new_low']]
