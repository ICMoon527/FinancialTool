import pandas as pd
import numpy as np
from ..base import BaseIndicator


class KAMA(BaseIndicator):
    """
    KAMA（Kaufman Adaptive Moving Average）指标，中文称为考夫曼自适应移动平均指标。

    KAMA指标是由佩里·考夫曼（Perry Kaufman）提出的，是一种自适应移动平均指标。
    KAMA指标根据市场波动性自动调整平滑系数，在波动小时快速响应，在波动大时慢速响应。

    计算公式：
    1. 计算价格变动：Change = |Close - Close_prev_N|
    2. 计算价格波动：Volatility = Σ(|Close - Close_prev|, N)
    3. 计算效率比：ER = Change / Volatility
    4. 计算平滑常数：SC = (ER × (Fast_Smoothing - Slow_Smoothing) + Slow_Smoothing)²
    5. 计算KAMA：KAMA = KAMA_prev + SC × (Close - KAMA_prev)

    使用场景：
    - KAMA上升时，表示上升趋势，为买入信号
    - KAMA下降时，表示下降趋势，为卖出信号
    - 价格上穿KAMA时，为买入信号
    - 价格下穿KAMA时，为卖出信号
    - KAMA在价格上方时，为阻力位
    - KAMA在价格下方时，为支撑位
    - KAMA平滑系数大时，表示市场波动小
    - KAMA平滑系数小时，表示市场波动大

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认10
    - fast_smoothing: 快速平滑常数，默认2
    - slow_smoothing: 慢速平滑常数，默认30

    输出参数：
    - Change: 价格变动
    - Volatility: 价格波动
    - ER: 效率比
    - SC: 平滑常数
    - KAMA: 考夫曼自适应移动平均
    - close_above_kama: 价格在KAMA上方信号
    - close_below_kama: 价格在KAMA下方信号
    - kama_rising: KAMA上升信号
    - kama_falling: KAMA下降信号
    - kama_cross_up: 价格上穿KAMA信号
    - kama_cross_down: 价格下穿KAMA信号
    - high_er: ER > 0.5信号（市场波动小）
    - low_er: ER < 0.2信号（市场波动大）

    注意事项：
    - KAMA是趋势指标，在震荡市中会产生频繁的假信号
    - KAMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - KAMA反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：KAMA和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的KAMA
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, fast_smoothing=2, slow_smoothing=30):
        """
        初始化KAMA指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        fast_smoothing : int, default 2
            快速平滑常数
        slow_smoothing : int, default 30
            慢速平滑常数
        """
        self.period = period
        self.fast_smoothing = fast_smoothing
        self.slow_smoothing = slow_smoothing

    def calculate(self, data):
        """
        计算KAMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含KAMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        close_prev_n = df['Close'].shift(self.period)

        df['Change'] = abs(df['Close'] - close_prev_n)
        df['Volatility'] = abs(df['Close'] - close_prev).rolling(window=self.period).sum()
        df['ER'] = df['Change'] / df['Volatility'].replace(0, np.nan)

        fast_sc = 2 / (self.fast_smoothing + 1)
        slow_sc = 2 / (self.slow_smoothing + 1)
        df['SC'] = (df['ER'] * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = [np.nan] * len(df)
        for i in range(self.period, len(df)):
            if i == self.period:
                kama[i] = df['Close'].iloc[i]
            else:
                kama[i] = kama[i - 1] + df['SC'].iloc[i] * (df['Close'].iloc[i] - kama[i - 1])

        df['KAMA'] = kama

        df['close_above_kama'] = df['Close'] > df['KAMA']
        df['close_below_kama'] = df['Close'] < df['KAMA']
        df['kama_rising'] = df['KAMA'] > df['KAMA'].shift(1)
        df['kama_falling'] = df['KAMA'] < df['KAMA'].shift(1)
        df['kama_cross_up'] = (df['Close'] > df['KAMA']) & (df['Close'].shift(1) <= df['KAMA'].shift(1))
        df['kama_cross_down'] = (df['Close'] < df['KAMA']) & (df['Close'].shift(1) >= df['KAMA'].shift(1))
        df['high_er'] = df['ER'] > 0.5
        df['low_er'] = df['ER'] < 0.2

        return df[['Change', 'Volatility', 'ER', 'SC', 'KAMA',
                   'close_above_kama', 'close_below_kama', 'kama_rising', 'kama_falling',
                   'kama_cross_up', 'kama_cross_down', 'high_er', 'low_er']]
