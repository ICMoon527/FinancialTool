import pandas as pd
import numpy as np
from ..base import BaseIndicator


class RAVI(BaseIndicator):
    """
    RAVI（Range Action Verification Index），中文称为区间动作验证指标。

    RAVI指标是由Markos Katsanos提出的，是一种趋势指标。
    RAVI指标通过计算短期和长期移动平均线的差值百分比，来判断价格的趋势强度。

    计算公式：
    1. 计算SMA_Short：SMA_Short = SMA(Close, S)
    2. 计算SMA_Long：SMA_Long = SMA(Close, L)
    3. 计算RAVI：RAVI = |(SMA_Short - SMA_Long) / SMA_Long| × 100

    使用场景：
    - RAVI上升时，表示趋势强度增加
    - RAVI下降时，表示趋势强度减少
    - RAVI > 阈值时，表示趋势较强
    - RAVI < 阈值时，表示趋势较弱
    - SMA_Short > SMA_Long时，表示上升趋势
    - SMA_Short < SMA_Long时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - short_period: 短期周期，默认7
    - long_period: 长期周期，默认65
    - threshold: 趋势阈值，默认0.5

    输出参数：
    - SMA_Short: 短期移动平均线
    - SMA_Long: 长期移动平均线
    - RAVI: 区间动作验证指标
    - RAVI_MA: RAVI的移动平均线
    - ravi_rising: RAVI上升信号
    - ravi_falling: RAVI下降信号
    - ravi_above_threshold: RAVI > 阈值信号
    - ravi_below_threshold: RAVI < 阈值信号
    - short_above_long: 短期在长期上方信号
    - short_below_long: 短期在长期下方信号
    - short_cross_long_up: 短期上穿长期信号
    - short_cross_long_down: 短期下穿长期信号

    注意事项：
    - RAVI是趋势强度指标，在震荡市中会产生频繁的假信号
    - RAVI在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：RAVI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的RAVI
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, short_period=7, long_period=65, threshold=0.5, ma_period=10):
        """
        初始化RAVI指标

        Parameters
        ----------
        short_period : int, default 7
            短期周期
        long_period : int, default 65
            长期周期
        threshold : float, default 0.5
            趋势阈值
        ma_period : int, default 10
            移动平均周期
        """
        self.short_period = short_period
        self.long_period = long_period
        self.threshold = threshold
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算RAVI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含RAVI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['SMA_Short'] = df['Close'].rolling(window=self.short_period).mean()
        df['SMA_Long'] = df['Close'].rolling(window=self.long_period).mean()
        df['RAVI'] = abs((df['SMA_Short'] - df['SMA_Long']) / df['SMA_Long']) * 100
        df['RAVI_MA'] = df['RAVI'].rolling(window=self.ma_period).mean()

        df['ravi_rising'] = df['RAVI'] > df['RAVI'].shift(1)
        df['ravi_falling'] = df['RAVI'] < df['RAVI'].shift(1)
        df['ravi_above_threshold'] = df['RAVI'] > self.threshold
        df['ravi_below_threshold'] = df['RAVI'] < self.threshold
        df['short_above_long'] = df['SMA_Short'] > df['SMA_Long']
        df['short_below_long'] = df['SMA_Short'] < df['SMA_Long']
        df['short_cross_long_up'] = (df['SMA_Short'] > df['SMA_Long']) & (df['SMA_Short'].shift(1) <= df['SMA_Long'].shift(1))
        df['short_cross_long_down'] = (df['SMA_Short'] < df['SMA_Long']) & (df['SMA_Short'].shift(1) >= df['SMA_Long'].shift(1))

        return df[['SMA_Short', 'SMA_Long', 'RAVI', 'RAVI_MA',
                   'ravi_rising', 'ravi_falling',
                   'ravi_above_threshold', 'ravi_below_threshold',
                   'short_above_long', 'short_below_long',
                   'short_cross_long_up', 'short_cross_long_down']]
