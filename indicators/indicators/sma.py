import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SMA(BaseIndicator):
    """
    SMA（简单移动平均线），中文称为简单移动平均线。

    SMA指标是一种算术平均移动平均线，对所有价格赋予相同的权重。
    SMA指标通过简单平均，来反应价格的平均变化。

    计算公式：
    SMA = sum(Close, N) / N

    使用场景：
    - 价格在SMA上方时，为多头市场
    - 价格在SMA下方时，为空头市场
    - SMA上升时，表示上升趋势
    - SMA下降时，表示下降趋势
    - 短期SMA上穿长期SMA时，为买入信号（金叉）
    - 短期SMA下穿长期SMA时，为卖出信号（死叉）

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认10

    输出参数：
    - SMA: 简单移动平均线
    - price_above_sma: 价格在SMA上方信号
    - price_below_sma: 价格在SMA下方信号
    - sma_rising: SMA上升信号
    - sma_falling: SMA下降信号

    注意事项：
    - SMA是趋势指标，在震荡市中会产生频繁的假信号
    - SMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和SMA同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的SMA
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化SMA指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算SMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['SMA'] = df['Close'].rolling(window=self.period).mean()

        df['price_above_sma'] = df['Close'] > df['SMA']
        df['price_below_sma'] = df['Close'] < df['SMA']
        df['sma_rising'] = df['SMA'] > df['SMA'].shift(1)
        df['sma_falling'] = df['SMA'] < df['SMA'].shift(1)

        return df[['SMA',
                   'price_above_sma', 'price_below_sma',
                   'sma_rising', 'sma_falling']]
