import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EMA(BaseIndicator):
    """
    EMA（指数移动平均线），中文称为指数移动平均线。

    EMA指标是一种加权移动平均线，对近期价格赋予更高的权重。
    EMA指标通过指数加权，来反应最新的价格变化。

    计算公式：
    EMA_today = (Price_today × K) + (EMA_yesterday × (1 - K))
    其中 K = 2 / (N + 1)

    使用场景：
    - 价格在EMA上方时，为多头市场
    - 价格在EMA下方时，为空头市场
    - EMA上升时，表示上升趋势
    - EMA下降时，表示下降趋势
    - 短期EMA上穿长期EMA时，为买入信号（金叉）
    - 短期EMA下穿长期EMA时，为卖出信号（死叉）

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认12

    输出参数：
    - EMA: 指数移动平均线
    - price_above_ema: 价格在EMA上方信号
    - price_below_ema: 价格在EMA下方信号
    - ema_rising: EMA上升信号
    - ema_falling: EMA下降信号

    注意事项：
    - EMA是趋势指标，在震荡市中会产生频繁的假信号
    - EMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和EMA同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的EMA
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=12):
        """
        初始化EMA指标

        Parameters
        ----------
        period : int, default 12
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算EMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含EMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA'] = df['Close'].ewm(span=self.period, adjust=False).mean()

        df['price_above_ema'] = df['Close'] > df['EMA']
        df['price_below_ema'] = df['Close'] < df['EMA']
        df['ema_rising'] = df['EMA'] > df['EMA'].shift(1)
        df['ema_falling'] = df['EMA'] < df['EMA'].shift(1)

        return df[['EMA',
                   'price_above_ema', 'price_below_ema',
                   'ema_rising', 'ema_falling']]
