import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ZX(BaseIndicator):
    """
    ZX（重心线），中文称为重心线。

    ZX指标是一种移动平均指标，通过计算价格的重心，来判断价格的趋势。
    ZX指标结合了最高价、最低价和收盘价，来反应价格的平均水平。

    计算公式：
    1. 计算TypicalPrice：TypicalPrice = (High + Low + Close) / 3
    2. 计算ZX：ZX = EMA(TypicalPrice, N)

    使用场景：
    - 价格在ZX上方时，为多头市场
    - 价格在ZX下方时，为空头市场
    - ZX上升时，表示上升趋势
    - ZX下降时，表示下降趋势
    - 价格上穿ZX时，为买入信号
    - 价格下穿ZX时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认10

    输出参数：
    - TypicalPrice: 典型价格
    - ZX: 重心线
    - price_above_zx: 价格在ZX上方信号
    - price_below_zx: 价格在ZX下方信号
    - zx_rising: ZX上升信号
    - zx_falling: ZX下降信号
    - price_cross_zx_up: 价格上穿ZX信号
    - price_cross_zx_down: 价格下穿ZX信号

    注意事项：
    - ZX是移动平均指标，在震荡市中会产生频繁的假信号
    - ZX在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和ZX同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ZX
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化ZX指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算ZX指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ZX计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['ZX'] = df['TypicalPrice'].ewm(span=self.period, adjust=False).mean()

        df['price_above_zx'] = df['Close'] > df['ZX']
        df['price_below_zx'] = df['Close'] < df['ZX']
        df['zx_rising'] = df['ZX'] > df['ZX'].shift(1)
        df['zx_falling'] = df['ZX'] < df['ZX'].shift(1)
        df['price_cross_zx_up'] = (df['Close'] > df['ZX']) & (df['Close'].shift(1) <= df['ZX'].shift(1))
        df['price_cross_zx_down'] = (df['Close'] < df['ZX']) & (df['Close'].shift(1) >= df['ZX'].shift(1))

        return df[['TypicalPrice', 'ZX',
                   'price_above_zx', 'price_below_zx',
                   'zx_rising', 'zx_falling',
                   'price_cross_zx_up', 'price_cross_zx_down']]
