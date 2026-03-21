import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SKDJ(BaseIndicator):
    """
    SKDJ（慢速随机指标），中文称为慢速随机指标。

    SKDJ指标是KD指标的变种，使用SMA而不是EMA，反应更慢但更稳定。
    SKDJ指标通过计算收盘价在一段时间内的位置，来判断价格的超买超卖状态。

    计算公式：
    1. 计算RSV（未成熟随机值）：RSV = (Close - Lowest_Low(N)) / (Highest_High(N) - Lowest_Low(N)) × 100
    2. 计算K：K = SMA(RSV, M)
    3. 计算D：D = SMA(K, P)

    使用场景：
    - K > 80 时，为超买信号，可能回落
    - K < 20 时，为超卖信号，可能反弹
    - K > D 时，为多头信号
    - K < D 时，为空头信号
    - K上穿D时，为买入信号（金叉）
    - K下穿D时，为卖出信号（死叉）
    - K与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认9
    - k_period: K线周期，默认3
    - d_period: D线周期，默认3

    输出参数：
    - RSV: 未成熟随机值
    - K: K线
    - D: D线
    - J: J线（3K - 2D）
    - k_overbought: K > 80信号
    - k_oversold: K < 20信号
    - k_above_d: K在D上方信号
    - k_below_d: K在D下方信号
    - k_cross_d_up: K上穿D信号
    - k_cross_d_down: K下穿D信号
    - k_rising: K上升信号
    - k_falling: K下降信号

    注意事项：
    - SKDJ是震荡指标，在单边市中会产生频繁的假信号
    - SKDJ在震荡市中表现最好
    - SKDJ反应较慢，不适合短线操作
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、BOLL）一起使用

    最佳实践建议：
    1. 趋势确认：K和D同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的SKDJ
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=9, k_period=3, d_period=3):
        """
        初始化SKDJ指标

        Parameters
        ----------
        period : int, default 9
            计算周期
        k_period : int, default 3
            K线周期
        d_period : int, default 3
            D线周期
        """
        self.period = period
        self.k_period = k_period
        self.d_period = d_period

    def calculate(self, data):
        """
        计算SKDJ指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SKDJ计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        lowest_low = df['Low'].rolling(window=self.period).min()
        highest_high = df['High'].rolling(window=self.period).max()
        df['RSV'] = (df['Close'] - lowest_low) / (highest_high - lowest_low) * 100
        df['K'] = df['RSV'].rolling(window=self.k_period).mean()
        df['D'] = df['K'].rolling(window=self.d_period).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        df['k_overbought'] = df['K'] > 80
        df['k_oversold'] = df['K'] < 20
        df['k_above_d'] = df['K'] > df['D']
        df['k_below_d'] = df['K'] < df['D']
        df['k_cross_d_up'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        df['k_cross_d_down'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
        df['k_rising'] = df['K'] > df['K'].shift(1)
        df['k_falling'] = df['K'] < df['K'].shift(1)

        return df[['RSV', 'K', 'D', 'J',
                   'k_overbought', 'k_oversold',
                   'k_above_d', 'k_below_d',
                   'k_cross_d_up', 'k_cross_d_down',
                   'k_rising', 'k_falling']]
