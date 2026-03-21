import pandas as pd
import numpy as np
from ..base import BaseIndicator


class StochasticRSI(BaseIndicator):
    """
    Stochastic RSI（随机相对强弱指标），中文称为随机RSI指标。

    Stochastic RSI指标是结合了RSI和随机指标的特点，用于判断价格的超买超卖状态。
    Stochastic RSI指标通过计算RSI在一段时间内的位置，来判断价格的超买超卖状态。

    计算公式：
    1. 计算RSI：RSI = RSI(Close, N)
    2. 计算Stochastic RSI：stoch_rsi = (RSI - RSI_min) / (RSI_max - RSI_min)
    3. 计算K：K = SMA(stoch_rsi, M)
    4. 计算D：D = SMA(K, P)

    使用场景：
    - Stochastic RSI > 0.8 时，为超买信号，可能回落
    - Stochastic RSI < 0.2 时，为超卖信号，可能反弹
    - K > D 时，为多头信号
    - K < D 时，为空头信号
    - K上穿D时，为买入信号（金叉）
    - K下穿D时，为卖出信号（死叉）
    - Stochastic RSI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - rsi_period: RSI周期，默认14
    - stoch_period: 随机周期，默认14
    - k_period: K线周期，默认3
    - d_period: D线周期，默认3

    输出参数：
    - RSI: 相对强弱指标
    - Stoch_RSI: 随机RSI
    - K: K线
    - D: D线
    - stoch_rsi_overbought: Stochastic RSI > 0.8信号
    - stoch_rsi_oversold: Stochastic RSI < 0.2信号
    - k_above_d: K在D上方信号
    - k_below_d: K在D下方信号
    - k_cross_d_up: K上穿D信号
    - k_cross_d_down: K下穿D信号
    - stoch_rsi_rising: Stochastic RSI上升信号
    - stoch_rsi_falling: Stochastic RSI下降信号

    注意事项：
    - Stochastic RSI是震荡指标，在单边市中会产生频繁的假信号
    - Stochastic RSI在震荡市中表现最好
    - Stochastic RSI反应较快，适合中短线操作
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、BOLL）一起使用

    最佳实践建议：
    1. 趋势确认：K和D同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Stochastic RSI
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """
        初始化Stochastic RSI指标

        Parameters
        ----------
        rsi_period : int, default 14
            RSI周期
        stoch_period : int, default 14
            随机周期
        k_period : int, default 3
            K线周期
        d_period : int, default 3
            D线周期
        """
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period

    def calculate(self, data):
        """
        计算Stochastic RSI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Stochastic RSI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        rsi_min = df['RSI'].rolling(window=self.stoch_period).min()
        rsi_max = df['RSI'].rolling(window=self.stoch_period).max()
        df['Stoch_RSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min)
        df['K'] = df['Stoch_RSI'].rolling(window=self.k_period).mean()
        df['D'] = df['K'].rolling(window=self.d_period).mean()

        df['stoch_rsi_overbought'] = df['Stoch_RSI'] > 0.8
        df['stoch_rsi_oversold'] = df['Stoch_RSI'] < 0.2
        df['k_above_d'] = df['K'] > df['D']
        df['k_below_d'] = df['K'] < df['D']
        df['k_cross_d_up'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        df['k_cross_d_down'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
        df['stoch_rsi_rising'] = df['Stoch_RSI'] > df['Stoch_RSI'].shift(1)
        df['stoch_rsi_falling'] = df['Stoch_RSI'] < df['Stoch_RSI'].shift(1)

        return df[['RSI', 'Stoch_RSI', 'K', 'D',
                   'stoch_rsi_overbought', 'stoch_rsi_oversold',
                   'k_above_d', 'k_below_d',
                   'k_cross_d_up', 'k_cross_d_down',
                   'stoch_rsi_rising', 'stoch_rsi_falling']]
