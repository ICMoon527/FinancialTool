import pandas as pd
import numpy as np
from ..base import BaseIndicator


class KDJCross(BaseIndicator):
    """
    KDJ金叉死叉指标，中文称为KDJ金叉死叉指标。

    KDJ金叉死叉指标是基于KDJ的技术指标，通过计算K线和D线的交叉，
    来判断价格趋势的变化。金叉表示K线上穿D线，为买入信号；死叉表示K线下穿D线，为卖出信号。

    计算公式：
    1. 计算RSV（未成熟随机值）：RSV = (Close - LLV(Low, N)) / (HHV(High, N) - LLV(Low, N)) × 100
    2. 计算K线：K = SMA(RSV, M1, 1)
    3. 计算D线：D = SMA(K, M2, 1)
    4. 计算J线：J = 3 × K - 2 × D
    5. 判断金叉：K上穿D
    6. 判断死叉：K下穿D

    使用场景：
    - 金叉出现时，为买入信号
    - 死叉出现时，为卖出信号
    - K在D上方时，为多头市场
    - K在D下方时，为空头市场
    - KDJ在超卖区（K<20）出现金叉时，买入信号更可靠
    - KDJ在超买区（K>80）出现死叉时，卖出信号更可靠
    - J线突破100时，为超买信号
    - J线跌破0时，为超卖信号

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - n_period: RSV计算周期，默认9
    - m1_period: K线计算周期，默认3
    - m2_period: D线计算周期，默认3

    输出参数：
    - K: K线
    - D: D线
    - J: J线
    - golden_cross: 金叉信号
    - death_cross: 死叉信号
    - k_above_d: K在D上方信号
    - k_below_d: K在D下方信号
    - overbought: 超买信号（K>80）
    - oversold: 超卖信号（K<20）
    - j_overbought: J超买信号（J>100）
    - j_oversold: J超卖信号（J<0）

    注意事项：
    - KDJ金叉死叉是震荡指标，在震荡市中表现最好
    - KDJ金叉死叉在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、MA）一起使用
    - KDJ金叉死叉反应快速，适合短线操作

    最佳实践建议：
    1. 超买超卖：在超卖区金叉和超买区死叉更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的KDJ金叉死叉
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：金叉死叉后等待1-2个交易日确认
    """

    def __init__(self, n_period=9, m1_period=3, m2_period=3):
        """
        初始化KDJ金叉死叉指标

        Parameters
        ----------
        n_period : int, default 9
            RSV计算周期
        m1_period : int, default 3
            K线计算周期
        m2_period : int, default 3
            D线计算周期
        """
        self.n_period = n_period
        self.m1_period = m1_period
        self.m2_period = m2_period

    def calculate(self, data):
        """
        计算KDJ金叉死叉指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含KDJ金叉死叉计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        low_min = df['Low'].rolling(window=self.n_period).min()
        high_max = df['High'].rolling(window=self.n_period).max()

        rsv = (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100

        df['K'] = rsv.ewm(com=self.m1_period - 1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=self.m2_period - 1, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        df['golden_cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        df['death_cross'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
        df['k_above_d'] = df['K'] > df['D']
        df['k_below_d'] = df['K'] < df['D']
        df['overbought'] = df['K'] > 80
        df['oversold'] = df['K'] < 20
        df['j_overbought'] = df['J'] > 100
        df['j_oversold'] = df['J'] < 0

        return df[['K', 'D', 'J', 'golden_cross', 'death_cross', 'k_above_d', 
                   'k_below_d', 'overbought', 'oversold', 'j_overbought', 'j_oversold']]
