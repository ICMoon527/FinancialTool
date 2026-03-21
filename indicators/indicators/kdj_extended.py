import pandas as pd
import numpy as np
from ..base import BaseIndicator


class KDJExtended(BaseIndicator):
    """
    KDJExtended（KDJ扩展），中文称为KDJ扩展。

    KDJExtended指标是KDJ指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    KDJExtended指标结合了未成熟随机值、K值、D值和J值，来反应价格的超买超卖状态。

    计算公式：
    1. 计算RSV：RSV = (Close - LLV(Low, N)) / (HHV(High, N) - LLV(Low, N)) × 100
    2. 计算K：K = SMA(RSV, M)
    3. 计算D：D = SMA(K, P)
    4. 计算J：J = 3 × K - 2 × D
    5. 计算KDJ_OSC：KDJ_OSC = K - D

    使用场景：
    - K上穿D时，为买入信号（金叉）
    - K下穿D时，为卖出信号（死叉）
    - KDJ > 80 时，表示超买，为卖出信号
    - KDJ < 20 时，表示超卖，为买入信号
    - J > 100 时，表示严重超买，为卖出信号
    - J < 0 时，表示严重超卖，为买入信号
    - KDJ上升时，表示上升趋势
    - KDJ下降时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - n: 计算周期，默认9
    - m: K值平滑周期，默认3
    - p: D值平滑周期，默认3

    输出参数：
    - RSV: 未成熟随机值
    - K: K值（快速线）
    - D: D值（慢速线）
    - J: J值（辅助线）
    - KDJ_OSC: KDJ震荡指标
    - kdj_overbought: KDJ超买信号（>80）
    - kdj_oversold: KDJ超卖信号（<20）
    - j_overbought: J值超买信号（>100）
    - j_oversold: J值超卖信号（<0）
    - k_above_d: K在D上方信号
    - k_below_d: K在D下方信号
    - kdj_rising: KDJ上升信号
    - kdj_falling: KDJ下降信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - KDJExtended是摆动指标，在震荡市中表现最好
    - KDJExtended在单边市中会产生钝化现象
    - KDJExtended反应较快，适合中短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和KDJ同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的KDJExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, n=9, m=3, p=3):
        """
        初始化KDJExtended指标

        Parameters
        ----------
        n : int, default 9
            计算周期
        m : int, default 3
            K值平滑周期
        p : int, default 3
            D值平滑周期
        """
        self.n = n
        self.m = m
        self.p = p

    def calculate(self, data):
        """
        计算KDJExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含KDJExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        low_min = df['Low'].rolling(window=self.n).min()
        high_max = df['High'].rolling(window=self.n).max()
        df['RSV'] = ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100
        df['RSV'] = df['RSV'].fillna(50)
        df['K'] = df['RSV'].rolling(window=self.m).mean()
        df['D'] = df['K'].rolling(window=self.p).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        df['KDJ_OSC'] = df['K'] - df['D']

        df['kdj_overbought'] = df['K'] > 80
        df['kdj_oversold'] = df['K'] < 20
        df['j_overbought'] = df['J'] > 100
        df['j_oversold'] = df['J'] < 0
        df['k_above_d'] = df['K'] > df['D']
        df['k_below_d'] = df['K'] < df['D']
        df['kdj_rising'] = df['K'] > df['K'].shift(1)
        df['kdj_falling'] = df['K'] < df['K'].shift(1)
        df['golden_cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        df['death_cross'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))

        return df[['RSV', 'K', 'D', 'J', 'KDJ_OSC',
                   'kdj_overbought', 'kdj_oversold',
                   'j_overbought', 'j_oversold',
                   'k_above_d', 'k_below_d',
                   'kdj_rising', 'kdj_falling',
                   'golden_cross', 'death_cross']]
