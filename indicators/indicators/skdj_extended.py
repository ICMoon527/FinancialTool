import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SKDJExtended(BaseIndicator):
    """
    SKDJExtended（慢速随机指标扩展），中文称为慢速随机指标扩展。

    SKDJExtended指标是SKDJ指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    SKDJExtended指标结合了最高价、最低价和收盘价，来反应价格的超买超卖。

    计算公式：
    1. 计算RSV：RSV = (Close - LLV(Low, N)) / (HHV(High, N) - LLV(Low, N)) × 100
    2. 计算FASTK：FASTK = EMA(RSV, M1)
    3. 计算FASTD：FASTD = EMA(FASTK, M2)
    4. 计算K：K = EMA(FASTD, M3)
    5. 计算D：D = EMA(K, M4)
    6. 计算J：J = 3 × K - 2 × D
    7. 计算SKDJ_OSC：SKDJ_OSC = (K - D) × 100

    使用场景：
    - K > 80 时，表示超买，为卖出信号
    - K < 20 时，表示超卖，为买入信号
    - D > 80 时，表示超买，为卖出信号
    - D < 20 时，表示超卖，为买入信号
    - J > 100 时，表示超买，为卖出信号
    - J < 0 时，表示超卖，为买入信号
    - K上穿D时，为买入信号（金叉）
    - K下穿D时，为卖出信号（死叉）
    - SKDJ_OSC > 0 时，表示量价配合良好
    - SKDJ_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认9
    - ma_period1: 第一移动平均周期，默认3
    - ma_period2: 第二移动平均周期，默认3
    - ma_period3: 第三移动平均周期，默认3
    - ma_period4: 第四移动平均周期，默认3

    输出参数：
    - RSV: 未成熟随机值
    - FASTK: 快速K值
    - FASTD: 快速D值
    - K: K值
    - D: D值
    - J: J值
    - SKDJ_OSC: SKDJ的震荡指标
    - k_overbought: K > 80超买信号
    - k_oversold: K < 20超卖信号
    - d_overbought: D > 80超买信号
    - d_oversold: D < 20超卖信号
    - j_overbought: J > 100超买信号
    - j_oversold: J < 0超卖信号
    - k_above_d: K在D上方信号
    - k_below_d: K在D下方信号
    - k_rising: K上升信号
    - k_falling: K下降信号
    - d_rising: D上升信号
    - d_falling: D下降信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - SKDJExtended是超买超卖指标，在震荡市中表现最好
    - SKDJExtended在单边市中会产生频繁的假信号
    - SKDJExtended反应较慢，适合中长线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：SKDJ和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的SKDJExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=9, ma_period1=3, ma_period2=3, ma_period3=3, ma_period4=3):
        """
        初始化SKDJExtended指标

        Parameters
        ----------
        period : int, default 9
            计算周期
        ma_period1 : int, default 3
            第一移动平均周期
        ma_period2 : int, default 3
            第二移动平均周期
        ma_period3 : int, default 3
            第三移动平均周期
        ma_period4 : int, default 3
            第四移动平均周期
        """
        self.period = period
        self.ma_period1 = ma_period1
        self.ma_period2 = ma_period2
        self.ma_period3 = ma_period3
        self.ma_period4 = ma_period4

    def calculate(self, data):
        """
        计算SKDJExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SKDJExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        hhv = df['High'].rolling(window=self.period).max()
        llv = df['Low'].rolling(window=self.period).min()
        df['RSV'] = ((df['Close'] - llv) / (hhv - llv).replace(0, np.nan)) * 100
        df['RSV'] = df['RSV'].fillna(50)
        df['FASTK'] = df['RSV'].ewm(span=self.ma_period1, adjust=False).mean()
        df['FASTD'] = df['FASTK'].ewm(span=self.ma_period2, adjust=False).mean()
        df['K'] = df['FASTD'].ewm(span=self.ma_period3, adjust=False).mean()
        df['D'] = df['K'].ewm(span=self.ma_period4, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        df['SKDJ_OSC'] = (df['K'] - df['D']) * 100

        df['k_overbought'] = df['K'] > 80
        df['k_oversold'] = df['K'] < 20
        df['d_overbought'] = df['D'] > 80
        df['d_oversold'] = df['D'] < 20
        df['j_overbought'] = df['J'] > 100
        df['j_oversold'] = df['J'] < 0
        df['k_above_d'] = df['K'] > df['D']
        df['k_below_d'] = df['K'] < df['D']
        df['k_rising'] = df['K'] > df['K'].shift(1)
        df['k_falling'] = df['K'] < df['K'].shift(1)
        df['d_rising'] = df['D'] > df['D'].shift(1)
        df['d_falling'] = df['D'] < df['D'].shift(1)
        df['golden_cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        df['death_cross'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))

        return df[['RSV', 'FASTK', 'FASTD', 'K', 'D', 'J', 'SKDJ_OSC',
                   'k_overbought', 'k_oversold',
                   'd_overbought', 'd_oversold',
                   'j_overbought', 'j_oversold',
                   'k_above_d', 'k_below_d',
                   'k_rising', 'k_falling',
                   'd_rising', 'd_falling',
                   'golden_cross', 'death_cross']]
