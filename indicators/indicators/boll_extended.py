import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BOLLExtended(BaseIndicator):
    """
    BOLLExtended（布林带扩展），中文称为布林带扩展。

    BOLLExtended指标是BOLL指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    BOLLExtended指标结合了中轨、上轨、下轨和带宽，来反应价格的波动范围。

    计算公式：
    1. 计算MID：MID = SMA(Close, N)
    2. 计算STD：STD = STD(Close, N)
    3. 计算UPPER：UPPER = MID + K × STD
    4. 计算LOWER：LOWER = MID - K × STD
    5. 计算BWIDTH：BWIDTH = (UPPER - LOWER) / MID × 100
    6. 计算%b：%b = (Close - LOWER) / (UPPER - LOWER)

    使用场景：
    - 价格突破上轨时，表示超买，为卖出信号
    - 价格跌破下轨时，表示超卖，为买入信号
    - 价格在中轨上方时，表示多头市场
    - 价格在中轨下方时，表示空头市场
    - 带宽扩大时，表示波动加剧
    - 带宽缩小时，表示波动收窄，可能出现突破
    - %b > 1 时，表示价格在上轨上方
    - %b < 0 时，表示价格在下轨下方

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20
    - std_dev: 标准差倍数，默认2

    输出参数：
    - MID: 中轨
    - STD: 标准差
    - UPPER: 上轨
    - LOWER: 下轨
    - BWIDTH: 带宽
    - PERCENT_B: %b指标
    - price_above_upper: 价格在上轨上方信号
    - price_below_lower: 价格在下轨下方信号
    - price_above_mid: 价格在中轨上方信号
    - price_below_mid: 价格在中轨下方信号
    - boll_rising: 中轨上升信号
    - boll_falling: 中轨下降信号
    - boll_widening: 带宽扩大信号
    - boll_narrowing: 带宽收窄信号

    注意事项：
    - BOLLExtended是通道指标，在震荡市中表现最好
    - BOLLExtended在单边市中会产生连续的突破信号
    - 带宽的变化比单纯的突破更重要
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和中轨同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的BOLLExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, std_dev=2):
        """
        初始化BOLLExtended指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        std_dev : float, default 2
            标准差倍数
        """
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data):
        """
        计算BOLLExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BOLLExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MID'] = df['Close'].rolling(window=self.period).mean()
        df['STD'] = df['Close'].rolling(window=self.period).std()
        df['UPPER'] = df['MID'] + self.std_dev * df['STD']
        df['LOWER'] = df['MID'] - self.std_dev * df['STD']
        df['BWIDTH'] = ((df['UPPER'] - df['LOWER']) / df['MID'].replace(0, np.nan)) * 100
        df['BWIDTH'] = df['BWIDTH'].fillna(0)
        df['PERCENT_B'] = ((df['Close'] - df['LOWER']) / (df['UPPER'] - df['LOWER']).replace(0, np.nan))
        df['PERCENT_B'] = df['PERCENT_B'].fillna(0.5)

        df['price_above_upper'] = df['Close'] > df['UPPER']
        df['price_below_lower'] = df['Close'] < df['LOWER']
        df['price_above_mid'] = df['Close'] > df['MID']
        df['price_below_mid'] = df['Close'] < df['MID']
        df['boll_rising'] = df['MID'] > df['MID'].shift(1)
        df['boll_falling'] = df['MID'] < df['MID'].shift(1)
        df['boll_widening'] = df['BWIDTH'] > df['BWIDTH'].shift(1)
        df['boll_narrowing'] = df['BWIDTH'] < df['BWIDTH'].shift(1)

        return df[['MID', 'STD', 'UPPER', 'LOWER', 'BWIDTH', 'PERCENT_B',
                   'price_above_upper', 'price_below_lower',
                   'price_above_mid', 'price_below_mid',
                   'boll_rising', 'boll_falling',
                   'boll_widening', 'boll_narrowing']]
