import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EhlerRSISmoother(BaseIndicator):
    """
    Ehlers RSI Smoother（埃勒斯RSI平滑器），中文称为埃勒斯RSI平滑器。

    Ehlers RSI Smoother指标是由John Ehlers提出的，是一种平滑的RSI指标。
    Ehlers RSI Smoother指标通过使用高级平滑技术，来减少RSI的噪音。

    计算公式：
    1. 计算PriceChange：PriceChange = Close - Close_prev
    2. 计算Gain：Gain = max(PriceChange, 0)
    3. 计算Loss：Loss = -min(PriceChange, 0)
    4. 计算SmoothedGain：SmoothedGain = EMA(Gain, N)
    5. 计算SmoothedLoss：SmoothedLoss = EMA(Loss, N)
    6. 计算RS：RS = SmoothedGain / SmoothedLoss
    7. 计算RSI：RSI = 100 - 100 / (1 + RS)
    8. 计算SmoothedRSI：SmoothedRSI = EhlersSuperSmoother(RSI, M)

    使用场景：
    - SmoothedRSI > 70 时，为超买信号，可能回落
    - SmoothedRSI < 30 时，为超卖信号，可能反弹
    - SmoothedRSI上升时，表示多头力量增强，为买入信号
    - SmoothedRSI下降时，表示空头力量增强，为卖出信号
    - SmoothedRSI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - rsi_period: RSI周期，默认14
    - smooth_period: 平滑周期，默认10

    输出参数：
    - Gain: 上涨
    - Loss: 下跌
    - SmoothedGain: 平滑上涨
    - SmoothedLoss: 平滑下跌
    - RS: 相对强弱
    - RSI: 相对强弱指标
    - SmoothedRSI: 平滑RSI
    - rsi_overbought: RSI > 70信号
    - rsi_oversold: RSI < 30信号
    - rsi_rising: RSI上升信号
    - rsi_falling: RSI下降信号
    - smoothed_rsi_overbought: SmoothedRSI > 70信号
    - smoothed_rsi_oversold: SmoothedRSI < 30信号
    - smoothed_rsi_rising: SmoothedRSI上升信号
    - smoothed_rsi_falling: SmoothedRSI下降信号

    注意事项：
    - Ehlers RSI Smoother是高级平滑RSI，在震荡市中会产生频繁的假信号
    - Ehlers RSI Smoother在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、BOLL）一起使用

    最佳实践建议：
    1. 趋势确认：SmoothedRSI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Ehlers RSI Smoother
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, rsi_period=14, smooth_period=10):
        """
        初始化Ehlers RSI Smoother指标

        Parameters
        ----------
        rsi_period : int, default 14
            RSI周期
        smooth_period : int, default 10
            平滑周期
        """
        self.rsi_period = rsi_period
        self.smooth_period = smooth_period

    def calculate(self, data):
        """
        计算Ehlers RSI Smoother指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Ehlers RSI Smoother计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        delta = df['Close'].diff()
        df['Gain'] = delta.where(delta > 0, 0)
        df['Loss'] = -delta.where(delta < 0, 0)
        df['SmoothedGain'] = df['Gain'].ewm(span=self.rsi_period, adjust=False).mean()
        df['SmoothedLoss'] = df['Loss'].ewm(span=self.rsi_period, adjust=False).mean()
        df['RS'] = df['SmoothedGain'] / df['SmoothedLoss']
        df['RSI'] = 100 - (100 / (1 + df['RS']))

        alpha = 2 / (self.smooth_period + 1)
        filt = df['RSI'].ewm(span=self.smooth_period, adjust=False).mean()
        df['SmoothedRSI'] = 2 * filt - filt.shift(1)

        df['rsi_overbought'] = df['RSI'] > 70
        df['rsi_oversold'] = df['RSI'] < 30
        df['rsi_rising'] = df['RSI'] > df['RSI'].shift(1)
        df['rsi_falling'] = df['RSI'] < df['RSI'].shift(1)
        df['smoothed_rsi_overbought'] = df['SmoothedRSI'] > 70
        df['smoothed_rsi_oversold'] = df['SmoothedRSI'] < 30
        df['smoothed_rsi_rising'] = df['SmoothedRSI'] > df['SmoothedRSI'].shift(1)
        df['smoothed_rsi_falling'] = df['SmoothedRSI'] < df['SmoothedRSI'].shift(1)

        return df[['Gain', 'Loss', 'SmoothedGain', 'SmoothedLoss', 'RS', 'RSI', 'SmoothedRSI',
                   'rsi_overbought', 'rsi_oversold', 'rsi_rising', 'rsi_falling',
                   'smoothed_rsi_overbought', 'smoothed_rsi_oversold',
                   'smoothed_rsi_rising', 'smoothed_rsi_falling']]
