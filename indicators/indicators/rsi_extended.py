import pandas as pd
import numpy as np
from ..base import BaseIndicator


class RSIExtended(BaseIndicator):
    """
    RSIExtended（RSI扩展），中文称为RSI扩展。

    RSIExtended指标是RSI指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    RSIExtended指标结合了上涨和下跌的平均幅度，来反应价格的相对强弱。

    计算公式：
    1. 计算PriceChange：PriceChange = Close - Close_prev
    2. 计算Gain：Gain = max(PriceChange, 0)
    3. 计算Loss：Loss = max(-PriceChange, 0)
    4. 计算AvgGain：AvgGain = EMA(Gain, N)
    5. 计算AvgLoss：AvgLoss = EMA(Loss, N)
    6. 计算RS：RS = AvgGain / AvgLoss
    7. 计算RSI：RSI = 100 - 100 / (1 + RS)
    8. 计算RSI_MA：RSI_MA = EMA(RSI, M)

    使用场景：
    - RSI > 70 时，表示超买，为卖出信号
    - RSI < 30 时，表示超卖，为买入信号
    - RSI上穿50时，为买入信号
    - RSI下穿50时，为卖出信号
    - RSI上升时，表示上升趋势
    - RSI下降时，表示下降趋势
    - RSI与价格同步时，趋势健康
    - RSI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认14
    - ma_period: RSI移动平均周期，默认9

    输出参数：
    - PriceChange: 价格变化
    - Gain: 上涨幅度
    - Loss: 下跌幅度
    - AvgGain: 平均上涨幅度
    - AvgLoss: 平均下跌幅度
    - RS: 相对强弱
    - RSI: 相对强弱指数
    - RSI_MA: RSI的移动平均线
    - rsi_overbought: RSI超买信号（>70）
    - rsi_oversold: RSI超卖信号（<30）
    - rsi_above_50: RSI在50上方信号
    - rsi_below_50: RSI在50下方信号
    - rsi_above_ma: RSI在MA上方信号
    - rsi_below_ma: RSI在MA下方信号
    - rsi_cross_50_up: RSI上穿50信号
    - rsi_cross_50_down: RSI下穿50信号

    注意事项：
    - RSIExtended是摆动指标，在震荡市中表现最好
    - RSIExtended在单边市中会产生钝化现象
    - RSIExtended反应较快，适合中短线操作
    - 建议结合其他指标（如MACD、KDJ）一起使用

    最佳实践建议：
    1. 趋势确认：价格和RSI同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的RSIExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14, ma_period=9):
        """
        初始化RSIExtended指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        ma_period : int, default 9
            RSI移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算RSIExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含RSIExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['PriceChange'] = df['Close'].diff()
        df['Gain'] = df['PriceChange'].where(df['PriceChange'] > 0, 0)
        df['Loss'] = -df['PriceChange'].where(df['PriceChange'] < 0, 0)

        df['AvgGain'] = df['Gain'].ewm(span=self.period, adjust=False).mean()
        df['AvgLoss'] = df['Loss'].ewm(span=self.period, adjust=False).mean()

        df['RS'] = df['AvgGain'] / df['AvgLoss'].replace(0, np.nan)
        df['RS'] = df['RS'].fillna(0)
        df['RSI'] = 100 - 100 / (1 + df['RS'])
        df['RSI_MA'] = df['RSI'].ewm(span=self.ma_period, adjust=False).mean()

        df['rsi_overbought'] = df['RSI'] > 70
        df['rsi_oversold'] = df['RSI'] < 30
        df['rsi_above_50'] = df['RSI'] > 50
        df['rsi_below_50'] = df['RSI'] < 50
        df['rsi_above_ma'] = df['RSI'] > df['RSI_MA']
        df['rsi_below_ma'] = df['RSI'] < df['RSI_MA']
        df['rsi_cross_50_up'] = (df['RSI'] > 50) & (df['RSI'].shift(1) <= 50)
        df['rsi_cross_50_down'] = (df['RSI'] < 50) & (df['RSI'].shift(1) >= 50)

        return df[['PriceChange', 'Gain', 'Loss', 'AvgGain', 'AvgLoss', 'RS', 'RSI', 'RSI_MA',
                   'rsi_overbought', 'rsi_oversold',
                   'rsi_above_50', 'rsi_below_50',
                   'rsi_above_ma', 'rsi_below_ma',
                   'rsi_cross_50_up', 'rsi_cross_50_down']]
