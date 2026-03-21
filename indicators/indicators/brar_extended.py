import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BRARExtended(BaseIndicator):
    """
    BRARExtended（情绪指标扩展），中文称为情绪指标扩展。

    BRARExtended指标是BRAR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    BRARExtended指标结合了人气意愿指标和人气指标，来反应市场的情绪。

    计算公式：
    1. 计算BR：BR = (sum(max(High - Close_prev, 0), N) / sum(max(Close_prev - Low, 0), N)) × 100
    2. 计算AR：AR = (sum(High - Open, N) / sum(Open - Low, N)) × 100
    3. 计算BR_MA：BR_MA = EMA(BR, N)
    4. 计算AR_MA：AR_MA = EMA(AR, N)
    5. 计算BRAR_OSC：BRAR_OSC = (BR_MA - AR_MA) × 100

    使用场景：
    - BR > 150 时，表示超买，为卖出信号
    - BR < 50 时，表示超卖，为买入信号
    - AR > 180 时，表示超买，为卖出信号
    - AR < 40 时，表示超卖，为买入信号
    - BR > 100 时，表示多头市场
    - AR > 100 时，表示多头市场
    - BR上升时，表示人气意愿增强
    - AR上升时，表示人气增强

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Open'、'Close'列
    - period: 计算周期，默认26
    - ma_period: 移动平均周期，默认10

    输出参数：
    - BR: 人气意愿指标
    - AR: 人气指标
    - BR_MA: BR的移动平均线
    - AR_MA: AR的移动平均线
    - BRAR_OSC: BRAR的震荡指标
    - br_overbought: BR超买信号（>150）
    - br_oversold: BR超卖信号（<50）
    - ar_overbought: AR超买信号（>180）
    - ar_oversold: AR超卖信号（<40）
    - br_above_100: BR > 100信号
    - ar_above_100: AR > 100信号
    - br_rising: BR上升信号
    - ar_rising: AR上升信号

    注意事项：
    - BRARExtended是情绪指标，反应市场情绪
    - BRARExtended在震荡市中会产生频繁的假信号
    - BRARExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：BR、AR和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的BRARExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=26, ma_period=10):
        """
        初始化BRARExtended指标

        Parameters
        ----------
        period : int, default 26
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算BRARExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Open'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BRARExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_close = df['Close'].shift(1)
        br_numerator = np.maximum(df['High'] - prev_close, 0).rolling(window=self.period).sum()
        br_denominator = np.maximum(prev_close - df['Low'], 0).rolling(window=self.period).sum()
        df['BR'] = (br_numerator / br_denominator.replace(0, np.nan)) * 100
        df['BR'] = df['BR'].fillna(100)

        ar_numerator = (df['High'] - df['Open']).rolling(window=self.period).sum()
        ar_denominator = (df['Open'] - df['Low']).rolling(window=self.period).sum()
        df['AR'] = (ar_numerator / ar_denominator.replace(0, np.nan)) * 100
        df['AR'] = df['AR'].fillna(100)

        df['BR_MA'] = df['BR'].ewm(span=self.ma_period, adjust=False).mean()
        df['AR_MA'] = df['AR'].ewm(span=self.ma_period, adjust=False).mean()
        df['BRAR_OSC'] = (df['BR_MA'] - df['AR_MA']) * 100

        df['br_overbought'] = df['BR'] > 150
        df['br_oversold'] = df['BR'] < 50
        df['ar_overbought'] = df['AR'] > 180
        df['ar_oversold'] = df['AR'] < 40
        df['br_above_100'] = df['BR'] > 100
        df['ar_above_100'] = df['AR'] > 100
        df['br_rising'] = df['BR'] > df['BR'].shift(1)
        df['ar_rising'] = df['AR'] > df['AR'].shift(1)

        return df[['BR', 'AR', 'BR_MA', 'AR_MA', 'BRAR_OSC',
                   'br_overbought', 'br_oversold',
                   'ar_overbought', 'ar_oversold',
                   'br_above_100', 'ar_above_100',
                   'br_rising', 'ar_rising']]
