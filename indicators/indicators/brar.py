import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BRAR(BaseIndicator):
    """
    BRAR指标，中文称为人气意愿指标或情绪指标。

    BRAR指标由BR指标（人气指标）和AR指标（意愿指标）组成，用于衡量市场人气和买卖意愿的强弱。
    BRAR指标通过计算最高价、最低价与前一日收盘价、开盘价之间的关系，来判断市场的买卖力量。

    计算公式：
    1. AR指标（意愿指标）：
       - AR = Σ(High - Open) / Σ(Open - Low) × 100
       - 分子：N日内最高价与开盘价差值的总和
       - 分母：N日内开盘价与最低价差值的总和
    2. BR指标（人气指标）：
       - BR = Σ(High - Close_prev) / Σ(Close_prev - Low) × 100
       - 分子：N日内最高价与前一日收盘价差值的总和
       - 分母：N日内前一日收盘价与最低价差值的总和

    使用场景：
    - AR > 100 时，多头力量较强
    - AR < 100 时，空头力量较强
    - BR > 100 时，多头人气旺盛
    - BR < 100 时，空头人气旺盛
    - AR在80-120之间为正常区间
    - BR在70-150之间为正常区间
    - AR > 150 时，处于超买区间，可能回落
    - AR < 50 时，处于超卖区间，可能反弹
    - BR > 300 时，处于超买区间，可能回落
    - BR < 40 时，处于超卖区间，可能反弹
    - BR由下往上穿越AR时，为买入信号
    - BR由上往下穿越AR时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Open'、'Close'列
    - period: 计算周期，默认26

    输出参数：
    - AR: 意愿指标
    - BR: 人气指标
    - ar_overbought: AR超买信号
    - ar_oversold: AR超卖信号
    - br_overbought: BR超买信号
    - br_oversold: BR超卖信号
    - br_cross_ar_up: BR上穿AR信号
    - br_cross_ar_down: BR下穿AR信号

    注意事项：
    - BRAR是短线指标，信号频繁
    - BRAR在震荡市中表现较好
    - 超买超卖阈值需要根据市场特性调整
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 双重确认：等待BR和AR同时发出信号再行动
    2. 趋势过滤：先判断大趋势，顺势操作
    3. 组合使用：与成交量指标配合使用
    4. 多周期使用：同时使用日线和周线的BRAR
    5. 背离确认：结合价格和BRAR的背离信号
    6. 时间过滤：避免在重要数据发布前操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, period=26):
        """
        初始化BRAR指标

        Parameters
        ----------
        period : int, default 26
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算BRAR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Open'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BRAR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_open = df['High'] - df['Open']
        open_low = df['Open'] - df['Low']

        ar_numerator = high_open.rolling(window=self.period).sum()
        ar_denominator = open_low.rolling(window=self.period).sum()

        df['AR'] = ar_numerator / ar_denominator.replace(0, np.nan) * 100

        close_prev = df['Close'].shift(1)
        high_close_prev = df['High'] - close_prev
        high_close_prev = np.where(high_close_prev > 0, high_close_prev, 0)

        close_prev_low = close_prev - df['Low']
        close_prev_low = np.where(close_prev_low > 0, close_prev_low, 0)

        br_numerator = pd.Series(high_close_prev, index=df.index).rolling(window=self.period).sum()
        br_denominator = pd.Series(close_prev_low, index=df.index).rolling(window=self.period).sum()

        df['BR'] = br_numerator / br_denominator.replace(0, np.nan) * 100

        df['ar_overbought'] = df['AR'] > 150
        df['ar_oversold'] = df['AR'] < 50
        df['br_overbought'] = df['BR'] > 300
        df['br_oversold'] = df['BR'] < 40

        df['br_cross_ar_up'] = (df['BR'] > df['AR']) & (df['BR'].shift(1) <= df['AR'].shift(1))
        df['br_cross_ar_down'] = (df['BR'] < df['AR']) & (df['BR'].shift(1) >= df['AR'].shift(1))

        return df[['AR', 'BR', 'ar_overbought', 'ar_oversold', 'br_overbought', 
                   'br_oversold', 'br_cross_ar_up', 'br_cross_ar_down']]
