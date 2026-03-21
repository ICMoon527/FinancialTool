import pandas as pd
import numpy as np
from ..base import BaseIndicator


class LWR(BaseIndicator):
    """
    LWR（Larry Williams %R）指标，中文称为威廉指标或威廉超买超卖指标（变种）。

    LWR指标是WR指标的变种，计算方式相同，但数值范围是0-100而不是0到-100。
    LWR指标通过计算当日收盘价在N日最高价最低价区间的位置，来判断价格的超买超卖状态。

    计算公式：
    LWR = (HHV(High, N) - Close) / (HHV(High, N) - LLV(Low, N)) × 100
    其中：
    - HHV(High, N)为N日内的最高价
    - LLV(Low, N)为N日内的最低价
    - N为计算周期，通常为10日、14日或20日

    使用场景：
    - LWR在0-100之间波动
    - LWR < 20 时，处于超买区间，可能回落，为卖出信号
    - LWR > 80 时，处于超卖区间，可能反弹，为买入信号
    - LWR由下往上穿越20时，为卖出信号
    - LWR由上往下跌破80时，为买入信号
    - LWR在50以下时，为多头市场
    - LWR在50以上时，为空头市场

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14

    输出参数：
    - LWR: 威廉指标
    - overbought: 超买信号（LWR < 20）
    - oversold: 超卖信号（LWR > 80）
    - cross_20_up: 上穿20信号
    - cross_80_down: 下穿80信号
    - bullish: LWR < 50信号
    - bearish: LWR > 50信号

    注意事项：
    - LWR是震荡指标，在震荡市中表现较好
    - LWR在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - LWR反应快速，适合短线操作

    最佳实践建议：
    1. 超买超卖：在超卖区买入和超买区卖出更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的LWR
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14):
        """
        初始化LWR指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算LWR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含LWR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        hh = df['High'].rolling(window=self.period).max()
        ll = df['Low'].rolling(window=self.period).min()

        df['LWR'] = (hh - df['Close']) / (hh - ll).replace(0, np.nan) * 100

        df['overbought'] = df['LWR'] < 20
        df['oversold'] = df['LWR'] > 80
        df['cross_20_up'] = (df['LWR'] < 20) & (df['LWR'].shift(1) >= 20)
        df['cross_80_down'] = (df['LWR'] > 80) & (df['LWR'].shift(1) <= 80)
        df['bullish'] = df['LWR'] < 50
        df['bearish'] = df['LWR'] > 50

        return df[['LWR', 'overbought', 'oversold', 'cross_20_up', 'cross_80_down', 
                   'bullish', 'bearish']]
