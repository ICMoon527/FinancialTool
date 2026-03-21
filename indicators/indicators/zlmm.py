import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ZLMM(BaseIndicator):
    """
    ZLMM（主力买卖），中文称为主力买卖。

    ZLMM指标是一种量价指标，通过计算主力资金的买卖情况，来判断主力的动向。
    ZLMM指标结合了成交量和价格，来反应主力资金的流入流出。

    计算公式：
    1. 计算MFM（资金流向乘数）：MFM = ((Close - Low) - (High - Close)) / (High - Low)
    2. 计算MFV（资金流向量）：MFV = MFM × Volume
    3. 计算ZLMM：ZLMM = EMA(MFV, N)

    使用场景：
    - ZLMM上升时，表示主力资金流入，为买入信号
    - ZLMM下降时，表示主力资金流出，为卖出信号
    - ZLMM与价格同步时，趋势健康
    - ZLMM与价格背离时，可能反转
    - ZLMM创新高时，表示主力资金强劲流入
    - ZLMM创新低时，表示主力资金强劲流出

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列
    - period: 计算周期，默认10

    输出参数：
    - MFM: 资金流向乘数
    - MFV: 资金流向量
    - ZLMM: 主力买卖
    - ZLMM_MA: ZLMM的移动平均线
    - zlmm_positive: ZLMM > 0信号
    - zlmm_negative: ZLMM < 0信号
    - zlmm_rising: ZLMM上升信号
    - zlmm_falling: ZLMM下降信号
    - zlmm_above_ma: ZLMM在MA上方信号
    - zlmm_below_ma: ZLMM在MA下方信号
    - zlmm_new_high: ZLMM创新高信号
    - zlmm_new_low: ZLMM创新低信号

    注意事项：
    - ZLMM是量价指标，反应主力资金流动
    - ZLMM在震荡市中会产生频繁的假信号
    - ZLMM在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ZLMM和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ZLMM
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period=10):
        """
        初始化ZLMM指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算ZLMM指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ZLMM计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low.replace(0, np.nan)
        df['MFM'] = df['MFM'].fillna(0)
        df['MFV'] = df['MFM'] * df['Volume']
        df['ZLMM'] = df['MFV'].ewm(span=self.period, adjust=False).mean()
        df['ZLMM_MA'] = df['ZLMM'].rolling(window=self.ma_period).mean()

        df['zlmm_positive'] = df['ZLMM'] > 0
        df['zlmm_negative'] = df['ZLMM'] < 0
        df['zlmm_rising'] = df['ZLMM'] > df['ZLMM'].shift(1)
        df['zlmm_falling'] = df['ZLMM'] < df['ZLMM'].shift(1)
        df['zlmm_above_ma'] = df['ZLMM'] > df['ZLMM_MA']
        df['zlmm_below_ma'] = df['ZLMM'] < df['ZLMM_MA']
        df['zlmm_new_high'] = df['ZLMM'] == df['ZLMM'].rolling(window=self.ma_period * 2).max()
        df['zlmm_new_low'] = df['ZLMM'] == df['ZLMM'].rolling(window=self.ma_period * 2).min()

        return df[['MFM', 'MFV', 'ZLMM', 'ZLMM_MA',
                   'zlmm_positive', 'zlmm_negative',
                   'zlmm_rising', 'zlmm_falling',
                   'zlmm_above_ma', 'zlmm_below_ma',
                   'zlmm_new_high', 'zlmm_new_low']]
