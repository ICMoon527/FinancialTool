import pandas as pd
import numpy as np
from ..base import BaseIndicator


class LON(BaseIndicator):
    """
    LON（龙系指标），中文称为龙系指标。

    LON指标是一种量价指标，通过计算成交量的变化，来判断资金的流入流出。
    LON指标结合了成交量和价格，来反应市场的资金流向。

    计算公式：
    1. 计算MFM（资金流向乘数）：MFM = ((Close - Low) - (High - Close)) / (High - Low)
    2. 计算MFV（资金流向量）：MFV = MFM × Volume
    3. 计算LON：LON = sum(MFV)

    使用场景：
    - LON上升时，表示资金流入，为买入信号
    - LON下降时，表示资金流出，为卖出信号
    - LON与价格同步时，趋势健康
    - LON与价格背离时，可能反转
    - LON创新高时，表示资金强劲流入
    - LON创新低时，表示资金强劲流出

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列

    输出参数：
    - MFM: 资金流向乘数
    - MFV: 资金流向量
    - LON: 龙系指标
    - LON_MA: LON的移动平均线
    - lon_positive: LON > 0信号
    - lon_negative: LON < 0信号
    - lon_rising: LON上升信号
    - lon_falling: LON下降信号
    - lon_above_ma: LON在MA上方信号
    - lon_below_ma: LON在MA下方信号
    - lon_new_high: LON创新高信号
    - lon_new_low: LON创新低信号

    注意事项：
    - LON是量价指标，反应资金流动
    - LON在震荡市中会产生频繁的假信号
    - LON在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：LON和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的LON
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化LON指标

        Parameters
        ----------
        ma_period : int, default 10
            移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算LON指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含LON计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low.replace(0, np.nan)
        df['MFM'] = df['MFM'].fillna(0)
        df['MFV'] = df['MFM'] * df['Volume']
        df['LON'] = df['MFV'].cumsum()
        df['LON_MA'] = df['LON'].rolling(window=self.ma_period).mean()

        df['lon_positive'] = df['LON'] > 0
        df['lon_negative'] = df['LON'] < 0
        df['lon_rising'] = df['LON'] > df['LON'].shift(1)
        df['lon_falling'] = df['LON'] < df['LON'].shift(1)
        df['lon_above_ma'] = df['LON'] > df['LON_MA']
        df['lon_below_ma'] = df['LON'] < df['LON_MA']
        df['lon_new_high'] = df['LON'] == df['LON'].rolling(window=self.ma_period * 2).max()
        df['lon_new_low'] = df['LON'] == df['LON'].rolling(window=self.ma_period * 2).min()

        return df[['MFM', 'MFV', 'LON', 'LON_MA',
                   'lon_positive', 'lon_negative',
                   'lon_rising', 'lon_falling',
                   'lon_above_ma', 'lon_below_ma',
                   'lon_new_high', 'lon_new_low']]
