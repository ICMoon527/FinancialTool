import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ADVOL(BaseIndicator):
    """
    ADVOL（成交量指标），中文称为成交量指标。

    ADVOL指标是一种量价指标，通过计算成交量的变化，来判断价格的趋势。
    ADVOL指标结合了成交量和价格，来反应市场的资金流向。

    计算公式：
    1. 计算MFM（资金流向乘数）：MFM = ((Close - Low) - (High - Close)) / (High - Low)
    2. 计算MFV（资金流向量）：MFV = MFM × Volume
    3. 计算ADVOL：ADVOL = cumsum(MFV)

    使用场景：
    - ADVOL上升时，表示资金流入，为买入信号
    - ADVOL下降时，表示资金流出，为卖出信号
    - ADVOL与价格同步时，趋势健康
    - ADVOL与价格背离时，可能反转
    - ADVOL创新高时，表示资金强劲流入
    - ADVOL创新低时，表示资金强劲流出

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列

    输出参数：
    - MFM: 资金流向乘数
    - MFV: 资金流向量
    - ADVOL: 成交量指标
    - ADVOL_MA: ADVOL的移动平均线
    - advol_positive: ADVOL > 0信号
    - advol_negative: ADVOL < 0信号
    - advol_rising: ADVOL上升信号
    - advol_falling: ADVOL下降信号
    - advol_above_ma: ADVOL在MA上方信号
    - advol_below_ma: ADVOL在MA下方信号
    - advol_new_high: ADVOL创新高信号
    - advol_new_low: ADVOL创新低信号

    注意事项：
    - ADVOL是量价指标，反应资金流动
    - ADVOL在震荡市中会产生频繁的假信号
    - ADVOL在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ADVOL和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ADVOL
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化ADVOL指标

        Parameters
        ----------
        ma_period : int, default 10
            移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算ADVOL指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ADVOL计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low.replace(0, np.nan)
        df['MFM'] = df['MFM'].fillna(0)
        df['MFV'] = df['MFM'] * df['Volume']
        df['ADVOL'] = df['MFV'].cumsum()
        df['ADVOL_MA'] = df['ADVOL'].rolling(window=self.ma_period).mean()

        df['advol_positive'] = df['ADVOL'] > 0
        df['advol_negative'] = df['ADVOL'] < 0
        df['advol_rising'] = df['ADVOL'] > df['ADVOL'].shift(1)
        df['advol_falling'] = df['ADVOL'] < df['ADVOL'].shift(1)
        df['advol_above_ma'] = df['ADVOL'] > df['ADVOL_MA']
        df['advol_below_ma'] = df['ADVOL'] < df['ADVOL_MA']
        df['advol_new_high'] = df['ADVOL'] == df['ADVOL'].rolling(window=self.ma_period * 2).max()
        df['advol_new_low'] = df['ADVOL'] == df['ADVOL'].rolling(window=self.ma_period * 2).min()

        return df[['MFM', 'MFV', 'ADVOL', 'ADVOL_MA',
                   'advol_positive', 'advol_negative',
                   'advol_rising', 'advol_falling',
                   'advol_above_ma', 'advol_below_ma',
                   'advol_new_high', 'advol_new_low']]
