import pandas as pd
import numpy as np
from ..base import BaseIndicator


class WAD(BaseIndicator):
    """
    WAD（威廉累积/派发指标），中文称为威廉累积派发指标。

    WAD指标是由Larry Williams提出的，是一种量价指标。
    WAD指标通过计算价格变动，来判断资金的累积和派发。

    计算公式：
    1. 计算TRH（真实高价）：TRH = max(High, Close_prev)
    2. 计算TRL（真实低价）：TRL = min(Low, Close_prev)
    3. 如果Close > Close_prev：WAD = Close - TRL
    4. 如果Close < Close_prev：WAD = Close - TRH
    5. 如果Close == Close_prev：WAD = 0
    6. 计算累积WAD：Accum_WAD = sum(WAD)

    使用场景：
    - WAD上升时，表示资金累积，为买入信号
    - WAD下降时，表示资金派发，为卖出信号
    - WAD与价格同步时，趋势健康
    - WAD与价格背离时，可能反转
    - WAD创新高时，表示资金强劲流入
    - WAD创新低时，表示资金强劲流出

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列

    输出参数：
    - TRH: 真实高价
    - TRL: 真实低价
    - WAD: 威廉累积/派发指标
    - Accum_WAD: 累积WAD
    - Accum_WAD_MA: 累积WAD的移动平均线
    - wad_positive: WAD > 0信号
    - wad_negative: WAD < 0信号
    - accum_wad_rising: 累积WAD上升信号
    - accum_wad_falling: 累积WAD下降信号
    - accum_wad_new_high: 累积WAD创新高信号
    - accum_wad_new_low: 累积WAD创新低信号

    注意事项：
    - WAD是量价指标，反应资金流动
    - WAD在震荡市中会产生频繁的假信号
    - WAD在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：WAD和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的WAD
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化WAD指标

        Parameters
        ----------
        ma_period : int, default 10
            移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算WAD指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含WAD计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        df['TRH'] = df[['High', close_prev]].max(axis=1)
        df['TRL'] = df[['Low', close_prev]].min(axis=1)

        wad_values = []
        for i in range(len(df)):
            if pd.isna(close_prev.iloc[i]):
                wad_values.append(0)
            elif df['Close'].iloc[i] > close_prev.iloc[i]:
                wad_values.append(df['Close'].iloc[i] - df['TRL'].iloc[i])
            elif df['Close'].iloc[i] < close_prev.iloc[i]:
                wad_values.append(df['Close'].iloc[i] - df['TRH'].iloc[i])
            else:
                wad_values.append(0)

        df['WAD'] = wad_values
        df['Accum_WAD'] = df['WAD'].cumsum()
        df['Accum_WAD_MA'] = df['Accum_WAD'].rolling(window=self.ma_period).mean()

        df['wad_positive'] = df['WAD'] > 0
        df['wad_negative'] = df['WAD'] < 0
        df['accum_wad_rising'] = df['Accum_WAD'] > df['Accum_WAD'].shift(1)
        df['accum_wad_falling'] = df['Accum_WAD'] < df['Accum_WAD'].shift(1)
        df['accum_wad_new_high'] = df['Accum_WAD'] == df['Accum_WAD'].rolling(window=self.ma_period * 2).max()
        df['accum_wad_new_low'] = df['Accum_WAD'] == df['Accum_WAD'].rolling(window=self.ma_period * 2).min()

        return df[['TRH', 'TRL', 'WAD', 'Accum_WAD', 'Accum_WAD_MA',
                   'wad_positive', 'wad_negative',
                   'accum_wad_rising', 'accum_wad_falling',
                   'accum_wad_new_high', 'accum_wad_new_low']]
