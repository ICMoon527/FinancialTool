import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CYW(BaseIndicator):
    """
    CYW（主力控盘），中文称为主力控盘。

    CYW指标是一种量价指标，通过计算成交量的变化，来判断主力的控盘程度。
    CYW指标结合了成交量和价格，来反应主力的控盘力度。

    计算公式：
    1. 计算MFM（资金流向乘数）：MFM = ((Close - Low) - (High - Close)) / (High - Low)
    2. 计算MFV（资金流向量）：MFV = MFM × Volume
    3. 计算CYW：CYW = sum(MFV) / sum(Volume)

    使用场景：
    - CYW上升时，表示主力控盘程度增加，为买入信号
    - CYW下降时，表示主力控盘程度减少，为卖出信号
    - CYW > 0 时，表示主力控盘
    - CYW < 0 时，表示主力未控盘
    - CYW与价格同步时，趋势健康
    - CYW与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列
    - period: 计算周期，默认10

    输出参数：
    - MFM: 资金流向乘数
    - MFV: 资金流向量
    - CYW: 主力控盘
    - CYW_MA: CYW的移动平均线
    - cyw_positive: CYW > 0信号
    - cyw_negative: CYW < 0信号
    - cyw_rising: CYW上升信号
    - cyw_falling: CYW下降信号
    - cyw_above_ma: CYW在MA上方信号
    - cyw_below_ma: CYW在MA下方信号

    注意事项：
    - CYW是量价指标，反应主力控盘程度
    - CYW在震荡市中会产生频繁的假信号
    - CYW在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：CYW和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的CYW
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period=10):
        """
        初始化CYW指标

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
        计算CYW指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CYW计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low.replace(0, np.nan)
        df['MFM'] = df['MFM'].fillna(0)
        df['MFV'] = df['MFM'] * df['Volume']
        df['CYW'] = df['MFV'].rolling(window=self.period).sum() / df['Volume'].rolling(window=self.period).sum().replace(0, np.nan)
        df['CYW'] = df['CYW'].fillna(0)
        df['CYW_MA'] = df['CYW'].rolling(window=self.ma_period).mean()

        df['cyw_positive'] = df['CYW'] > 0
        df['cyw_negative'] = df['CYW'] < 0
        df['cyw_rising'] = df['CYW'] > df['CYW'].shift(1)
        df['cyw_falling'] = df['CYW'] < df['CYW'].shift(1)
        df['cyw_above_ma'] = df['CYW'] > df['CYW_MA']
        df['cyw_below_ma'] = df['CYW'] < df['CYW_MA']

        return df[['MFM', 'MFV', 'CYW', 'CYW_MA',
                   'cyw_positive', 'cyw_negative',
                   'cyw_rising', 'cyw_falling',
                   'cyw_above_ma', 'cyw_below_ma']]
