import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MFIExtended(BaseIndicator):
    """
    MFIExtended（资金流量指标扩展），中文称为资金流量指标扩展。

    MFIExtended指标是MFI指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    MFIExtended指标结合了典型价格和成交量，来反应市场的资金流向。

    计算公式：
    1. 计算TypicalPrice：TypicalPrice = (High + Low + Close) / 3
    2. 计算RawMoneyFlow：RawMoneyFlow = TypicalPrice × Volume
    3. 计算PositiveMoneyFlow：PositiveMoneyFlow = RawMoneyFlow if TypicalPrice > TypicalPrice_prev else 0
    4. 计算NegativeMoneyFlow：NegativeMoneyFlow = RawMoneyFlow if TypicalPrice < TypicalPrice_prev else 0
    5. 计算MoneyFlowRatio：MoneyFlowRatio = sum(PositiveMoneyFlow, N) / sum(NegativeMoneyFlow, N)
    6. 计算MFI：MFI = 100 - 100 / (1 + MoneyFlowRatio)
    7. 计算MFI_MA：MFI_MA = EMA(MFI, M)

    使用场景：
    - MFI > 80 时，表示超买，为卖出信号
    - MFI < 20 时，表示超卖，为买入信号
    - MFI上升时，表示资金流入，为买入信号
    - MFI下降时，表示资金流出，为卖出信号
    - MFI与价格同步时，趋势健康
    - MFI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列
    - period: 计算周期，默认14
    - ma_period: MFI移动平均周期，默认10

    输出参数：
    - TypicalPrice: 典型价格
    - RawMoneyFlow: 原始资金流量
    - PositiveMoneyFlow: 正资金流量
    - NegativeMoneyFlow: 负资金流量
    - MoneyFlowRatio: 资金流量比率
    - MFI: 资金流量指标
    - MFI_MA: MFI的移动平均线
    - mfi_overbought: MFI超买信号（>80）
    - mfi_oversold: MFI超卖信号（<20）
    - mfi_above_ma: MFI在MA上方信号
    - mfi_below_ma: MFI在MA下方信号
    - mfi_rising: MFI上升信号
    - mfi_falling: MFI下降信号

    注意事项：
    - MFIExtended是量价指标，反应资金流动
    - MFIExtended在震荡市中会产生频繁的假信号
    - MFIExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：MFI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的MFIExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14, ma_period=10):
        """
        初始化MFIExtended指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        ma_period : int, default 10
            MFI移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算MFIExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MFIExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['RawMoneyFlow'] = df['TypicalPrice'] * df['Volume']

        price_change = df['TypicalPrice'].diff()
        df['PositiveMoneyFlow'] = np.where(price_change > 0, df['RawMoneyFlow'], 0)
        df['NegativeMoneyFlow'] = np.where(price_change < 0, df['RawMoneyFlow'], 0)

        positive_sum = df['PositiveMoneyFlow'].rolling(window=self.period).sum()
        negative_sum = df['NegativeMoneyFlow'].rolling(window=self.period).sum()
        df['MoneyFlowRatio'] = positive_sum / negative_sum.replace(0, np.nan)
        df['MoneyFlowRatio'] = df['MoneyFlowRatio'].fillna(0)
        df['MFI'] = 100 - 100 / (1 + df['MoneyFlowRatio'])
        df['MFI_MA'] = df['MFI'].ewm(span=self.ma_period, adjust=False).mean()

        df['mfi_overbought'] = df['MFI'] > 80
        df['mfi_oversold'] = df['MFI'] < 20
        df['mfi_above_ma'] = df['MFI'] > df['MFI_MA']
        df['mfi_below_ma'] = df['MFI'] < df['MFI_MA']
        df['mfi_rising'] = df['MFI'] > df['MFI'].shift(1)
        df['mfi_falling'] = df['MFI'] < df['MFI'].shift(1)

        return df[['TypicalPrice', 'RawMoneyFlow', 'PositiveMoneyFlow', 'NegativeMoneyFlow', 'MoneyFlowRatio', 'MFI', 'MFI_MA',
                   'mfi_overbought', 'mfi_oversold',
                   'mfi_above_ma', 'mfi_below_ma',
                   'mfi_rising', 'mfi_falling']]
