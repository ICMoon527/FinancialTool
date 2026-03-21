import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PVT(BaseIndicator):
    """
    PVT（Price Volume Trend）指标，中文称为价格成交量趋势指标。

    PVT指标是一种量价指标，通过计算价格变动率与成交量的乘积，
    来衡量市场的买卖力量和资金流向。

    计算公式：
    1. 计算价格变动率：Price_Change = (Close - Close_prev) / Close_prev
    2. 计算PVT变化量：PVT_Change = Volume × Price_Change
    3. 计算PVT：PVT = Σ(PVT_Change)

    使用场景：
    - PVT > 0 时，表示资金流入，为多头市场
    - PVT < 0 时，表示资金流出，为空头市场
    - PVT上升时，表示资金流入增强，为买入信号
    - PVT下降时，表示资金流出增强，为卖出信号
    - PVT创新高时，表示资金持续流入
    - PVT创新低时，表示资金持续流出
    - PVT与价格同步上升时，趋势健康
    - PVT与价格同步下降时，趋势健康
    - PVT与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列

    输出参数：
    - Price_Change: 价格变动率
    - PVT_Change: PVT变化量
    - PVT: 价格成交量趋势指标
    - PVT_MA: PVT的移动平均线
    - pvt_positive: PVT > 0信号
    - pvt_negative: PVT < 0信号
    - pvt_rising: PVT上升信号
    - pvt_falling: PVT下降信号
    - pvt_new_high: PVT创新高信号
    - pvt_new_low: PVT创新低信号
    - pvt_above_ma: PVT在MA上方信号
    - pvt_below_ma: PVT在MA下方信号

    注意事项：
    - PVT是量价指标，反应资金流向
    - PVT在震荡市中会产生频繁的假信号
    - PVT在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：PVT和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的PVT
    4. 背离判断：注意PVT与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化PVT指标

        Parameters
        ----------
        ma_period : int, default 10
            移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算PVT指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PVT计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        df['Price_Change'] = (df['Close'] - close_prev) / close_prev.replace(0, np.nan)
        df['PVT_Change'] = df['Volume'] * df['Price_Change']
        df['PVT'] = df['PVT_Change'].cumsum()
        df['PVT_MA'] = df['PVT'].rolling(window=self.ma_period).mean()

        df['pvt_positive'] = df['PVT'] > 0
        df['pvt_negative'] = df['PVT'] < 0
        df['pvt_rising'] = df['PVT'] > df['PVT'].shift(1)
        df['pvt_falling'] = df['PVT'] < df['PVT'].shift(1)
        df['pvt_new_high'] = df['PVT'] == df['PVT'].rolling(window=self.ma_period).max()
        df['pvt_new_low'] = df['PVT'] == df['PVT'].rolling(window=self.ma_period).min()
        df['pvt_above_ma'] = df['PVT'] > df['PVT_MA']
        df['pvt_below_ma'] = df['PVT'] < df['PVT_MA']

        return df[['Price_Change', 'PVT_Change', 'PVT', 'PVT_MA',
                   'pvt_positive', 'pvt_negative', 'pvt_rising', 'pvt_falling',
                   'pvt_new_high', 'pvt_new_low', 'pvt_above_ma', 'pvt_below_ma']]
