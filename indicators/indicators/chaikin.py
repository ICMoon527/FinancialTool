import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Chaikin(BaseIndicator):
    """
    Chaikin A/D Line（蔡金资金流向指标），中文称为蔡金资金流向指标。

    Chaikin A/D Line指标是由马克·蔡金（Marc Chaikin）提出的，是一种量价指标。
    Chaikin A/D Line指标通过计算收盘价在高低价区间的位置和成交量，
    来衡量市场的买卖力量和资金流向。

    计算公式：
    1. 计算Money Flow Multiplier（资金流向乘数）：MFM = ((Close - Low) - (High - Close)) / (High - Low)
    2. 计算Money Flow Volume（资金流向成交量）：MFV = MFM × Volume
    3. 计算Chaikin A/D Line：AD = AD_prev + MFV

    使用场景：
    - AD上升时，表示资金流入，为买入信号
    - AD下降时，表示资金流出，为卖出信号
    - AD创新高时，表示资金持续流入，为买入信号
    - AD创新低时，表示资金持续流出，为卖出信号
    - AD与价格同步上升时，趋势健康
    - AD与价格同步下降时，趋势健康
    - AD与价格背离时，可能反转
    - AD由负转正时，为买入信号
    - AD由正转负时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列
    - ma_period: 移动平均周期，默认10

    输出参数：
    - MFM: 资金流向乘数
    - MFV: 资金流向成交量
    - AD: 蔡金资金流向线
    - AD_MA: AD的移动平均线
    - ad_positive: AD > 0信号
    - ad_negative: AD < 0信号
    - ad_rising: AD上升信号
    - ad_falling: AD下降信号
    - ad_new_high: AD创新高信号
    - ad_new_low: AD创新低信号
    - ad_above_ma: AD在MA上方信号
    - ad_below_ma: AD在MA下方信号
    - ad_cross_zero_up: AD上穿0线信号
    - ad_cross_zero_down: AD下穿0线信号

    注意事项：
    - Chaikin A/D Line是量价指标，反应资金流向
    - Chaikin A/D Line在震荡市中会产生频繁的假信号
    - Chaikin A/D Line在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：AD和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Chaikin A/D Line
    4. 背离判断：注意AD与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化Chaikin A/D Line指标

        Parameters
        ----------
        ma_period : int, default 10
            移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算Chaikin A/D Line指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Chaikin A/D Line计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low_range = df['High'] - df['Low']
        df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_range.replace(0, np.nan)
        df['MFV'] = df['MFM'] * df['Volume']
        df['AD'] = df['MFV'].cumsum()
        df['AD_MA'] = df['AD'].rolling(window=self.ma_period).mean()

        df['ad_positive'] = df['AD'] > 0
        df['ad_negative'] = df['AD'] < 0
        df['ad_rising'] = df['AD'] > df['AD'].shift(1)
        df['ad_falling'] = df['AD'] < df['AD'].shift(1)
        df['ad_new_high'] = df['AD'] == df['AD'].rolling(window=self.ma_period).max()
        df['ad_new_low'] = df['AD'] == df['AD'].rolling(window=self.ma_period).min()
        df['ad_above_ma'] = df['AD'] > df['AD_MA']
        df['ad_below_ma'] = df['AD'] < df['AD_MA']
        df['ad_cross_zero_up'] = (df['AD'] > 0) & (df['AD'].shift(1) <= 0)
        df['ad_cross_zero_down'] = (df['AD'] < 0) & (df['AD'].shift(1) >= 0)

        return df[['MFM', 'MFV', 'AD', 'AD_MA',
                   'ad_positive', 'ad_negative', 'ad_rising', 'ad_falling',
                   'ad_new_high', 'ad_new_low', 'ad_above_ma', 'ad_below_ma',
                   'ad_cross_zero_up', 'ad_cross_zero_down']]
