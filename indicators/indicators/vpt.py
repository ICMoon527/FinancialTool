import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VPT(BaseIndicator):
    """
    VPT（Volume Price Trend）指标，中文称为量价趋势指标。

    VPT指标是一种结合了价格和成交量的指标，通过计算价格变动与成交量的关系，
    来衡量市场的买卖力量和资金流向。

    计算公式：
    1. 计算价格变动率：Price_Change = (Close - Close_prev) / Close_prev
    2. 计算VPT：VPT = VPT_prev + Volume × Price_Change

    使用场景：
    - VPT上升时，表示资金流入，为买入信号
    - VPT下降时，表示资金流出，为卖出信号
    - VPT创新高时，表示资金持续流入，为买入信号
    - VPT创新低时，表示资金持续流出，为卖出信号
    - VPT与价格同步上升时，趋势健康
    - VPT与价格同步下降时，趋势健康
    - VPT与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列

    输出参数：
    - VPT: 量价趋势指标
    - VPT_MA: VPT的移动平均线
    - vpt_rising: VPT上升信号
    - vpt_falling: VPT下降信号
    - vpt_new_high: VPT创新高信号
    - vpt_new_low: VPT创新低信号
    - vpt_above_ma: VPT在MA上方信号
    - vpt_below_ma: VPT在MA下方信号
    - vpt_positive: VPT > 0信号
    - vpt_negative: VPT < 0信号

    注意事项：
    - VPT是量价指标，反应资金流向
    - VPT在震荡市中会产生频繁的假信号
    - VPT在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VPT和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的VPT
    4. 背离判断：注意VPT与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化VPT指标

        Parameters
        ----------
        ma_period : int, default 10
            移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算VPT指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VPT计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        price_change = (df['Close'] - close_prev) / close_prev.replace(0, np.nan)
        vpt_changes = df['Volume'] * price_change
        df['VPT'] = vpt_changes.cumsum()

        df['VPT_MA'] = df['VPT'].rolling(window=self.ma_period).mean()

        df['vpt_rising'] = df['VPT'] > df['VPT'].shift(1)
        df['vpt_falling'] = df['VPT'] < df['VPT'].shift(1)
        df['vpt_new_high'] = df['VPT'] == df['VPT'].rolling(window=self.ma_period).max()
        df['vpt_new_low'] = df['VPT'] == df['VPT'].rolling(window=self.ma_period).min()
        df['vpt_above_ma'] = df['VPT'] > df['VPT_MA']
        df['vpt_below_ma'] = df['VPT'] < df['VPT_MA']
        df['vpt_positive'] = df['VPT'] > 0
        df['vpt_negative'] = df['VPT'] < 0

        return df[['VPT', 'VPT_MA', 'vpt_rising', 'vpt_falling', 
                   'vpt_new_high', 'vpt_new_low', 'vpt_above_ma', 'vpt_below_ma',
                   'vpt_positive', 'vpt_negative']]
