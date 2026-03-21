import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BOP(BaseIndicator):
    """
    BOP（Balance of Power）指标，中文称为平衡力量指标。

    BOP指标是一种量价指标，通过计算开盘价、最高价、最低价、收盘价的关系和成交量，
    来衡量买卖双方的力量对比。

    计算公式：
    BOP = (Close - Open) / (High - Low) × Volume

    使用场景：
    - BOP > 0 时，表示买方力量较强，为买入信号
    - BOP < 0 时，表示卖方力量较强，为卖出信号
    - BOP上升时，表示买方力量增强，为买入信号
    - BOP下降时，表示卖方力量增强，为卖出信号
    - BOP创新高时，表示买方力量强劲
    - BOP创新低时，表示卖方力量强劲
    - BOP由负转正时，为买入信号
    - BOP由正转负时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Open'、'High'、'Low'、'Close'、'Volume'列
    - sma_period: SMA周期，默认14

    输出参数：
    - BOP: 平衡力量指标
    - BOP_SMA: BOP的简单移动平均线
    - bop_positive: BOP > 0信号
    - bop_negative: BOP < 0信号
    - bop_rising: BOP上升信号
    - bop_falling: BOP下降信号
    - bop_new_high: BOP创新高信号
    - bop_new_low: BOP创新低信号
    - bop_cross_zero_up: BOP上穿0线信号
    - bop_cross_zero_down: BOP下穿0线信号
    - bop_above_sma: BOP在SMA上方信号
    - bop_below_sma: BOP在SMA下方信号

    注意事项：
    - BOP是量价指标，反应买卖双方力量对比
    - BOP在震荡市中会产生频繁的假信号
    - BOP在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：BOP和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的BOP
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, sma_period=14):
        """
        初始化BOP指标

        Parameters
        ----------
        sma_period : int, default 14
            SMA周期
        """
        self.sma_period = sma_period

    def calculate(self, data):
        """
        计算BOP指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Open'、'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BOP计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        price_range = df['High'] - df['Low']
        df['BOP'] = (df['Close'] - df['Open']) / price_range.replace(0, np.nan) * df['Volume']
        df['BOP_SMA'] = df['BOP'].rolling(window=self.sma_period).mean()

        df['bop_positive'] = df['BOP'] > 0
        df['bop_negative'] = df['BOP'] < 0
        df['bop_rising'] = df['BOP'] > df['BOP'].shift(1)
        df['bop_falling'] = df['BOP'] < df['BOP'].shift(1)
        df['bop_new_high'] = df['BOP'] == df['BOP'].rolling(window=self.sma_period).max()
        df['bop_new_low'] = df['BOP'] == df['BOP'].rolling(window=self.sma_period).min()
        df['bop_cross_zero_up'] = (df['BOP'] > 0) & (df['BOP'].shift(1) <= 0)
        df['bop_cross_zero_down'] = (df['BOP'] < 0) & (df['BOP'].shift(1) >= 0)
        df['bop_above_sma'] = df['BOP'] > df['BOP_SMA']
        df['bop_below_sma'] = df['BOP'] < df['BOP_SMA']

        return df[['BOP', 'BOP_SMA', 'bop_positive', 'bop_negative',
                   'bop_rising', 'bop_falling', 'bop_new_high', 'bop_new_low',
                   'bop_cross_zero_up', 'bop_cross_zero_down',
                   'bop_above_sma', 'bop_below_sma']]
