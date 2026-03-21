import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DBQR(BaseIndicator):
    """
    DBQR（对比强弱指标），中文称为对比强弱指标。

    DBQR指标是一种对比指标，通过计算股票收益率与大盘收益率的比值，
    来衡量股票相对于大盘的强弱程度。

    计算公式：
    1. 计算股票收益率：Stock_Return = (Close - Close_prev) / Close_prev
    2. 计算大盘收益率：Index_Return = (Index_Close - Index_Close_prev) / Index_Close_prev
    3. 计算DBQR = (Stock_Return - Index_Return) × 100
    4. 计算DBQR_MA = MA(DBQR, N)

    使用场景：
    - DBQR > 0 时，表示股票强于大盘，为买入信号
    - DBQR < 0 时，表示股票弱于大盘，为卖出信号
    - DBQR上升时，表示股票相对大盘走强，为买入信号
    - DBQR下降时，表示股票相对大盘走弱，为卖出信号
    - DBQR由负转正时，为买入信号
    - DBQR由正转负时，为卖出信号
    - DBQR创新高时，表示股票相对大盘持续走强，为买入信号
    - DBQR创新低时，表示股票相对大盘持续走弱，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'、'Index_Close'列
    - period: 移动平均周期，默认10

    输出参数：
    - Stock_Return: 股票收益率
    - Index_Return: 大盘收益率
    - DBQR: 对比强弱指标
    - DBQR_MA: DBQR的移动平均线
    - dbqr_positive: DBQR > 0信号
    - dbqr_negative: DBQR < 0信号
    - dbqr_rising: DBQR上升信号
    - dbqr_falling: DBQR下降信号
    - dbqr_new_high: DBQR创新高信号
    - dbqr_new_low: DBQR创新低信号
    - dbqr_cross_zero_up: DBQR上穿0线信号
    - dbqr_cross_zero_down: DBQR下穿0线信号
    - dbqr_above_ma: DBQR在MA上方信号
    - dbqr_below_ma: DBQR在MA下方信号

    注意事项：
    - DBQR是对比指标，反应股票相对大盘的强弱
    - DBQR需要大盘数据配合使用
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 对比分析：DBQR > 0 表示股票强于大盘
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DBQR
    4. 背离判断：注意DBQR与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化DBQR指标

        Parameters
        ----------
        period : int, default 10
            移动平均周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算DBQR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Index_Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DBQR计算结果的DataFrame
        """
        df = data.copy()

        close_prev = df['Close'].shift(1)
        index_close_prev = df['Index_Close'].shift(1)

        df['Stock_Return'] = (df['Close'] - close_prev) / close_prev.replace(0, np.nan)
        df['Index_Return'] = (df['Index_Close'] - index_close_prev) / index_close_prev.replace(0, np.nan)
        df['DBQR'] = (df['Stock_Return'] - df['Index_Return']) * 100
        df['DBQR_MA'] = df['DBQR'].rolling(window=self.period).mean()

        df['dbqr_positive'] = df['DBQR'] > 0
        df['dbqr_negative'] = df['DBQR'] < 0
        df['dbqr_rising'] = df['DBQR'] > df['DBQR'].shift(1)
        df['dbqr_falling'] = df['DBQR'] < df['DBQR'].shift(1)
        df['dbqr_new_high'] = df['DBQR'] == df['DBQR'].rolling(window=self.period).max()
        df['dbqr_new_low'] = df['DBQR'] == df['DBQR'].rolling(window=self.period).min()
        df['dbqr_cross_zero_up'] = (df['DBQR'] > 0) & (df['DBQR'].shift(1) <= 0)
        df['dbqr_cross_zero_down'] = (df['DBQR'] < 0) & (df['DBQR'].shift(1) >= 0)
        df['dbqr_above_ma'] = df['DBQR'] > df['DBQR_MA']
        df['dbqr_below_ma'] = df['DBQR'] < df['DBQR_MA']

        return df[['Stock_Return', 'Index_Return', 'DBQR', 'DBQR_MA', 'dbqr_positive', 'dbqr_negative',
                   'dbqr_rising', 'dbqr_falling', 'dbqr_new_high', 'dbqr_new_low',
                   'dbqr_cross_zero_up', 'dbqr_cross_zero_down', 'dbqr_above_ma', 'dbqr_below_ma']]
