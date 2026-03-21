import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DBLB(BaseIndicator):
    """
    DBLB（对比量比），中文称为对比量比。

    DBLB指标是一种量比指标，通过计算个股成交量与指数成交量的比率，来判断个股相对于大盘的量能变化。
    DBLB指标结合了个股和指数的成交量，来反应个股的相对量能程度。

    计算公式：
    1. 计算Ratio：Ratio = Volume / IndexVolume
    2. 计算DBLB：DBLB = EMA(Ratio, N)

    使用场景：
    - DBLB上升时，表示个股量能强于大盘，为买入信号
    - DBLB下降时，表示个股量能弱于大盘，为卖出信号
    - DBLB > 1 时，表示个股量能强于大盘
    - DBLB < 1 时，表示个股量能弱于大盘
    - DBLB创新高时，表示个股量能相对强势
    - DBLB创新低时，表示个股量能相对弱势

    输入参数：
    - data: DataFrame，必须包含'Volume'和'IndexVolume'列
    - period: 计算周期，默认10

    输出参数：
    - Ratio: 成交量与指数成交量的比率
    - DBLB: 对比量比
    - DBLB_MA: DBLB的移动平均线
    - dblb_above_one: DBLB > 1信号
    - dblb_below_one: DBLB < 1信号
    - dblb_rising: DBLB上升信号
    - dblb_falling: DBLB下降信号
    - dblb_above_ma: DBLB在MA上方信号
    - dblb_below_ma: DBLB在MA下方信号
    - dblb_new_high: DBLB创新高信号
    - dblb_new_low: DBLB创新低信号

    注意事项：
    - DBLB是量比指标，反应个股相对于大盘的量能变化
    - 需要同时有个股和指数的成交量数据
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 数据准备：确保同时有个股和指数的成交量数据
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DBLB
    4. 量价配合：结合价格变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period=10):
        """
        初始化DBLB指标

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
        计算DBLB指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Volume'和'IndexVolume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DBLB计算结果的DataFrame
        """
        df = data.copy()

        if 'IndexVolume' not in df.columns:
            raise ValueError("DataFrame must contain 'IndexVolume' column for DBLB calculation")

        df['Ratio'] = df['Volume'] / df['IndexVolume'].replace(0, np.nan)
        df['DBLB'] = df['Ratio'].ewm(span=self.period, adjust=False).mean()
        df['DBLB_MA'] = df['DBLB'].rolling(window=self.ma_period).mean()

        df['dblb_above_one'] = df['DBLB'] > 1
        df['dblb_below_one'] = df['DBLB'] < 1
        df['dblb_rising'] = df['DBLB'] > df['DBLB'].shift(1)
        df['dblb_falling'] = df['DBLB'] < df['DBLB'].shift(1)
        df['dblb_above_ma'] = df['DBLB'] > df['DBLB_MA']
        df['dblb_below_ma'] = df['DBLB'] < df['DBLB_MA']
        df['dblb_new_high'] = df['DBLB'] == df['DBLB'].rolling(window=self.ma_period * 2).max()
        df['dblb_new_low'] = df['DBLB'] == df['DBLB'].rolling(window=self.ma_period * 2).min()

        return df[['Ratio', 'DBLB', 'DBLB_MA',
                   'dblb_above_one', 'dblb_below_one',
                   'dblb_rising', 'dblb_falling',
                   'dblb_above_ma', 'dblb_below_ma',
                   'dblb_new_high', 'dblb_new_low']]
