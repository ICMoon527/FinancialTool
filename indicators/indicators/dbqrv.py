import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DBQRV(BaseIndicator):
    """
    DBQRV（对比强弱），中文称为对比强弱。

    DBQRV指标是一种相对强弱指标，通过计算价格与指数的比率，来判断个股相对于大盘的强弱。
    DBQRV指标结合了个股和指数，来反应个股的相对强弱程度。

    计算公式：
    1. 计算Ratio：Ratio = Close / IndexClose
    2. 计算DBQRV：DBQRV = EMA(Ratio, N)

    使用场景：
    - DBQRV上升时，表示个股强于大盘，为买入信号
    - DBQRV下降时，表示个股弱于大盘，为卖出信号
    - DBQRV > 1 时，表示个股强于大盘
    - DBQRV < 1 时，表示个股弱于大盘
    - DBQRV创新高时，表示个股相对强势
    - DBQRV创新低时，表示个股相对弱势

    输入参数：
    - data: DataFrame，必须包含'Close'和'IndexClose'列
    - period: 计算周期，默认10

    输出参数：
    - Ratio: 价格与指数的比率
    - DBQRV: 对比强弱
    - DBQRV_MA: DBQRV的移动平均线
    - dbqrv_above_one: DBQRV > 1信号
    - dbqrv_below_one: DBQRV < 1信号
    - dbqrv_rising: DBQRV上升信号
    - dbqrv_falling: DBQRV下降信号
    - dbqrv_above_ma: DBQRV在MA上方信号
    - dbqrv_below_ma: DBQRV在MA下方信号
    - dbqrv_new_high: DBQRV创新高信号
    - dbqrv_new_low: DBQRV创新低信号

    注意事项：
    - DBQRV是相对强弱指标，反应个股相对于大盘的强弱
    - 需要同时有个股和指数的数据
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 数据准备：确保同时有个股和指数的数据
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DBQRV
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period=10):
        """
        初始化DBQRV指标

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
        计算DBQRV指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'和'IndexClose'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DBQRV计算结果的DataFrame
        """
        df = data.copy()

        if 'IndexClose' not in df.columns:
            raise ValueError("DataFrame must contain 'IndexClose' column for DBQRV calculation")

        df['Ratio'] = df['Close'] / df['IndexClose'].replace(0, np.nan)
        df['DBQRV'] = df['Ratio'].ewm(span=self.period, adjust=False).mean()
        df['DBQRV_MA'] = df['DBQRV'].rolling(window=self.ma_period).mean()

        df['dbqrv_above_one'] = df['DBQRV'] > 1
        df['dbqrv_below_one'] = df['DBQRV'] < 1
        df['dbqrv_rising'] = df['DBQRV'] > df['DBQRV'].shift(1)
        df['dbqrv_falling'] = df['DBQRV'] < df['DBQRV'].shift(1)
        df['dbqrv_above_ma'] = df['DBQRV'] > df['DBQRV_MA']
        df['dbqrv_below_ma'] = df['DBQRV'] < df['DBQRV_MA']
        df['dbqrv_new_high'] = df['DBQRV'] == df['DBQRV'].rolling(window=self.ma_period * 2).max()
        df['dbqrv_new_low'] = df['DBQRV'] == df['DBQRV'].rolling(window=self.ma_period * 2).min()

        return df[['Ratio', 'DBQRV', 'DBQRV_MA',
                   'dbqrv_above_one', 'dbqrv_below_one',
                   'dbqrv_rising', 'dbqrv_falling',
                   'dbqrv_above_ma', 'dbqrv_below_ma',
                   'dbqrv_new_high', 'dbqrv_new_low']]
