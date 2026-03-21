import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DBQRExtended(BaseIndicator):
    """
    DBQRExtended（对比强弱扩展），中文称为对比强弱扩展。

    DBQRExtended指标是DBQR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    DBQRExtended指标结合了股价和指数，来反应个股与大盘的相对强弱。

    计算公式：
    1. 计算EMA1：EMA1 = EMA(Close, N)
    2. 计算EMA2：EMA2 = EMA(Index, N)
    3. 计算DBQR：DBQR = (EMA1 - EMA2) / EMA2 × 100
    4. 计算DBQR_MA1：DBQR_MA1 = EMA(DBQR, M)
    5. 计算DBQR_MA2：DBQR_MA2 = EMA(DBQR_MA1, P)
    6. 计算DBQR_OSC：DBQR_OSC = (DBQR_MA1 - DBQR_MA2) × 100

    使用场景：
    - DBQR > 0 时，表示个股强于大盘
    - DBQR < 0 时，表示个股弱于大盘
    - DBQR上升时，表示个股相对强势增强
    - DBQR下降时，表示个股相对强势减弱
    - DBQR上穿0时，为买入信号
    - DBQR下穿0时，为卖出信号
    - DBQR_OSC > 0 时，表示量价配合良好
    - DBQR_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - index_data: DataFrame，必须包含'Close'列的指数数据
    - period: 计算周期，默认10
    - ma_period1: 第一移动平均周期，默认5
    - ma_period2: 第二移动平均周期，默认5

    输出参数：
    - EMA1: 股价的指数移动平均线
    - EMA2: 指数的指数移动平均线
    - DBQR: 对比强弱指标
    - DBQR_MA1: DBQR的第一移动平均线
    - DBQR_MA2: DBQR的第二移动平均线
    - DBQR_OSC: DBQR的震荡指标
    - dbqr_positive: DBQR > 0信号
    - dbqr_negative: DBQR < 0信号
    - dbqr_rising: DBQR上升信号
    - dbqr_falling: DBQR下降信号
    - dbqr_ma1_above_ma2: DBQR_MA1在DBQR_MA2上方信号
    - dbqr_ma1_below_ma2: DBQR_MA1在DBQR_MA2下方信号
    - dbqr_osc_positive: DBQR_OSC > 0信号
    - dbqr_osc_negative: DBQR_OSC < 0信号
    - dbqr_cross_up_0: DBQR上穿0信号
    - dbqr_cross_down_0: DBQR下穿0信号

    注意事项：
    - DBQRExtended是相对强弱指标，反应个股与大盘的对比
    - DBQRExtended在震荡市中会产生频繁的假信号
    - DBQRExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：DBQR和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DBQRExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10, ma_period1=5, ma_period2=5):
        """
        初始化DBQRExtended指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        ma_period1 : int, default 5
            第一移动平均周期
        ma_period2 : int, default 5
            第二移动平均周期
        """
        self.period = period
        self.ma_period1 = ma_period1
        self.ma_period2 = ma_period2

    def calculate(self, data, index_data=None):
        """
        计算DBQRExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据
        index_data : pandas.DataFrame, optional
            包含'Close'列的指数数据，如果不提供，则假设data的Close列就是指数

        Returns
        -------
        pandas.DataFrame
            包含DBQRExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        if index_data is None:
            index_close = df['Close']
        else:
            index_close = index_data['Close']

        df['EMA1'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['EMA2'] = pd.Series(index_close).ewm(span=self.period, adjust=False).mean().values
        df['DBQR'] = ((df['EMA1'] - df['EMA2']) / df['EMA2'].replace(0, np.nan)) * 100
        df['DBQR'] = df['DBQR'].fillna(0)
        df['DBQR_MA1'] = df['DBQR'].ewm(span=self.ma_period1, adjust=False).mean()
        df['DBQR_MA2'] = df['DBQR_MA1'].ewm(span=self.ma_period2, adjust=False).mean()
        df['DBQR_OSC'] = (df['DBQR_MA1'] - df['DBQR_MA2']) * 100

        df['dbqr_positive'] = df['DBQR'] > 0
        df['dbqr_negative'] = df['DBQR'] < 0
        df['dbqr_rising'] = df['DBQR'] > df['DBQR'].shift(1)
        df['dbqr_falling'] = df['DBQR'] < df['DBQR'].shift(1)
        df['dbqr_ma1_above_ma2'] = df['DBQR_MA1'] > df['DBQR_MA2']
        df['dbqr_ma1_below_ma2'] = df['DBQR_MA1'] < df['DBQR_MA2']
        df['dbqr_osc_positive'] = df['DBQR_OSC'] > 0
        df['dbqr_osc_negative'] = df['DBQR_OSC'] < 0
        df['dbqr_cross_up_0'] = (df['DBQR'] > 0) & (df['DBQR'].shift(1) <= 0)
        df['dbqr_cross_down_0'] = (df['DBQR'] < 0) & (df['DBQR'].shift(1) >= 0)

        return df[['EMA1', 'EMA2', 'DBQR', 'DBQR_MA1', 'DBQR_MA2', 'DBQR_OSC',
                   'dbqr_positive', 'dbqr_negative',
                   'dbqr_rising', 'dbqr_falling',
                   'dbqr_ma1_above_ma2', 'dbqr_ma1_below_ma2',
                   'dbqr_osc_positive', 'dbqr_osc_negative',
                   'dbqr_cross_up_0', 'dbqr_cross_down_0']]
