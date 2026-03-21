import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TRIXExtended(BaseIndicator):
    """
    TRIXExtended（三重指数平滑移动平均扩展），中文称为三重指数平滑移动平均扩展。

    TRIXExtended指标是TRIX指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    TRIXExtended指标结合了三次指数移动平均，来反应价格的趋势。

    计算公式：
    1. 计算EMA1：EMA1 = EMA(Close, N)
    2. 计算EMA2：EMA2 = EMA(EMA1, N)
    3. 计算EMA3：EMA3 = EMA(EMA2, N)
    4. 计算TRIX：TRIX = (EMA3 - EMA3_prev) / EMA3_prev × 100
    5. 计算TRMA：TRMA = EMA(TRIX, M)
    6. 计算TRIX_OSC：TRIX_OSC = (TRIX - TRMA) × 100

    使用场景：
    - TRIX > 0 时，表示多头市场
    - TRIX < 0 时，表示空头市场
    - TRIX上升时，表示上升趋势
    - TRIX下降时，表示下降趋势
    - TRIX上穿TRMA时，为买入信号（金叉）
    - TRIX下穿TRMA时，为卖出信号（死叉）
    - TRIX_OSC > 0 时，表示量价配合良好
    - TRIX_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认12
    - ma_period: TRMA周期，默认20

    输出参数：
    - EMA1: 第一重指数移动平均线
    - EMA2: 第二重指数移动平均线
    - EMA3: 第三重指数移动平均线
    - TRIX: 三重指数平滑移动平均线
    - TRMA: TRMA线
    - TRIX_OSC: TRIX的震荡指标
    - trix_positive: TRIX > 0信号
    - trix_negative: TRIX < 0信号
    - trix_rising: TRIX上升信号
    - trix_falling: TRIX下降信号
    - trix_above_trma: TRIX在TRMA上方信号
    - trix_below_trma: TRIX在TRMA下方信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - TRIXExtended是趋势指标，在单边市中表现最好
    - TRIXExtended在震荡市中会产生频繁的假信号
    - TRIXExtended反应较慢，不适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：TRIX和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的TRIXExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=12, ma_period=20):
        """
        初始化TRIXExtended指标

        Parameters
        ----------
        period : int, default 12
            计算周期
        ma_period : int, default 20
            TRMA周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算TRIXExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含TRIXExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA1'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['EMA2'] = df['EMA1'].ewm(span=self.period, adjust=False).mean()
        df['EMA3'] = df['EMA2'].ewm(span=self.period, adjust=False).mean()
        df['TRIX'] = ((df['EMA3'] - df['EMA3'].shift(1)) / df['EMA3'].shift(1).replace(0, np.nan)) * 100
        df['TRIX'] = df['TRIX'].fillna(0)
        df['TRMA'] = df['TRIX'].ewm(span=self.ma_period, adjust=False).mean()
        df['TRIX_OSC'] = (df['TRIX'] - df['TRMA']) * 100

        df['trix_positive'] = df['TRIX'] > 0
        df['trix_negative'] = df['TRIX'] < 0
        df['trix_rising'] = df['TRIX'] > df['TRIX'].shift(1)
        df['trix_falling'] = df['TRIX'] < df['TRIX'].shift(1)
        df['trix_above_trma'] = df['TRIX'] > df['TRMA']
        df['trix_below_trma'] = df['TRIX'] < df['TRMA']
        df['golden_cross'] = (df['TRIX'] > df['TRMA']) & (df['TRIX'].shift(1) <= df['TRMA'].shift(1))
        df['death_cross'] = (df['TRIX'] < df['TRMA']) & (df['TRIX'].shift(1) >= df['TRMA'].shift(1))

        return df[['EMA1', 'EMA2', 'EMA3', 'TRIX', 'TRMA', 'TRIX_OSC',
                   'trix_positive', 'trix_negative',
                   'trix_rising', 'trix_falling',
                   'trix_above_trma', 'trix_below_trma',
                   'golden_cross', 'death_cross']]
