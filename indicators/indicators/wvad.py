import pandas as pd
import numpy as np
from ..base import BaseIndicator


class WVAD(BaseIndicator):
    """
    WVAD（Williams Variable Accumulation Distribution）指标，中文称为威廉变异离散量指标。

    WVAD指标是一种结合了价格和成交量的指标，通过计算开盘价与收盘价的关系和成交量，
    来衡量市场的买卖力量和资金流向。

    计算公式：
    1. 计算A = Close - Open
    2. 计算B = High - Low
    3. 计算V = Volume
    4. 计算WVAD = (A / B) × V
    5. WVAD = Σ(WVAD, N)

    使用场景：
    - WVAD上升时，表示资金流入，为买入信号
    - WVAD下降时，表示资金流出，为卖出信号
    - WVAD创新高时，表示资金持续流入，为买入信号
    - WVAD创新低时，表示资金持续流出，为卖出信号
    - WVAD与价格同步上升时，趋势健康
    - WVAD与价格同步下降时，趋势健康
    - WVAD与价格背离时，可能反转
    - WVAD由负转正时，为买入信号
    - WVAD由正转负时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Open'、'High'、'Low'、'Close'、'Volume'列
    - period: 累加周期，默认6

    输出参数：
    - WVAD: 威廉变异离散量指标
    - WVAD_MA: WVAD的移动平均线
    - wvad_rising: WVAD上升信号
    - wvad_falling: WVAD下降信号
    - wvad_new_high: WVAD创新高信号
    - wvad_new_low: WVAD创新低信号
    - wvad_above_ma: WVAD在MA上方信号
    - wvad_below_ma: WVAD在MA下方信号
    - wvad_positive: WVAD > 0信号
    - wvad_negative: WVAD < 0信号
    - wvad_cross_zero_up: WVAD上穿0线信号
    - wvad_cross_zero_down: WVAD下穿0线信号

    注意事项：
    - WVAD是量价指标，反应资金流向
    - WVAD在震荡市中会产生频繁的假信号
    - WVAD在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：WVAD和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的WVAD
    4. 背离判断：注意WVAD与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=6, ma_period=6):
        """
        初始化WVAD指标

        Parameters
        ----------
        period : int, default 6
            累加周期
        ma_period : int, default 6
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算WVAD指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Open'、'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含WVAD计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        a = df['Close'] - df['Open']
        b = df['High'] - df['Low']
        wvad_daily = (a / b.replace(0, np.nan)) * df['Volume']
        df['WVAD'] = wvad_daily.rolling(window=self.period).sum()

        df['WVAD_MA'] = df['WVAD'].rolling(window=self.ma_period).mean()

        df['wvad_rising'] = df['WVAD'] > df['WVAD'].shift(1)
        df['wvad_falling'] = df['WVAD'] < df['WVAD'].shift(1)
        df['wvad_new_high'] = df['WVAD'] == df['WVAD'].rolling(window=self.ma_period).max()
        df['wvad_new_low'] = df['WVAD'] == df['WVAD'].rolling(window=self.ma_period).min()
        df['wvad_above_ma'] = df['WVAD'] > df['WVAD_MA']
        df['wvad_below_ma'] = df['WVAD'] < df['WVAD_MA']
        df['wvad_positive'] = df['WVAD'] > 0
        df['wvad_negative'] = df['WVAD'] < 0
        df['wvad_cross_zero_up'] = (df['WVAD'] > 0) & (df['WVAD'].shift(1) <= 0)
        df['wvad_cross_zero_down'] = (df['WVAD'] < 0) & (df['WVAD'].shift(1) >= 0)

        return df[['WVAD', 'WVAD_MA', 'wvad_rising', 'wvad_falling', 
                   'wvad_new_high', 'wvad_new_low', 'wvad_above_ma', 'wvad_below_ma',
                   'wvad_positive', 'wvad_negative', 'wvad_cross_zero_up', 'wvad_cross_zero_down']]
