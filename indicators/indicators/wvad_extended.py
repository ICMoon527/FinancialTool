import pandas as pd
import numpy as np
from ..base import BaseIndicator


class WVADExtended(BaseIndicator):
    """
    WVADExtended（威廉变异离散量扩展），中文称为威廉变异离散量扩展。

    WVADExtended指标是WVAD指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    WVADExtended指标结合了成交量和价格，来反应市场的量价关系。

    计算公式：
    1. 计算WVAD：WVAD = sum((Close - Open) / (High - Low) × Volume, N)
    2. 计算WVAD_MA1：WVAD_MA1 = EMA(WVAD, M)
    3. 计算WVAD_MA2：WVAD_MA2 = EMA(WVAD_MA1, P)
    4. 计算WVAD_OSC：WVAD_OSC = (WVAD_MA1 - WVAD_MA2) × 100

    使用场景：
    - WVAD > 0 时，表示多头市场
    - WVAD < 0 时，表示空头市场
    - WVAD上升时，表示上升趋势
    - WVAD下降时，表示下降趋势
    - WVAD上穿0时，为买入信号
    - WVAD下穿0时，为卖出信号
    - WVAD_OSC > 0 时，表示量价配合良好
    - WVAD_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Open'、'High'、'Low'、'Close'、'Volume'列
    - period: 计算周期，默认24
    - ma_period1: 第一移动平均周期，默认6
    - ma_period2: 第二移动平均周期，默认6

    输出参数：
    - WVAD: 威廉变异离散量
    - WVAD_MA1: WVAD的第一移动平均线
    - WVAD_MA2: WVAD的第二移动平均线
    - WVAD_OSC: WVAD的震荡指标
    - wvad_positive: WVAD > 0信号
    - wvad_negative: WVAD < 0信号
    - wvad_rising: WVAD上升信号
    - wvad_falling: WVAD下降信号
    - wvad_ma1_above_ma2: WVAD_MA1在WVAD_MA2上方信号
    - wvad_ma1_below_ma2: WVAD_MA1在WVAD_MA2下方信号
    - wvad_osc_positive: WVAD_OSC > 0信号
    - wvad_osc_negative: WVAD_OSC < 0信号
    - wvad_cross_up_0: WVAD上穿0信号
    - wvad_cross_down_0: WVAD下穿0信号

    注意事项：
    - WVADExtended是量价指标，反应资金流动
    - WVADExtended在震荡市中会产生频繁的假信号
    - WVADExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：WVAD和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的WVADExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=24, ma_period1=6, ma_period2=6):
        """
        初始化WVADExtended指标

        Parameters
        ----------
        period : int, default 24
            计算周期
        ma_period1 : int, default 6
            第一移动平均周期
        ma_period2 : int, default 6
            第二移动平均周期
        """
        self.period = period
        self.ma_period1 = ma_period1
        self.ma_period2 = ma_period2

    def calculate(self, data):
        """
        计算WVADExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Open'、'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含WVADExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        price_range = df['High'] - df['Low']
        price_range = price_range.replace(0, np.nan)
        wvad_single = ((df['Close'] - df['Open']) / price_range) * df['Volume']
        wvad_single = wvad_single.fillna(0)
        df['WVAD'] = wvad_single.rolling(window=self.period).sum()
        df['WVAD_MA1'] = df['WVAD'].ewm(span=self.ma_period1, adjust=False).mean()
        df['WVAD_MA2'] = df['WVAD_MA1'].ewm(span=self.ma_period2, adjust=False).mean()
        df['WVAD_OSC'] = (df['WVAD_MA1'] - df['WVAD_MA2']) * 100

        df['wvad_positive'] = df['WVAD'] > 0
        df['wvad_negative'] = df['WVAD'] < 0
        df['wvad_rising'] = df['WVAD'] > df['WVAD'].shift(1)
        df['wvad_falling'] = df['WVAD'] < df['WVAD'].shift(1)
        df['wvad_ma1_above_ma2'] = df['WVAD_MA1'] > df['WVAD_MA2']
        df['wvad_ma1_below_ma2'] = df['WVAD_MA1'] < df['WVAD_MA2']
        df['wvad_osc_positive'] = df['WVAD_OSC'] > 0
        df['wvad_osc_negative'] = df['WVAD_OSC'] < 0
        df['wvad_cross_up_0'] = (df['WVAD'] > 0) & (df['WVAD'].shift(1) <= 0)
        df['wvad_cross_down_0'] = (df['WVAD'] < 0) & (df['WVAD'].shift(1) >= 0)

        return df[['WVAD', 'WVAD_MA1', 'WVAD_MA2', 'WVAD_OSC',
                   'wvad_positive', 'wvad_negative',
                   'wvad_rising', 'wvad_falling',
                   'wvad_ma1_above_ma2', 'wvad_ma1_below_ma2',
                   'wvad_osc_positive', 'wvad_osc_negative',
                   'wvad_cross_up_0', 'wvad_cross_down_0']]
