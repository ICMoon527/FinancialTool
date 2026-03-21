import pandas as pd
import numpy as np
from ..base import BaseIndicator


class UDLExtended(BaseIndicator):
    """
    UDLExtended（引力线扩展），中文称为引力线扩展。

    UDLExtended指标是UDL指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    UDLExtended指标结合了多个移动平均线，来反应价格的趋势。

    计算公式：
    1. 计算MA1：MA1 = EMA(Close, N1)
    2. 计算MA2：MA2 = EMA(Close, N2)
    3. 计算MA3：MA3 = EMA(Close, N3)
    4. 计算MA4：MA4 = EMA(Close, N4)
    5. 计算UDL：UDL = (MA1 + MA2 + MA3 + MA4) / 4
    6. 计算UDL_MA：UDL_MA = EMA(UDL, P)
    7. 计算UDL_OSC：UDL_OSC = (UDL - UDL_MA) × 100

    使用场景：
    - 价格 > UDL 时，表示上升趋势
    - 价格 < UDL 时，表示下降趋势
    - UDL上升时，表示上升趋势
    - UDL下降时，表示下降趋势
    - UDL上穿价格时，为买入信号
    - UDL下穿价格时，为卖出信号
    - UDL_OSC > 0 时，表示量价配合良好
    - UDL_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period1: 第一周期，默认3
    - period2: 第二周期，默认5
    - period3: 第三周期，默认10
    - period4: 第四周期，默认20
    - ma_period: UDL移动平均周期，默认6

    输出参数：
    - MA1: 第一移动平均线
    - MA2: 第二移动平均线
    - MA3: 第三移动平均线
    - MA4: 第四移动平均线
    - UDL: 引力线
    - UDL_MA: 引力线移动平均
    - UDL_OSC: 引力线震荡指标
    - price_above_udl: 价格在UDL上方信号
    - price_below_udl: 价格在UDL下方信号
    - udl_rising: UDL上升信号
    - udl_falling: UDL下降信号
    - udl_above_ma: UDL在MA上方信号
    - udl_below_ma: UDL在MA下方信号

    注意事项：
    - UDLExtended是趋势指标，在单边市中表现最好
    - UDLExtended在震荡市中会产生频繁的假信号
    - UDLExtended反应较慢，不适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：UDL和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的UDLExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=3, period2=5, period3=10, period4=20, ma_period=6):
        """
        初始化UDLExtended指标

        Parameters
        ----------
        period1 : int, default 3
            第一周期
        period2 : int, default 5
            第二周期
        period3 : int, default 10
            第三周期
        period4 : int, default 20
            第四周期
        ma_period : int, default 6
            UDL移动平均周期
        """
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.period4 = period4
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算UDLExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含UDLExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA1'] = df['Close'].ewm(span=self.period1, adjust=False).mean()
        df['MA2'] = df['Close'].ewm(span=self.period2, adjust=False).mean()
        df['MA3'] = df['Close'].ewm(span=self.period3, adjust=False).mean()
        df['MA4'] = df['Close'].ewm(span=self.period4, adjust=False).mean()
        df['UDL'] = (df['MA1'] + df['MA2'] + df['MA3'] + df['MA4']) / 4
        df['UDL_MA'] = df['UDL'].ewm(span=self.ma_period, adjust=False).mean()
        df['UDL_OSC'] = (df['UDL'] - df['UDL_MA']) * 100

        df['price_above_udl'] = df['Close'] > df['UDL']
        df['price_below_udl'] = df['Close'] < df['UDL']
        df['udl_rising'] = df['UDL'] > df['UDL'].shift(1)
        df['udl_falling'] = df['UDL'] < df['UDL'].shift(1)
        df['udl_above_ma'] = df['UDL'] > df['UDL_MA']
        df['udl_below_ma'] = df['UDL'] < df['UDL_MA']

        return df[['MA1', 'MA2', 'MA3', 'MA4', 'UDL', 'UDL_MA', 'UDL_OSC',
                   'price_above_udl', 'price_below_udl',
                   'udl_rising', 'udl_falling',
                   'udl_above_ma', 'udl_below_ma']]
