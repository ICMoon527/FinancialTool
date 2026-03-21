import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CCIIExtended(BaseIndicator):
    """
    CCIIExtended（顺势指标扩展），中文称为顺势指标扩展。

    CCIIExtended指标是CCI指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    CCIIExtended指标结合了典型价格和平均偏离，来反应价格的超买超卖状态。

    计算公式：
    1. 计算TypicalPrice：TypicalPrice = (High + Low + Close) / 3
    2. 计算SMA_TP：SMA_TP = SMA(TypicalPrice, N)
    3. 计算MeanDeviation：MeanDeviation = SMA(|TypicalPrice - SMA_TP|, N)
    4. 计算CCI：CCI = (TypicalPrice - SMA_TP) / (0.015 × MeanDeviation)
    5. 计算CCI_MA：CCI_MA = EMA(CCI, M)

    使用场景：
    - CCI > 100 时，表示超买，为卖出信号
    - CCI < -100 时，表示超卖，为买入信号
    - CCI在100到-100之间时，表示正常波动
    - CCI上穿0时，为买入信号
    - CCI下穿0时，为卖出信号
    - CCI上升时，表示上升趋势
    - CCI下降时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认20
    - ma_period: CCI移动平均周期，默认10

    输出参数：
    - TypicalPrice: 典型价格
    - SMA_TP: 典型价格的移动平均线
    - MeanDeviation: 平均偏离
    - CCI: 顺势指标
    - CCI_MA: CCI的移动平均线
    - cci_overbought: CCI超买信号（>100）
    - cci_oversold: CCI超卖信号（<-100）
    - cci_above_0: CCI在0上方信号
    - cci_below_0: CCI在0下方信号
    - cci_above_ma: CCI在MA上方信号
    - cci_below_ma: CCI在MA下方信号
    - cci_rising: CCI上升信号
    - cci_falling: CCI下降信号

    注意事项：
    - CCIIExtended是摆动指标，在震荡市中表现最好
    - CCIIExtended在单边市中会产生钝化现象
    - CCIIExtended反应较快，适合中短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和CCI同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的CCIIExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, ma_period=10):
        """
        初始化CCIIExtended指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        ma_period : int, default 10
            CCI移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算CCIIExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CCIIExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['SMA_TP'] = df['TypicalPrice'].rolling(window=self.period).mean()
        df['MeanDeviation'] = abs(df['TypicalPrice'] - df['SMA_TP']).rolling(window=self.period).mean()
        df['CCI'] = (df['TypicalPrice'] - df['SMA_TP']) / (0.015 * df['MeanDeviation'].replace(0, np.nan))
        df['CCI'] = df['CCI'].fillna(0)
        df['CCI_MA'] = df['CCI'].ewm(span=self.ma_period, adjust=False).mean()

        df['cci_overbought'] = df['CCI'] > 100
        df['cci_oversold'] = df['CCI'] < -100
        df['cci_above_0'] = df['CCI'] > 0
        df['cci_below_0'] = df['CCI'] < 0
        df['cci_above_ma'] = df['CCI'] > df['CCI_MA']
        df['cci_below_ma'] = df['CCI'] < df['CCI_MA']
        df['cci_rising'] = df['CCI'] > df['CCI'].shift(1)
        df['cci_falling'] = df['CCI'] < df['CCI'].shift(1)

        return df[['TypicalPrice', 'SMA_TP', 'MeanDeviation', 'CCI', 'CCI_MA',
                   'cci_overbought', 'cci_oversold',
                   'cci_above_0', 'cci_below_0',
                   'cci_above_ma', 'cci_below_ma',
                   'cci_rising', 'cci_falling']]
