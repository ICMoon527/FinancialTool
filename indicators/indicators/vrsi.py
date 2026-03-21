import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VRSI(BaseIndicator):
    """
    VRSI（成交量相对强弱指标），中文称为成交量相对强弱指标。

    VRSI指标是基于成交量的RSI指标，通过计算成交量的变化，来判断市场的超买超卖状态。
    VRSI指标通过成交量的相对强弱，来反应市场的情绪变化。

    计算公式：
    1. 计算VolumeChange：VolumeChange = Volume - Volume_prev
    2. 计算Gain：Gain = max(VolumeChange, 0)
    3. 计算Loss：Loss = -min(VolumeChange, 0)
    4. 计算AvgGain：AvgGain = EMA(Gain, N)
    5. 计算AvgLoss：AvgLoss = EMA(Loss, N)
    6. 计算RS：RS = AvgGain / AvgLoss
    7. 计算VRSI：VRSI = 100 - 100 / (1 + RS)

    使用场景：
    - VRSI > 70 时，为超买信号，可能回落
    - VRSI < 30 时，为超卖信号，可能反弹
    - VRSI上升时，表示成交量增加，为买入信号
    - VRSI下降时，表示成交量减少，为卖出信号
    - VRSI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Volume'列
    - period: 计算周期，默认14

    输出参数：
    - VolumeChange: 成交量变化
    - Gain: 上涨
    - Loss: 下跌
    - AvgGain: 平均上涨
    - AvgLoss: 平均下跌
    - RS: 相对强弱
    - VRSI: 成交量相对强弱指标
    - VRSI_MA: VRSI的移动平均线
    - vrsi_overbought: VRSI > 70信号
    - vrsi_oversold: VRSI < 30信号
    - vrsi_rising: VRSI上升信号
    - vrsi_falling: VRSI下降信号
    - vrsi_above_ma: VRSI在MA上方信号
    - vrsi_below_ma: VRSI在MA下方信号

    注意事项：
    - VRSI是成交量震荡指标，在震荡市中表现最好
    - VRSI在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VRSI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的VRSI
    4. 量价配合：结合价格变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14, ma_period=10):
        """
        初始化VRSI指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算VRSI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VRSI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        volume_prev = df['Volume'].shift(1)
        df['VolumeChange'] = df['Volume'] - volume_prev
        df['Gain'] = df['VolumeChange'].where(df['VolumeChange'] > 0, 0)
        df['Loss'] = -df['VolumeChange'].where(df['VolumeChange'] < 0, 0)
        df['AvgGain'] = df['Gain'].ewm(span=self.period, adjust=False).mean()
        df['AvgLoss'] = df['Loss'].ewm(span=self.period, adjust=False).mean()
        df['RS'] = df['AvgGain'] / df['AvgLoss']
        df['VRSI'] = 100 - (100 / (1 + df['RS']))
        df['VRSI_MA'] = df['VRSI'].rolling(window=self.ma_period).mean()

        df['vrsi_overbought'] = df['VRSI'] > 70
        df['vrsi_oversold'] = df['VRSI'] < 30
        df['vrsi_rising'] = df['VRSI'] > df['VRSI'].shift(1)
        df['vrsi_falling'] = df['VRSI'] < df['VRSI'].shift(1)
        df['vrsi_above_ma'] = df['VRSI'] > df['VRSI_MA']
        df['vrsi_below_ma'] = df['VRSI'] < df['VRSI_MA']

        return df[['VolumeChange', 'Gain', 'Loss', 'AvgGain', 'AvgLoss', 'RS', 'VRSI', 'VRSI_MA',
                   'vrsi_overbought', 'vrsi_oversold',
                   'vrsi_rising', 'vrsi_falling',
                   'vrsi_above_ma', 'vrsi_below_ma']]
