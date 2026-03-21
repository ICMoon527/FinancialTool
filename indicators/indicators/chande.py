import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Chande(BaseIndicator):
    """
    Chande（钱德动量摆动指标），中文称为钱德动量摆动指标。

    Chande指标是由图莎尔·钱德（Tushar Chande）提出的，是一种动量指标。
    Chande指标通过计算价格的涨跌幅度，来衡量价格的动量变化。

    计算公式：
    1. 计算上涨动量：UpMomentum = Max(Close - Close_prev, 0)
    2. 计算下跌动量：DownMomentum = Max(Close_prev - Close, 0)
    3. 计算上涨动量和：SumUp = Σ(UpMomentum, N)
    4. 计算下跌动量和：SumDown = Σ(DownMomentum, N)
    5. 计算Chande：Chande = 100 × (SumUp - SumDown) / (SumUp + SumDown)

    使用场景：
    - Chande > +50 时，为超买信号，可能回落
    - Chande < -50 时，为超卖信号，可能反弹
    - Chande由下往上穿越-50时，为买入信号
    - Chande由上往下跌破+50时，为卖出信号
    - Chande由负转正时，为买入信号
    - Chande由正转负时，为卖出信号
    - Chande创新高时，为强势信号
    - Chande创新低时，为弱势信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20

    输出参数：
    - UpMomentum: 上涨动量
    - DownMomentum: 下跌动量
    - SumUp: 上涨动量和
    - SumDown: 下跌动量和
    - Chande: 钱德动量摆动指标
    - Chande_MA: Chande的移动平均线
    - chande_overbought: Chande > +50超买信号
    - chande_oversold: Chande < -50超卖信号
    - chande_cross_50_up: Chande上穿-50信号
    - chande_cross_50_down: Chande下穿+50信号
    - chande_positive: Chande > 0信号
    - chande_negative: Chande < 0信号
    - chande_cross_zero_up: Chande上穿0线信号
    - chande_cross_zero_down: Chande下穿0线信号
    - chande_new_high: Chande创新高信号
    - chande_new_low: Chande创新低信号

    注意事项：
    - Chande是动量指标，反应快速
    - Chande在震荡市中会产生频繁的假信号
    - Chande在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 超买超卖：Chande < -50 买入和Chande > +50 卖出更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Chande
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, ma_period=10):
        """
        初始化Chande指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        ma_period : int, default 10
            移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算Chande指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Chande计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        up_momentum = np.where(df['Close'] > close_prev, df['Close'] - close_prev, 0)
        down_momentum = np.where(df['Close'] < close_prev, close_prev - df['Close'], 0)

        df['UpMomentum'] = up_momentum
        df['DownMomentum'] = down_momentum

        sum_up = df['UpMomentum'].rolling(window=self.period).sum()
        sum_down = df['DownMomentum'].rolling(window=self.period).sum()

        df['SumUp'] = sum_up
        df['SumDown'] = sum_down

        denominator = sum_up + sum_down
        df['Chande'] = 100 * (sum_up - sum_down) / denominator.replace(0, np.nan)
        df['Chande_MA'] = df['Chande'].rolling(window=self.ma_period).mean()

        df['chande_overbought'] = df['Chande'] > 50
        df['chande_oversold'] = df['Chande'] < -50
        df['chande_cross_50_up'] = (df['Chande'] > -50) & (df['Chande'].shift(1) <= -50)
        df['chande_cross_50_down'] = (df['Chande'] < 50) & (df['Chande'].shift(1) >= 50)
        df['chande_positive'] = df['Chande'] > 0
        df['chande_negative'] = df['Chande'] < 0
        df['chande_cross_zero_up'] = (df['Chande'] > 0) & (df['Chande'].shift(1) <= 0)
        df['chande_cross_zero_down'] = (df['Chande'] < 0) & (df['Chande'].shift(1) >= 0)
        df['chande_new_high'] = df['Chande'] == df['Chande'].rolling(window=self.ma_period).max()
        df['chande_new_low'] = df['Chande'] == df['Chande'].rolling(window=self.ma_period).min()

        return df[['UpMomentum', 'DownMomentum', 'SumUp', 'SumDown', 'Chande', 'Chande_MA',
                   'chande_overbought', 'chande_oversold', 'chande_cross_50_up', 'chande_cross_50_down',
                   'chande_positive', 'chande_negative', 'chande_cross_zero_up', 'chande_cross_zero_down',
                   'chande_new_high', 'chande_new_low']]
