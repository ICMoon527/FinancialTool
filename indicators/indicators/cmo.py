import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CMO(BaseIndicator):
    """
    CMO（Chande Momentum Oscillator）指标，中文称为钱德动量震荡指标。

    CMO指标是由图莎尔·钱德（Tushar Chande）提出的，是一种动量指标。
    CMO指标通过计算价格的涨跌幅度，来衡量价格的动量变化。

    计算公式：
    1. 计算上涨变化：UpChange = Max(Close - Close_prev, 0)
    2. 计算下跌变化：DownChange = Max(Close_prev - Close, 0)
    3. 计算上涨变化和：SumUp = Σ(UpChange, N)
    4. 计算下跌变化和：SumDown = Σ(DownChange, N)
    5. 计算CMO：CMO = 100 × (SumUp - SumDown) / (SumUp + SumDown)

    使用场景：
    - CMO > +50 时，为超买信号，可能回落
    - CMO < -50 时，为超卖信号，可能反弹
    - CMO由下往上穿越-50时，为买入信号
    - CMO由上往下跌破+50时，为卖出信号
    - CMO由负转正时，为买入信号
    - CMO由正转负时，为卖出信号
    - CMO创新高时，为强势信号
    - CMO创新低时，为弱势信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20
    - sma_period: SMA周期，默认10

    输出参数：
    - UpChange: 上涨变化
    - DownChange: 下跌变化
    - SumUp: 上涨变化和
    - SumDown: 下跌变化和
    - CMO: 钱德动量震荡指标
    - CMO_SMA: CMO的简单移动平均线
    - cmo_overbought: CMO > +50超买信号
    - cmo_oversold: CMO < -50超卖信号
    - cmo_cross_50_up: CMO上穿-50信号
    - cmo_cross_50_down: CMO下穿+50信号
    - cmo_positive: CMO > 0信号
    - cmo_negative: CMO < 0信号
    - cmo_cross_zero_up: CMO上穿0线信号
    - cmo_cross_zero_down: CMO下穿0线信号
    - cmo_new_high: CMO创新高信号
    - cmo_new_low: CMO创新低信号
    - cmo_above_sma: CMO在SMA上方信号
    - cmo_below_sma: CMO在SMA下方信号

    注意事项：
    - CMO是动量指标，反应快速
    - CMO在震荡市中会产生频繁的假信号
    - CMO在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 超买超卖：CMO < -50 买入和CMO > +50 卖出更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的CMO
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, sma_period=10):
        """
        初始化CMO指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        sma_period : int, default 10
            SMA周期
        """
        self.period = period
        self.sma_period = sma_period

    def calculate(self, data):
        """
        计算CMO指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CMO计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)
        up_change = np.where(df['Close'] > close_prev, df['Close'] - close_prev, 0)
        down_change = np.where(df['Close'] < close_prev, close_prev - df['Close'], 0)

        df['UpChange'] = up_change
        df['DownChange'] = down_change

        sum_up = df['UpChange'].rolling(window=self.period).sum()
        sum_down = df['DownChange'].rolling(window=self.period).sum()

        df['SumUp'] = sum_up
        df['SumDown'] = sum_down

        denominator = sum_up + sum_down
        df['CMO'] = 100 * (sum_up - sum_down) / denominator.replace(0, np.nan)
        df['CMO_SMA'] = df['CMO'].rolling(window=self.sma_period).mean()

        df['cmo_overbought'] = df['CMO'] > 50
        df['cmo_oversold'] = df['CMO'] < -50
        df['cmo_cross_50_up'] = (df['CMO'] > -50) & (df['CMO'].shift(1) <= -50)
        df['cmo_cross_50_down'] = (df['CMO'] < 50) & (df['CMO'].shift(1) >= 50)
        df['cmo_positive'] = df['CMO'] > 0
        df['cmo_negative'] = df['CMO'] < 0
        df['cmo_cross_zero_up'] = (df['CMO'] > 0) & (df['CMO'].shift(1) <= 0)
        df['cmo_cross_zero_down'] = (df['CMO'] < 0) & (df['CMO'].shift(1) >= 0)
        df['cmo_new_high'] = df['CMO'] == df['CMO'].rolling(window=self.sma_period).max()
        df['cmo_new_low'] = df['CMO'] == df['CMO'].rolling(window=self.sma_period).min()
        df['cmo_above_sma'] = df['CMO'] > df['CMO_SMA']
        df['cmo_below_sma'] = df['CMO'] < df['CMO_SMA']

        return df[['UpChange', 'DownChange', 'SumUp', 'SumDown', 'CMO', 'CMO_SMA',
                   'cmo_overbought', 'cmo_oversold', 'cmo_cross_50_up', 'cmo_cross_50_down',
                   'cmo_positive', 'cmo_negative', 'cmo_cross_zero_up', 'cmo_cross_zero_down',
                   'cmo_new_high', 'cmo_new_low', 'cmo_above_sma', 'cmo_below_sma']]
