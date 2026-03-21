import pandas as pd
import numpy as np
from ..base import BaseIndicator


class UDL(BaseIndicator):
    """
    UDL（Ultimate Oscillator）指标，中文称为终极震荡指标。

    UDL指标由拉里·威廉斯（Larry Williams）提出，是一种结合了多个时间周期的震荡指标。
    UDL指标通过计算三个不同时间周期的真实波幅，来衡量价格的超买超卖状态。

    计算公式：
    1. 计算BP（买入压力）：BP = Close - Min(Low, Close_prev)
    2. 计算TR（真实波幅）：TR = Max(High, Close_prev) - Min(Low, Close_prev)
    3. 计算平均7日BP：Avg7 = Σ(BP, 7) / Σ(TR, 7)
    4. 计算平均14日BP：Avg14 = Σ(BP, 14) / Σ(TR, 14)
    5. 计算平均28日BP：Avg28 = Σ(BP, 28) / Σ(TR, 28)
    6. 计算UDL：UDL = 100 × (4 × Avg7 + 2 × Avg14 + Avg28) / 7

    使用场景：
    - UDL在0-100之间波动
    - UDL > 70 时，处于超买区间，可能回落，为卖出信号
    - UDL < 30 时，处于超卖区间，可能反弹，为买入信号
    - UDL由下往上穿越30时，为买入信号
    - UDL由上往下跌破70时，为卖出信号
    - UDL在50以上时，为多头市场
    - UDL在50以下时，为空头市场

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period1: 第一周期，默认7
    - period2: 第二周期，默认14
    - period3: 第三周期，默认28

    输出参数：
    - UDL: 终极震荡指标
    - overbought: 超买信号（UDL > 70）
    - oversold: 超卖信号（UDL < 30）
    - cross_30_up: 上穿30信号
    - cross_70_down: 下穿70信号
    - bullish: UDL > 50信号
    - bearish: UDL < 50信号

    注意事项：
    - UDL是震荡指标，在震荡市中表现较好
    - UDL在单边市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - UDL反应较慢，不适合短线操作

    最佳实践建议：
    1. 超买超卖：在超卖区买入和超买区卖出更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的UDL
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=7, period2=14, period3=28):
        """
        初始化UDL指标

        Parameters
        ----------
        period1 : int, default 7
            第一周期
        period2 : int, default 14
            第二周期
        period3 : int, default 28
            第三周期
        """
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3

    def calculate(self, data):
        """
        计算UDL指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含UDL计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)

        bp = df['Close'] - pd.concat([df['Low'], close_prev], axis=1).min(axis=1)
        tr = pd.concat([df['High'], close_prev], axis=1).max(axis=1) - pd.concat([df['Low'], close_prev], axis=1).min(axis=1)

        bp_sum1 = bp.rolling(window=self.period1).sum()
        tr_sum1 = tr.rolling(window=self.period1).sum()
        avg1 = bp_sum1 / tr_sum1.replace(0, np.nan)

        bp_sum2 = bp.rolling(window=self.period2).sum()
        tr_sum2 = tr.rolling(window=self.period2).sum()
        avg2 = bp_sum2 / tr_sum2.replace(0, np.nan)

        bp_sum3 = bp.rolling(window=self.period3).sum()
        tr_sum3 = tr.rolling(window=self.period3).sum()
        avg3 = bp_sum3 / tr_sum3.replace(0, np.nan)

        df['UDL'] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        df['overbought'] = df['UDL'] > 70
        df['oversold'] = df['UDL'] < 30
        df['cross_30_up'] = (df['UDL'] > 30) & (df['UDL'].shift(1) <= 30)
        df['cross_70_down'] = (df['UDL'] < 70) & (df['UDL'].shift(1) >= 70)
        df['bullish'] = df['UDL'] > 50
        df['bearish'] = df['UDL'] < 50

        return df[['UDL', 'overbought', 'oversold', 'cross_30_up', 'cross_70_down', 
                   'bullish', 'bearish']]
