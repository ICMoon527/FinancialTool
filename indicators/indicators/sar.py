import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SAR(BaseIndicator):
    """
    SAR（Stop and Reverse）指标，中文称为抛物线转向指标或停损点转向指标。

    SAR指标由威尔斯·怀尔德（Welles Wilder）于1978年提出，是一种趋势跟踪指标，
    同时也用于设置止损位。SAR指标会在价格下方或上方形成一系列的点，当价格突破这些点时，
    表示趋势可能反转。

    计算公式：
    1. 初始阶段：
       - 假设为上升趋势，第一个SAR点为前N日最低价中的最低值
       - 加速因子（AF）初始为0.02
       - 极值点（EP）为前N日最高价中的最高值
    2. 上升趋势中：
       - SAR = 前一日SAR + AF × (前一日EP - 前一日SAR)
       - 如果当日最高价 > 前一日EP，则EP = 当日最高价，AF = min(AF + 0.02, 0.2)
       - SAR不能高于前两日最低价，否则取前两日最低价
    3. 下降趋势中：
       - SAR = 前一日SAR - AF × (前一日SAR - 前一日EP)
       - 如果当日最低价 < 前一日EP，则EP = 当日最低价，AF = min(AF + 0.02, 0.2)
       - SAR不能低于前两日最高价，否则取前两日最高价
    4. 转向条件：
       - 上升趋势中，当价格跌破SAR时，转为下降趋势
       - 下降趋势中，当价格突破SAR时，转为上升趋势

    使用场景：
    - SAR点在价格下方时，为上升趋势，持有或买入
    - SAR点在价格上方时，为下降趋势，空仓或卖出
    - 当价格从下往上突破SAR时，为买入信号
    - 当价格从上往下跌破SAR时，为卖出信号
    - SAR点可以作为动态止损位
    - SAR点密集时，表示趋势可能反转
    - SAR点稀疏时，表示趋势稳定

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - af_start: 初始加速因子，默认0.02
    - af_step: 加速因子步长，默认0.02
    - af_max: 最大加速因子，默认0.2

    输出参数：
    - SAR: 抛物线转向指标值
    - EP: 极值点
    - AF: 加速因子
    - trend: 趋势信号（1表示上升，-1表示下降）
    - buy_signal: 买入信号
    - sell_signal: 卖出信号

    注意事项：
    - SAR在震荡市中会产生频繁的假信号
    - SAR在单边市中表现最好
    - SAR不适合在横盘整理时使用
    - 建议结合其他指标（如MACD、RSI）一起使用
    - SAR的参数可以根据市场特性调整

    最佳实践建议：
    1. 趋势过滤：先判断大趋势，只在趋势方向上操作
    2. 组合使用：与MA、MACD等趋势指标配合使用
    3. 震荡规避：在横盘整理时避免使用SAR
    4. 止损设置：SAR本身就是很好的止损位
    5. 确认信号：等待SAR转向信号确认后再操作
    6. 多周期使用：同时使用日线和周线的SAR
    7. 量价配合：结合成交量变化验证信号
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, af_start=0.02, af_step=0.02, af_max=0.2):
        """
        初始化SAR指标

        Parameters
        ----------
        af_start : float, default 0.02
            初始加速因子
        af_step : float, default 0.02
            加速因子步长
        af_max : float, default 0.2
            最大加速因子
        """
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max

    def calculate(self, data):
        """
        计算SAR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SAR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        n = len(df)
        sar = np.zeros(n)
        ep = np.zeros(n)
        af = np.zeros(n)
        trend = np.zeros(n)

        if n < 2:
            return pd.DataFrame({'SAR': [np.nan], 'EP': [np.nan], 'AF': [np.nan], 
                                'trend': [np.nan], 'buy_signal': [False], 'sell_signal': [False]}, 
                               index=df.index)

        if df['Close'].iloc[1] > df['Close'].iloc[0]:
            trend[0] = 1
            sar[0] = df['Low'].iloc[0]
            ep[0] = df['High'].iloc[0]
        else:
            trend[0] = -1
            sar[0] = df['High'].iloc[0]
            ep[0] = df['Low'].iloc[0]

        af[0] = self.af_start

        for i in range(1, n):
            if trend[i-1] == 1:
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = min(sar[i], df['Low'].iloc[i-1])
                if i > 1:
                    sar[i] = min(sar[i], df['Low'].iloc[i-2])

                if df['High'].iloc[i] > ep[i-1]:
                    ep[i] = df['High'].iloc[i]
                    af[i] = min(af[i-1] + self.af_step, self.af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

                if df['Low'].iloc[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = df['Low'].iloc[i]
                    af[i] = self.af_start
                else:
                    trend[i] = 1
            else:
                sar[i] = sar[i-1] - af[i-1] * (sar[i-1] - ep[i-1])
                sar[i] = max(sar[i], df['High'].iloc[i-1])
                if i > 1:
                    sar[i] = max(sar[i], df['High'].iloc[i-2])

                if df['Low'].iloc[i] < ep[i-1]:
                    ep[i] = df['Low'].iloc[i]
                    af[i] = min(af[i-1] + self.af_step, self.af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

                if df['High'].iloc[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = df['High'].iloc[i]
                    af[i] = self.af_start
                else:
                    trend[i] = -1

        df['SAR'] = sar
        df['EP'] = ep
        df['AF'] = af
        df['trend'] = trend

        df['buy_signal'] = (trend == 1) & (np.roll(trend, 1) == -1)
        df['sell_signal'] = (trend == -1) & (np.roll(trend, 1) == 1)
        df['buy_signal'].iloc[0] = False
        df['sell_signal'].iloc[0] = False

        return df[['SAR', 'EP', 'AF', 'trend', 'buy_signal', 'sell_signal']]
