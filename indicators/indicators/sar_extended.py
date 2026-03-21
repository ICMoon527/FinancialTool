import pandas as pd
import numpy as np
from ..base import BaseIndicator


class SARExtended(BaseIndicator):
    """
    SARExtended（抛物线转向扩展），中文称为抛物线转向扩展。

    SARExtended指标是SAR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    SARExtended指标结合了止损和反转点，来判断价格的趋势。

    计算公式：
    1. 初始化SAR：SAR_prev = Low[0] if uptrend else High[0]
    2. 计算EP（极值点）：EP = max(High) if uptrend else min(Low)
    3. 计算AF（加速因子）：AF = min(AF + 0.02, 0.2)
    4. 计算SAR：SAR = SAR_prev + AF × (EP - SAR_prev)
    5. 反转条件：如果uptrend且Low < SAR，则转为downtrend；如果downtrend且High > SAR，则转为uptrend

    使用场景：
    - 价格在SAR上方时，表示上升趋势
    - 价格在SAR下方时，表示下降趋势
    - SAR向上移动时，表示上升趋势
    - SAR向下移动时，表示下降趋势
    - 价格上穿SAR时，为买入信号
    - 价格下穿SAR时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - af_start: 初始加速因子，默认0.02
    - af_max: 最大加速因子，默认0.2
    - af_step: 加速因子步长，默认0.02

    输出参数：
    - SAR: 抛物线转向
    - EP: 极值点
    - AF: 加速因子
    - Trend: 趋势方向（1=上升，-1=下降）
    - price_above_sar: 价格在SAR上方信号
    - price_below_sar: 价格在SAR下方信号
    - sar_rising: SAR上升信号
    - sar_falling: SAR下降信号
    - uptrend: 上升趋势信号
    - downtrend: 下降趋势信号
    - trend_reversal_up: 趋势反转向上信号
    - trend_reversal_down: 趋势反转向下信号

    注意事项：
    - SARExtended是趋势跟踪指标，在单边市中表现最好
    - SARExtended在震荡市中会产生频繁的假信号
    - SARExtended反应较快，适合趋势跟踪
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和SAR同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的SARExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：SAR本身就是止损位
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, af_start=0.02, af_max=0.2, af_step=0.02):
        """
        初始化SARExtended指标

        Parameters
        ----------
        af_start : float, default 0.02
            初始加速因子
        af_max : float, default 0.2
            最大加速因子
        af_step : float, default 0.02
            加速因子步长
        """
        self.af_start = af_start
        self.af_max = af_max
        self.af_step = af_step

    def calculate(self, data):
        """
        计算SARExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含SARExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        sar = pd.Series(index=df.index, dtype='float64')
        ep = pd.Series(index=df.index, dtype='float64')
        af = pd.Series(index=df.index, dtype='float64')
        trend = pd.Series(index=df.index, dtype='int64')

        is_uptrend = True
        sar[0] = df['Low'][0]
        ep[0] = df['High'][0]
        af[0] = self.af_start
        trend[0] = 1

        for i in range(1, len(df)):
            if is_uptrend:
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = min(sar[i], df['Low'][i-1], df['Low'][i] if i-2 >= 0 else df['Low'][i-1])
                ep[i] = max(ep[i-1], df['High'][i])
                af[i] = min(af[i-1] + self.af_step, self.af_max) if df['High'][i] > ep[i-1] else af[i-1]
                
                if df['Low'][i] < sar[i]:
                    is_uptrend = False
                    sar[i] = ep[i]
                    ep[i] = df['Low'][i]
                    af[i] = self.af_start
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = max(sar[i], df['High'][i-1], df['High'][i] if i-2 >= 0 else df['High'][i-1])
                ep[i] = min(ep[i-1], df['Low'][i])
                af[i] = min(af[i-1] + self.af_step, self.af_max) if df['Low'][i] < ep[i-1] else af[i-1]
                
                if df['High'][i] > sar[i]:
                    is_uptrend = True
                    sar[i] = ep[i]
                    ep[i] = df['High'][i]
                    af[i] = self.af_start
                    trend[i] = 1
                else:
                    trend[i] = -1

        df['SAR'] = sar
        df['EP'] = ep
        df['AF'] = af
        df['Trend'] = trend

        df['price_above_sar'] = df['Close'] > df['SAR']
        df['price_below_sar'] = df['Close'] < df['SAR']
        df['sar_rising'] = df['SAR'] > df['SAR'].shift(1)
        df['sar_falling'] = df['SAR'] < df['SAR'].shift(1)
        df['uptrend'] = df['Trend'] == 1
        df['downtrend'] = df['Trend'] == -1
        df['trend_reversal_up'] = (df['Trend'] == 1) & (df['Trend'].shift(1) == -1)
        df['trend_reversal_down'] = (df['Trend'] == -1) & (df['Trend'].shift(1) == 1)

        return df[['SAR', 'EP', 'AF', 'Trend',
                   'price_above_sar', 'price_below_sar',
                   'sar_rising', 'sar_falling',
                   'uptrend', 'downtrend',
                   'trend_reversal_up', 'trend_reversal_down']]
