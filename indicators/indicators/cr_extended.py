import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CRExtended(BaseIndicator):
    """
    CRExtended（价格动量指标扩展），中文称为价格动量指标扩展。

    CRExtended指标是CR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    CRExtended指标结合了最高价、最低价和中间价，来反应价格的动量。

    计算公式：
    1. 计算MID：MID = (High + Low + Close) / 3
    2. 计算UP：UP = max(High - MID_prev, 0)
    3. 计算DN：DN = max(MID_prev - Low, 0)
    4. 计算CR：CR = sum(UP, N) / sum(DN, N) × 100
    5. 计算MA1：MA1 = EMA(CR, M1)
    6. 计算MA2：MA2 = EMA(CR, M2)
    7. 计算CR_OSC：CR_OSC = (MA1 - MA2) × 100

    使用场景：
    - CR > 300 时，表示超买，为卖出信号
    - CR < 40 时，表示超卖，为买入信号
    - CR > 200 时，表示强势
    - CR < 100 时，表示弱势
    - CR上升时，表示上升趋势
    - CR下降时，表示下降趋势
    - CR上穿MA1时，为买入信号
    - CR下穿MA1时，为卖出信号
    - CR_OSC > 0 时，表示量价配合良好
    - CR_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认26
    - ma_period1: 第一移动平均周期，默认10
    - ma_period2: 第二移动平均周期，默认20

    输出参数：
    - MID: 中间价
    - UP: 上涨动力
    - DN: 下跌动力
    - CR: 价格动量指标
    - MA1: CR的第一移动平均线
    - MA2: CR的第二移动平均线
    - CR_OSC: CR的震荡指标
    - cr_overbought: CR > 300超买信号
    - cr_oversold: CR < 40超卖信号
    - cr_strong: CR > 200强势信号
    - cr_weak: CR < 100弱势信号
    - cr_rising: CR上升信号
    - cr_falling: CR下降信号
    - cr_above_ma1: CR在MA1上方信号
    - cr_below_ma1: CR在MA1下方信号
    - cr_osc_positive: CR_OSC > 0信号
    - cr_osc_negative: CR_OSC < 0信号

    注意事项：
    - CRExtended是动量指标，反应价格动能
    - CRExtended在震荡市中会产生频繁的假信号
    - CRExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：CR和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的CRExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=26, ma_period1=10, ma_period2=20):
        """
        初始化CRExtended指标

        Parameters
        ----------
        period : int, default 26
            计算周期
        ma_period1 : int, default 10
            第一移动平均周期
        ma_period2 : int, default 20
            第二移动平均周期
        """
        self.period = period
        self.ma_period1 = ma_period1
        self.ma_period2 = ma_period2

    def calculate(self, data):
        """
        计算CRExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CRExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MID'] = (df['High'] + df['Low'] + df['Close']) / 3
        prev_mid = df['MID'].shift(1)
        df['UP'] = np.maximum(df['High'] - prev_mid, 0)
        df['DN'] = np.maximum(prev_mid - df['Low'], 0)
        sum_up = df['UP'].rolling(window=self.period).sum()
        sum_dn = df['DN'].rolling(window=self.period).sum()
        df['CR'] = (sum_up / sum_dn.replace(0, np.nan)) * 100
        df['CR'] = df['CR'].fillna(100)
        df['MA1'] = df['CR'].ewm(span=self.ma_period1, adjust=False).mean()
        df['MA2'] = df['CR'].ewm(span=self.ma_period2, adjust=False).mean()
        df['CR_OSC'] = (df['MA1'] - df['MA2']) * 100

        df['cr_overbought'] = df['CR'] > 300
        df['cr_oversold'] = df['CR'] < 40
        df['cr_strong'] = df['CR'] > 200
        df['cr_weak'] = df['CR'] < 100
        df['cr_rising'] = df['CR'] > df['CR'].shift(1)
        df['cr_falling'] = df['CR'] < df['CR'].shift(1)
        df['cr_above_ma1'] = df['CR'] > df['MA1']
        df['cr_below_ma1'] = df['CR'] < df['MA1']
        df['cr_osc_positive'] = df['CR_OSC'] > 0
        df['cr_osc_negative'] = df['CR_OSC'] < 0

        return df[['MID', 'UP', 'DN', 'CR', 'MA1', 'MA2', 'CR_OSC',
                   'cr_overbought', 'cr_oversold',
                   'cr_strong', 'cr_weak',
                   'cr_rising', 'cr_falling',
                   'cr_above_ma1', 'cr_below_ma1',
                   'cr_osc_positive', 'cr_osc_negative']]
