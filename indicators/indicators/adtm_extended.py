import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ADTMExtended(BaseIndicator):
    """
    ADTMExtended（动态买卖气指标扩展），中文称为动态买卖气指标扩展。

    ADTMExtended指标是ADTM指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    ADTMExtended指标结合了最高价和最低价，来反应市场的买卖气。

    计算公式：
    1. 计算DTM：如果High > High_prev，则DTM = High - High_prev，否则DTM = 0
    2. 计算DBM：如果Low < Low_prev，则DBM = Low_prev - Low，否则DBM = 0
    3. 计算STM：STM = sum(DTM, N)
    4. 计算SBM：SBM = sum(DBM, N)
    5. 计算ADTM：ADTM = (STM - SBM) / max(STM, SBM)
    6. 计算ADTM_MA：ADTM_MA = EMA(ADTM, M)
    7. 计算ADTM_OSC：ADTM_OSC = (ADTM - ADTM_MA) × 100

    使用场景：
    - ADTM > 0 时，表示多头市场
    - ADTM < 0 时，表示空头市场
    - ADTM上升时，表示上升趋势
    - ADTM下降时，表示下降趋势
    - ADTM上穿ADTM_MA时，为买入信号（金叉）
    - ADTM下穿ADTM_MA时，为卖出信号（死叉）
    - ADTM_OSC > 0 时，表示量价配合良好
    - ADTM_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'列
    - period: 计算周期，默认23
    - ma_period: ADTM移动平均周期，默认8

    输出参数：
    - DTM: 动态买盘
    - DBM: 动态卖盘
    - STM: 买盘总和
    - SBM: 卖盘总和
    - ADTM: 动态买卖气指标
    - ADTM_MA: ADTM的移动平均线
    - ADTM_OSC: ADTM的震荡指标
    - adtm_positive: ADTM > 0信号
    - adtm_negative: ADTM < 0信号
    - adtm_rising: ADTM上升信号
    - adtm_falling: ADTM下降信号
    - adtm_above_ma: ADTM在MA上方信号
    - adtm_below_ma: ADTM在MA下方信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - ADTMExtended是量价指标，反应资金流动
    - ADTMExtended在震荡市中会产生频繁的假信号
    - ADTMExtended在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ADTM和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ADTMExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=23, ma_period=8):
        """
        初始化ADTMExtended指标

        Parameters
        ----------
        period : int, default 23
            计算周期
        ma_period : int, default 8
            ADTM移动平均周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算ADTMExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ADTMExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        df['DTM'] = np.where(df['High'] > prev_high, df['High'] - prev_high, 0)
        df['DBM'] = np.where(df['Low'] < prev_low, prev_low - df['Low'], 0)
        df['STM'] = df['DTM'].rolling(window=self.period).sum()
        df['SBM'] = df['DBM'].rolling(window=self.period).sum()
        max_stm_sbm = pd.concat([df['STM'], df['SBM']], axis=1).max(axis=1)
        df['ADTM'] = (df['STM'] - df['SBM']) / max_stm_sbm.replace(0, np.nan)
        df['ADTM'] = df['ADTM'].fillna(0)
        df['ADTM_MA'] = df['ADTM'].ewm(span=self.ma_period, adjust=False).mean()
        df['ADTM_OSC'] = (df['ADTM'] - df['ADTM_MA']) * 100

        df['adtm_positive'] = df['ADTM'] > 0
        df['adtm_negative'] = df['ADTM'] < 0
        df['adtm_rising'] = df['ADTM'] > df['ADTM'].shift(1)
        df['adtm_falling'] = df['ADTM'] < df['ADTM'].shift(1)
        df['adtm_above_ma'] = df['ADTM'] > df['ADTM_MA']
        df['adtm_below_ma'] = df['ADTM'] < df['ADTM_MA']
        df['golden_cross'] = (df['ADTM'] > df['ADTM_MA']) & (df['ADTM'].shift(1) <= df['ADTM_MA'].shift(1))
        df['death_cross'] = (df['ADTM'] < df['ADTM_MA']) & (df['ADTM'].shift(1) >= df['ADTM_MA'].shift(1))

        return df[['DTM', 'DBM', 'STM', 'SBM', 'ADTM', 'ADTM_MA', 'ADTM_OSC',
                   'adtm_positive', 'adtm_negative',
                   'adtm_rising', 'adtm_falling',
                   'adtm_above_ma', 'adtm_below_ma',
                   'golden_cross', 'death_cross']]
