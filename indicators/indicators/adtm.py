import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ADTM(BaseIndicator):
    """
    ADTM（Dynamic Trading Momentum）指标，中文称为动态交易动量指标。

    ADTM指标是一种结合了价格和成交量的动量指标，通过计算价格变动与成交量的关系，
    来衡量市场的买卖力量和动量变化。

    计算公式：
    1. 计算DTM（动态买卖动量）：
       - 如果High > High_prev，则DTM = High - High_prev
       - 否则DTM = 0
    2. 计算DBM（动态买卖动量）：
       - 如果Low < Low_prev，则DBM = Low_prev - Low
       - 否则DBM = 0
    3. 计算STM（动态买卖动量总和）：
       STM = Σ(DTM, N) - Σ(DBM, N)
    4. 计算ADTM：
       ADTM = STM / Max(Σ(DTM, N), Σ(DBM, N))

    使用场景：
    - ADTM在-1到1之间波动
    - ADTM > 0 时，表示多头力量较强，为买入信号
    - ADTM < 0 时，表示空头力量较强，为卖出信号
    - ADTM由下往上穿越0线时，为买入信号
    - ADTM由上往下穿越0线时，为卖出信号
    - ADTM接近1时，表示严重超买，可能回落
    - ADTM接近-1时，表示严重超卖，可能反弹

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'列
    - period: 计算周期，默认23

    输出参数：
    - DTM: 动态买卖动量（上涨）
    - DBM: 动态买卖动量（下跌）
    - STM: 动态买卖动量总和
    - ADTM: 动态交易动量指标
    - adtm_positive: ADTM > 0信号
    - adtm_negative: ADTM < 0信号
    - adtm_cross_zero_up: ADTM上穿0线信号
    - adtm_cross_zero_down: ADTM下穿0线信号
    - overbought: 超买信号（ADTM > 0.8）
    - oversold: 超卖信号（ADTM < -0.8）

    注意事项：
    - ADTM是动量指标，反应快速
    - ADTM在震荡市中会产生频繁的假信号
    - ADTM在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ADTM和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ADTM
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=23):
        """
        初始化ADTM指标

        Parameters
        ----------
        period : int, default 23
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算ADTM指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ADTM计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_prev = df['High'].shift(1)
        low_prev = df['Low'].shift(1)

        dtm = np.where(df['High'] > high_prev, df['High'] - high_prev, 0)
        dbm = np.where(df['Low'] < low_prev, low_prev - df['Low'], 0)

        df['DTM'] = dtm
        df['DBM'] = dbm

        dtm_sum = df['DTM'].rolling(window=self.period).sum()
        dbm_sum = df['DBM'].rolling(window=self.period).sum()

        df['STM'] = dtm_sum - dbm_sum

        max_sum = pd.concat([dtm_sum, dbm_sum], axis=1).max(axis=1)
        df['ADTM'] = df['STM'] / max_sum.replace(0, np.nan)

        df['adtm_positive'] = df['ADTM'] > 0
        df['adtm_negative'] = df['ADTM'] < 0
        df['adtm_cross_zero_up'] = (df['ADTM'] > 0) & (df['ADTM'].shift(1) <= 0)
        df['adtm_cross_zero_down'] = (df['ADTM'] < 0) & (df['ADTM'].shift(1) >= 0)
        df['overbought'] = df['ADTM'] > 0.8
        df['oversold'] = df['ADTM'] < -0.8

        return df[['DTM', 'DBM', 'STM', 'ADTM', 'adtm_positive', 'adtm_negative', 
                   'adtm_cross_zero_up', 'adtm_cross_zero_down', 'overbought', 'oversold']]
