import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DMI(BaseIndicator):
    """
    DMI（Directional Movement Index）指标，中文称为动向指标或趋向指标。

    DMI指标由威尔斯·怀尔德（Welles Wilder）于1978年提出，是一种衡量价格趋势方向和强度的技术指标。
    DMI指标由+DI（上升动向指标）、-DI（下降动向指标）和ADX（平均趋向指标）组成。

    计算公式：
    1. 计算真实波幅（TR）：
       TR = Max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    2. 计算上升动向（+DM）和下降动向（-DM）：
       +DM = High - High_prev（如果High - High_prev > Low_prev - Low，否则为0）
       -DM = Low_prev - Low（如果Low_prev - Low > High - High_prev，否则为0）
    3. 计算平滑后的TR、+DM、-DM（使用Wilder的平滑方法）：
       初始值 = 前N日的和
       后续值 = 前一日值 - 前一日值/N + 当日值
    4. 计算+DI和-DI：
       +DI = (+DM / TR) × 100
       -DI = (-DM / TR) × 100
    5. 计算DX（动向指标）：
       DX = |+DI - -DI| / |+DI + -DI| × 100
    6. 计算ADX（平均趋向指标）：
       ADX = DX的N日平滑移动平均

    使用场景：
    - +DI上穿-DI时，为买入信号
    - +DI下穿-DI时，为卖出信号
    - ADX > 25时，表示趋势明显
    - ADX < 20时，表示趋势不明显，适合震荡操作
    - ADX由上升转为下降时，表示趋势可能结束
    - ADX在低位拐头向上时，表示新趋势开始

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14
    - adx_period: ADX计算周期，默认14

    输出参数：
    - TR: 真实波幅
    - PDI: +DI（上升动向指标）
    - MDI: -DI（下降动向指标）
    - DX: 动向指标
    - ADX: 平均趋向指标
    - pdi_cross_mdi_up: +DI上穿-DI信号
    - pdi_cross_mdi_down: +DI下穿-DI信号
    - adx_trending: ADX > 25（趋势明显）信号

    注意事项：
    - DMI是趋势指标，在震荡市中会产生频繁的假信号
    - DMI在单边市中表现最好
    - ADX只衡量趋势强度，不指示方向
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势过滤：ADX > 25时才使用趋势信号
    2. 组合使用：与MA、MACD等趋势指标配合使用
    3. 震荡规避：ADX < 20时使用震荡指标
    4. 确认信号：等待+DI和-DI交叉确认后再操作
    5. 多周期使用：同时使用日线和周线的DMI
    6. 趋势跟踪：ADX上升时持有，下降时离场
    7. 止损设置：根据ADX调整止损位
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, period=14, adx_period=14):
        """
        初始化DMI指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        adx_period : int, default 14
            ADX计算周期
        """
        self.period = period
        self.adx_period = adx_period

    def calculate(self, data):
        """
        计算DMI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DMI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        n = len(df)

        high_prev = df['High'].shift(1)
        low_prev = df['Low'].shift(1)
        close_prev = df['Close'].shift(1)

        tr1 = df['High'] - df['Low']
        tr2 = np.abs(df['High'] - close_prev)
        tr3 = np.abs(df['Low'] - close_prev)
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = np.where((df['High'] - high_prev) > (low_prev - df['Low']), 
                           np.maximum(df['High'] - high_prev, 0), 0)
        minus_dm = np.where((low_prev - df['Low']) > (df['High'] - high_prev), 
                            np.maximum(low_prev - df['Low'], 0), 0)

        tr_smooth = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)

        if n > self.period:
            tr_smooth[self.period] = df['TR'].iloc[1:self.period+1].sum()
            plus_dm_smooth[self.period] = pd.Series(plus_dm, index=df.index).iloc[1:self.period+1].sum()
            minus_dm_smooth[self.period] = pd.Series(minus_dm, index=df.index).iloc[1:self.period+1].sum()

            for i in range(self.period + 1, n):
                tr_smooth[i] = tr_smooth[i-1] - tr_smooth[i-1]/self.period + df['TR'].iloc[i]
                plus_dm_smooth[i] = plus_dm_smooth[i-1] - plus_dm_smooth[i-1]/self.period + plus_dm[i]
                minus_dm_smooth[i] = minus_dm_smooth[i-1] - minus_dm_smooth[i-1]/self.period + minus_dm[i]

        df['TR_smooth'] = tr_smooth
        df['plus_dm_smooth'] = plus_dm_smooth
        df['minus_dm_smooth'] = minus_dm_smooth

        df['PDI'] = (df['plus_dm_smooth'] / df['TR_smooth'].replace(0, np.nan)) * 100
        df['MDI'] = (df['minus_dm_smooth'] / df['TR_smooth'].replace(0, np.nan)) * 100

        df['DX'] = np.abs(df['PDI'] - df['MDI']) / np.abs(df['PDI'] + df['MDI']).replace(0, np.nan) * 100

        df['ADX'] = df['DX'].rolling(window=self.adx_period).mean()

        adx_smooth = np.zeros(n)
        if n > self.period + self.adx_period:
            adx_smooth[self.period + self.adx_period - 1] = df['DX'].iloc[self.period:self.period+self.adx_period].mean()
            for i in range(self.period + self.adx_period, n):
                adx_smooth[i] = (adx_smooth[i-1] * (self.adx_period - 1) + df['DX'].iloc[i]) / self.adx_period
        df['ADX'] = adx_smooth

        df['pdi_cross_mdi_up'] = (df['PDI'] > df['MDI']) & (df['PDI'].shift(1) <= df['MDI'].shift(1))
        df['pdi_cross_mdi_down'] = (df['PDI'] < df['MDI']) & (df['PDI'].shift(1) >= df['MDI'].shift(1))
        df['adx_trending'] = df['ADX'] > 25

        return df[['TR', 'PDI', 'MDI', 'DX', 'ADX', 'pdi_cross_mdi_up', 
                   'pdi_cross_mdi_down', 'adx_trending']]
