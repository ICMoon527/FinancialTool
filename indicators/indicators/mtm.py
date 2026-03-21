import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MTM(BaseIndicator):
    """
    MTM（Momentum）指标，中文称为动量指标。

    动量指标是一种专门研究股价波动的中短期技术分析工具。动量指标认为股价的涨跌幅度随着时间的推移会逐渐变小，
    股价变化的速度和能量也会慢慢减缓，行情可能反转。

    计算公式：
    MTM = C - Cn
    其中：
    - C 为当日收盘价
    - Cn 为N日前的收盘价
    - N 为计算周期，通常为10日、12日或14日

    或者使用百分比形式：
    MTM = (C / Cn - 1) × 100

    使用场景：
    - 当MTM由下往上穿越0线时，为买入信号
    - 当MTM由上往下穿越0线时，为卖出信号
    - 当股价继续上涨，MTM却下降时，为顶背离，可能见顶
    - 当股价继续下跌，MTM却上升时，为底背离，可能见底
    - MTM在高位回头向下时，为卖出信号
    - MTM在低位回头向上时，为买入信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认10
    - use_percent: 是否使用百分比形式，默认False

    输出参数：
    - MTM: 动量值
    - MTM_MA: MTM的移动平均线（默认周期为6）
    - mtm_cross_up: MTM上穿0线信号
    - mtm_cross_down: MTM下穿0线信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - MTM是短线指标，信号频繁，容易产生伪信号
    - 建议结合其他指标（如MA、MACD）一起使用
    - 在震荡市中MTM表现较差，在趋势市中表现较好
    - 周期参数可以根据市场特性调整，10-14天是常用范围
    - 百分比形式的MTM更适合不同价格股票的比较

    最佳实践建议：
    1. 双重确认：使用MTM和MTM均线的交叉信号互相确认
    2. 趋势过滤：先判断大趋势，只在趋势方向上操作
    3. 背离确认：背离信号需要等待2-3个交易日确认
    4. 止损设置：MTM信号失败时立即止损
    5. 周期组合：同时使用短期和长期MTM周期
    6. 量价配合：结合成交量变化过滤假信号
    7. 超买超卖：设置MTM的超买超卖阈值
    8. 平滑处理：对MTM进行平滑处理减少噪音
    """

    def __init__(self, period=10, ma_period=6, use_percent=False):
        """
        初始化MTM指标

        Parameters
        ----------
        period : int, default 10
            动量计算周期
        ma_period : int, default 6
            MTM均线周期
        use_percent : bool, default False
            是否使用百分比形式
        """
        self.period = period
        self.ma_period = ma_period
        self.use_percent = use_percent

    def calculate(self, data):
        """
        计算MTM指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MTM计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        if self.use_percent:
            df['MTM'] = (df['Close'] / df['Close'].shift(self.period) - 1) * 100
        else:
            df['MTM'] = df['Close'] - df['Close'].shift(self.period)

        df['MTM_MA'] = df['MTM'].rolling(window=self.ma_period).mean()

        df['mtm_cross_up'] = (df['MTM'] > 0) & (df['MTM'].shift(1) <= 0)
        df['mtm_cross_down'] = (df['MTM'] < 0) & (df['MTM'].shift(1) >= 0)

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['MTM', 'MTM_MA', 'mtm_cross_up', 'mtm_cross_down', 
                   'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和MTM数据的DataFrame
        divergence_type : str
            'top'表示顶背离，'bottom'表示底背离

        Returns
        -------
        pandas.Series
            背离信号序列
        """
        divergence = pd.Series(False, index=df.index)

        lookback = 20

        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i+1]

            if divergence_type == 'top':
                price_high_idx = window['Close'].idxmax()
                mtm_high_idx = window['MTM'].idxmax()

                if price_high_idx > mtm_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[mtm_high_idx, 'Close'] and window.loc[price_high_idx, 'MTM'] < window.loc[mtm_high_idx, 'MTM']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                mtm_low_idx = window['MTM'].idxmin()

                if price_low_idx > mtm_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[mtm_low_idx, 'Close'] and window.loc[price_low_idx, 'MTM'] > window.loc[mtm_low_idx, 'MTM']:
                    divergence.iloc[i] = True

        return divergence
