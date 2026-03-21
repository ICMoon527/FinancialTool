import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ROC(BaseIndicator):
    """
    ROC（Rate of Change）指标，中文称为变动率指标。

    ROC指标是一种动能指标，通过比较当日收盘价与N日前收盘价的差异，来衡量价格变动的速度和强度。
    ROC指标可以帮助投资者判断市场的超买超卖状态和趋势反转信号。

    计算公式：
    ROC = (C - Cn) / Cn × 100
    其中：
    - C 为当日收盘价
    - Cn 为N日前的收盘价
    - N 为计算周期，通常为10日、12日或14日

    或者使用ROC的另一种形式：
    ROC = (C / Cn - 1) × 100

    使用场景：
    - 当ROC由下往上穿越0线时，为买入信号
    - 当ROC由上往下穿越0线时，为卖出信号
    - ROC进入超买区间（如>10），可能回落，为卖出信号
    - ROC进入超卖区间（如<-10），可能反弹，为买入信号
    - 当股价创新高但ROC未创新高时，为顶背离，可能见顶
    - 当股价创新低但ROC未创新低时，为底背离，可能见底
    - ROC向上突破下降趋势线时，为买入信号
    - ROC向下跌破上升趋势线时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认12
    - overbought_threshold: 超买阈值，默认10
    - oversold_threshold: 超卖阈值，默认-10

    输出参数：
    - ROC: 变动率指标值
    - ROC_MA: ROC的移动平均线（默认周期为6）
    - overbought: 超买信号
    - oversold: 超卖信号
    - roc_cross_up: ROC上穿0线信号
    - roc_cross_down: ROC下穿0线信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - ROC是短线指标，信号频繁，容易产生伪信号
    - 超买超卖阈值需要根据市场特性调整
    - 在震荡市中ROC表现较好，在单边市中容易产生伪信号
    - 周期参数可以根据市场特性调整，10-14天是常用范围
    - 建议结合其他指标（如MA、MACD）一起使用

    最佳实践建议：
    1. 双重确认：使用ROC和ROC均线的交叉信号互相确认
    2. 趋势过滤：先判断大趋势，只在趋势方向上操作
    3. 背离确认：背离信号需要等待2-3个交易日确认
    4. 止损设置：设置严格止损，控制风险
    5. 周期组合：同时使用短期和长期ROC周期
    6. 量价配合：结合成交量变化过滤假信号
    7. 趋势线分析：绘制ROC的趋势线辅助判断
    8. 平滑处理：对ROC进行平滑处理减少噪音
    """

    def __init__(self, period=12, ma_period=6, overbought_threshold=10, oversold_threshold=-10):
        """
        初始化ROC指标

        Parameters
        ----------
        period : int, default 12
            计算周期
        ma_period : int, default 6
            ROC均线周期
        overbought_threshold : float, default 10
            超买阈值
        oversold_threshold : float, default -10
            超卖阈值
        """
        self.period = period
        self.ma_period = ma_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def calculate(self, data):
        """
        计算ROC指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ROC计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['ROC'] = (df['Close'] / df['Close'].shift(self.period) - 1) * 100

        df['ROC_MA'] = df['ROC'].rolling(window=self.ma_period).mean()

        df['overbought'] = df['ROC'] > self.overbought_threshold
        df['oversold'] = df['ROC'] < self.oversold_threshold

        df['roc_cross_up'] = (df['ROC'] > 0) & (df['ROC'].shift(1) <= 0)
        df['roc_cross_down'] = (df['ROC'] < 0) & (df['ROC'].shift(1) >= 0)

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['ROC', 'ROC_MA', 'overbought', 'oversold', 'roc_cross_up', 
                   'roc_cross_down', 'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和ROC数据的DataFrame
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
                roc_high_idx = window['ROC'].idxmax()

                if price_high_idx > roc_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[roc_high_idx, 'Close'] and window.loc[price_high_idx, 'ROC'] < window.loc[roc_high_idx, 'ROC']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                roc_low_idx = window['ROC'].idxmin()

                if price_low_idx > roc_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[roc_low_idx, 'Close'] and window.loc[price_low_idx, 'ROC'] > window.loc[roc_low_idx, 'ROC']:
                    divergence.iloc[i] = True

        return divergence
