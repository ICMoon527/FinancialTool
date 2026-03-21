import pandas as pd
import numpy as np
from ..base import BaseIndicator


class QACD(BaseIndicator):
    """
    QACD（Quick and Slow MACD）指标，中文称为快速慢速MACD指标。

    QACD指标是MACD指标的变种，使用更快速的EMA周期进行计算。QACD指标通过分析快速EMA和慢速EMA的差异，
    来更敏感地捕捉价格趋势的变化。

    计算公式：
    1. 计算快速EMA：EMA_Fast = EMA(Close, Fast_Period)
    2. 计算慢速EMA：EMA_Slow = EMA(Close, Slow_Period)
    3. 计算DIF（差离值）：DIF = EMA_Fast - EMA_Slow
    4. 计算DEA（讯号线）：DEA = EMA(DIF, Signal_Period)
    5. 计算MACD柱状图：MACD_Bar = DIF - DEA

    使用场景：
    - QACD由下往上穿越0线时，为买入信号
    - QACD由上往下穿越0线时，为卖出信号
    - DIF上穿DEA时，为买入信号
    - DIF下穿DEA时，为卖出信号
    - QACD与价格同步上升时，趋势健康
    - QACD与价格同步下降时，趋势健康
    - 当价格创新高但QACD未创新高时，为顶背离，可能见顶
    - 当价格创新低但QACD未创新低时，为底背离，可能见底

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认6
    - slow_period: 慢速EMA周期，默认12
    - signal_period: 信号线周期，默认5

    输出参数：
    - EMA_Fast: 快速EMA
    - EMA_Slow: 慢速EMA
    - DIF: 差离值
    - DEA: 讯号线
    - MACD_Bar: MACD柱状图
    - dif_cross_dea_up: DIF上穿DEA信号
    - dif_cross_dea_down: DIF下穿DEA信号
    - macd_cross_zero_up: MACD上穿0线信号
    - macd_cross_zero_down: MACD下穿0线信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - QACD反应快速，信号频繁
    - QACD在震荡市中会产生频繁的假信号
    - QACD在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：QACD和价格趋势同步更可靠
    2. 组合使用：与MACD等指标配合使用
    3. 多周期使用：同时使用日线和周线的QACD
    4. 背离确认：背离信号需要等待价格确认后再操作
    5. 量价配合：结合成交量变化验证信号
    6. 趋势过滤：先判断大趋势，顺势操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, fast_period=6, slow_period=12, signal_period=5):
        """
        初始化QACD指标

        Parameters
        ----------
        fast_period : int, default 6
            快速EMA周期
        slow_period : int, default 12
            慢速EMA周期
        signal_period : int, default 5
            信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算QACD指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含QACD计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA_Fast'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()

        df['DIF'] = df['EMA_Fast'] - df['EMA_Slow']
        df['DEA'] = df['DIF'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_Bar'] = df['DIF'] - df['DEA']

        df['dif_cross_dea_up'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
        df['dif_cross_dea_down'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

        df['macd_cross_zero_up'] = (df['DIF'] > 0) & (df['DIF'].shift(1) <= 0)
        df['macd_cross_zero_down'] = (df['DIF'] < 0) & (df['DIF'].shift(1) >= 0)

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['EMA_Fast', 'EMA_Slow', 'DIF', 'DEA', 'MACD_Bar', 'dif_cross_dea_up', 
                   'dif_cross_dea_down', 'macd_cross_zero_up', 'macd_cross_zero_down', 
                   'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和QACD数据的DataFrame
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
                dif_high_idx = window['DIF'].idxmax()

                if price_high_idx > dif_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[dif_high_idx, 'Close'] and window.loc[price_high_idx, 'DIF'] < window.loc[dif_high_idx, 'DIF']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                dif_low_idx = window['DIF'].idxmin()

                if price_low_idx > dif_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[dif_low_idx, 'Close'] and window.loc[price_low_idx, 'DIF'] > window.loc[dif_low_idx, 'DIF']:
                    divergence.iloc[i] = True

        return divergence
