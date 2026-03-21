import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VMACD(BaseIndicator):
    """
    VMACD（Volume Moving Average Convergence Divergence）指标，中文称为成交量MACD指标。

    VMACD指标是MACD指标的变种，使用成交量代替价格进行计算。VMACD指标通过分析成交量的变化，
    来预测价格趋势的变化，认为成交量是价格变化的先行指标。

    计算公式：
    1. 计算成交量的快速EMA：EMA_Fast = EMA(Volume, Fast_Period)
    2. 计算成交量的慢速EMA：EMA_Slow = EMA(Volume, Slow_Period)
    3. 计算DIF（差离值）：DIF = EMA_Fast - EMA_Slow
    4. 计算DEA（讯号线）：DEA = EMA(DIF, Signal_Period)
    5. 计算MACD柱状图：MACD_Bar = 2 × (DIF - DEA)

    使用场景：
    - VMACD由下往上穿越0线时，为买入信号
    - VMACD由上往下穿越0线时，为卖出信号
    - DIF上穿DEA时，为买入信号
    - DIF下穿DEA时，为卖出信号
    - VMACD与价格同步上升时，趋势健康
    - VMACD与价格同步下降时，趋势健康
    - 当价格创新高但VMACD未创新高时，为顶背离，可能见顶
    - 当价格创新低但VMACD未创新低时，为底背离，可能见底

    输入参数：
    - data: DataFrame，必须包含'Volume'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号线周期，默认9

    输出参数：
    - EMA_Fast: 成交量快速EMA
    - EMA_Slow: 成交量慢速EMA
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
    - VMACD是成交量指标，不直接反映价格
    - VMACD在成交量异常时需要谨慎解读
    - VMACD在震荡市中会产生频繁的假信号
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VMACD和价格趋势同步更可靠
    2. 组合使用：与MACD等价格指标配合使用
    3. 多周期使用：同时使用日线和周线的VMACD
    4. 背离确认：背离信号需要等待价格确认后再操作
    5. 量价配合：结合成交量变化验证信号
    6. 趋势过滤：先判断大趋势，顺势操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化VMACD指标

        Parameters
        ----------
        fast_period : int, default 12
            快速EMA周期
        slow_period : int, default 26
            慢速EMA周期
        signal_period : int, default 9
            信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算VMACD指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VMACD计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA_Fast'] = df['Volume'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['Volume'].ewm(span=self.slow_period, adjust=False).mean()

        df['DIF'] = df['EMA_Fast'] - df['EMA_Slow']
        df['DEA'] = df['DIF'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_Bar'] = 2 * (df['DIF'] - df['DEA'])

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
            包含价格和VMACD数据的DataFrame
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
