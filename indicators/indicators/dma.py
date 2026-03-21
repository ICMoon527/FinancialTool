import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DMA(BaseIndicator):
    """
    DMA（Different of Moving Average）指标，中文称为平均线差指标或趋向平均线指标。

    DMA指标是一种基于移动平均线的技术指标，通过计算两条不同周期移动平均线的差值，
    来判断价格趋势的变化。DMA指标类似于MACD，但计算方法更简单。

    计算公式：
    1. 计算短期移动平均线：MA1 = MA(Close, N1)
    2. 计算长期移动平均线：MA2 = MA(Close, N2)
    3. 计算DMA：DMA = MA1 - MA2
    4. 计算AMA（DMA的移动平均线）：AMA = MA(DMA, N3)

    使用场景：
    - DMA由下往上穿越AMA时，为买入信号
    - DMA由上往下穿越AMA时，为卖出信号
    - DMA > 0时，表示短期趋势强于长期趋势
    - DMA < 0时，表示短期趋势弱于长期趋势
    - DMA与价格同步上升时，趋势健康
    - DMA与价格同步下降时，趋势健康
    - 当价格创新高但DMA未创新高时，为顶背离，可能见顶
    - 当价格创新低但DMA未创新低时，为底背离，可能见底

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - short_period: 短期均线周期，默认10
    - long_period: 长期均线周期，默认50
    - ama_period: AMA周期，默认10

    输出参数：
    - MA_Short: 短期移动平均线
    - MA_Long: 长期移动平均线
    - DMA: 平均线差值
    - AMA: DMA的移动平均线
    - dma_cross_ama_up: DMA上穿AMA信号
    - dma_cross_ama_down: DMA下穿AMA信号
    - dma_positive: DMA > 0信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - DMA是趋势指标，在震荡市中会产生频繁的假信号
    - DMA在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - DMA与MACD类似，但计算更简单，反应更快

    最佳实践建议：
    1. 趋势确认：DMA和AMA同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用短期和长期DMA
    4. 背离确认：背离信号需要等待价格确认后再操作
    5. 量价配合：结合成交量变化验证信号
    6. 趋势过滤：先判断大趋势，顺势操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, short_period=10, long_period=50, ama_period=10):
        """
        初始化DMA指标

        Parameters
        ----------
        short_period : int, default 10
            短期均线周期
        long_period : int, default 50
            长期均线周期
        ama_period : int, default 10
            AMA周期
        """
        self.short_period = short_period
        self.long_period = long_period
        self.ama_period = ama_period

    def calculate(self, data):
        """
        计算DMA指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DMA计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA_Short'] = df['Close'].rolling(window=self.short_period).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_period).mean()

        df['DMA'] = df['MA_Short'] - df['MA_Long']
        df['AMA'] = df['DMA'].rolling(window=self.ama_period).mean()

        df['dma_cross_ama_up'] = (df['DMA'] > df['AMA']) & (df['DMA'].shift(1) <= df['AMA'].shift(1))
        df['dma_cross_ama_down'] = (df['DMA'] < df['AMA']) & (df['DMA'].shift(1) >= df['AMA'].shift(1))
        df['dma_positive'] = df['DMA'] > 0

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['MA_Short', 'MA_Long', 'DMA', 'AMA', 'dma_cross_ama_up', 
                   'dma_cross_ama_down', 'dma_positive', 'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和DMA数据的DataFrame
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
                dma_high_idx = window['DMA'].idxmax()

                if price_high_idx > dma_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[dma_high_idx, 'Close'] and window.loc[price_high_idx, 'DMA'] < window.loc[dma_high_idx, 'DMA']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                dma_low_idx = window['DMA'].idxmin()

                if price_low_idx > dma_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[dma_low_idx, 'Close'] and window.loc[price_low_idx, 'DMA'] > window.loc[dma_low_idx, 'DMA']:
                    divergence.iloc[i] = True

        return divergence
