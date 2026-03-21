import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ASI(BaseIndicator):
    """
    ASI（Accumulation Swing Index）指标，中文称为累积震荡指标或振动升降指标。

    ASI指标由威尔斯·怀尔德（Welles Wilder）提出，是一种衡量价格真实波动的技术指标。
    ASI指标通过计算开盘价、最高价、最低价和收盘价之间的关系，来过滤虚假的价格波动，
    识别真正的趋势变化。

    计算公式：
    1. 计算A、B、C、D：
       A = |High - Close_prev|
       B = |Low - Close_prev|
       C = |High - Low_prev|
       D = |Close_prev - Open_prev|
    2. 计算真实波幅（TR）：
       TR = Max(A, B, C)
    3. 计算K：
       K = Max(A, B)
    4. 计算R：
       R = TR + 0.5 × D
    5. 计算SI（震荡指标）：
       SI = 50 × (Close - Close_prev + 0.5 × (Close - Open_prev) + 0.25 × (Close_prev - Open_prev_prev)) / R × K / TR
    6. 计算ASI（累积震荡指标）：
       ASI = 前一日ASI + 当日SI

    使用场景：
    - ASI与价格同步上升时，趋势健康
    - ASI与价格同步下降时，趋势健康
    - 当价格创新高但ASI未创新高时，为顶背离，可能见顶
    - 当价格创新低但ASI未创新低时，为底背离，可能见底
    - ASI向上突破前期高点时，为买入信号
    - ASI向下跌破前期低点时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Open'、'High'、'Low'、'Close'列

    输出参数：
    - SI: 震荡指标
    - ASI: 累积震荡指标
    - ASI_MA: ASI的移动平均线（默认周期为10）
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - ASI计算复杂，反应较慢
    - ASI在震荡市中会产生频繁的假信号
    - ASI在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用
    - ASI的绝对值没有意义，主要看趋势和背离

    最佳实践建议：
    1. 趋势确认：ASI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ASI
    4. 背离确认：背离信号需要等待价格确认后再操作
    5. 量价配合：结合成交量变化验证信号
    6. 趋势过滤：先判断大趋势，顺势操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, ma_period=10):
        """
        初始化ASI指标

        Parameters
        ----------
        ma_period : int, default 10
            ASI均线周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算ASI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Open'、'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ASI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        n = len(df)

        close_prev = df['Close'].shift(1)
        close_prev_prev = df['Close'].shift(2)
        open_prev = df['Open'].shift(1)
        low_prev = df['Low'].shift(1)

        A = np.abs(df['High'] - close_prev)
        B = np.abs(df['Low'] - close_prev)
        C = np.abs(df['High'] - low_prev)
        D = np.abs(close_prev - open_prev)

        TR = pd.concat([A, B, C], axis=1).max(axis=1)
        K = pd.concat([A, B], axis=1).max(axis=1)
        R = TR + 0.5 * D

        SI = 50 * (df['Close'] - close_prev + 0.5 * (df['Close'] - df['Open']) + 
                   0.25 * (close_prev - open_prev)) / R.replace(0, np.nan) * K / TR.replace(0, np.nan)

        df['SI'] = SI

        asi = np.zeros(n)
        asi[0] = SI.iloc[0] if pd.notna(SI.iloc[0]) else 0

        for i in range(1, n):
            if pd.notna(SI.iloc[i]):
                asi[i] = asi[i-1] + SI.iloc[i]
            else:
                asi[i] = asi[i-1]

        df['ASI'] = asi
        df['ASI_MA'] = df['ASI'].rolling(window=self.ma_period).mean()

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['SI', 'ASI', 'ASI_MA', 'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和ASI数据的DataFrame
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
                asi_high_idx = window['ASI'].idxmax()

                if price_high_idx > asi_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[asi_high_idx, 'Close'] and window.loc[price_high_idx, 'ASI'] < window.loc[asi_high_idx, 'ASI']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                asi_low_idx = window['ASI'].idxmin()

                if price_low_idx > asi_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[asi_low_idx, 'Close'] and window.loc[price_low_idx, 'ASI'] > window.loc[asi_low_idx, 'ASI']:
                    divergence.iloc[i] = True

        return divergence
