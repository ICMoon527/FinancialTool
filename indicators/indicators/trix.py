import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TRIX(BaseIndicator):
    """
    TRIX（Triple Exponential Average）指标，中文称为三重指数平滑移动平均指标。

    TRIX指标由杰克·赫特森（Jack Hutson）提出，是一种基于三重指数移动平均的趋势指标。
    TRIX指标通过计算三重指数平滑移动平均的变化率，来过滤短期波动，识别长期趋势。

    计算公式：
    1. 计算收盘价的N日指数移动平均：EMA1 = EMA(Close, N)
    2. 计算EMA1的N日指数移动平均：EMA2 = EMA(EMA1, N)
    3. 计算EMA2的N日指数移动平均：EMA3 = EMA(EMA2, N)
    4. 计算TRIX：TRIX = (EMA3 - EMA3_prev) / EMA3_prev × 100
    5. 计算TRMA（TRIX的移动平均线）：TRMA = MA(TRIX, M)

    使用场景：
    - TRIX由下往上穿越TRMA时，为买入信号
    - TRIX由上往下穿越TRMA时，为卖出信号
    - TRIX > 0时，表示处于上升趋势
    - TRIX < 0时，表示处于下降趋势
    - TRIX与价格同步上升时，趋势健康
    - TRIX与价格同步下降时，趋势健康
    - 当价格创新高但TRIX未创新高时，为顶背离，可能见顶
    - 当价格创新低但TRIX未创新低时，为底背离，可能见底

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: TRIX计算周期，默认12
    - signal_period: 信号线周期，默认20

    输出参数：
    - EMA1: 第一重指数移动平均
    - EMA2: 第二重指数移动平均
    - EMA3: 第三重指数移动平均
    - TRIX: 三重指数平滑指标
    - TRMA: TRIX的移动平均线
    - trix_cross_trma_up: TRIX上穿TRMA信号
    - trix_cross_trma_down: TRIX下穿TRMA信号
    - trix_positive: TRIX > 0信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - TRIX是趋势指标，反应较慢，不适合短线操作
    - TRIX在震荡市中会产生频繁的假信号
    - TRIX在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：TRIX和TRMA同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的TRIX
    4. 背离确认：背离信号需要等待价格确认后再操作
    5. 量价配合：结合成交量变化验证信号
    6. 趋势过滤：先判断大趋势，顺势操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, period=12, signal_period=20):
        """
        初始化TRIX指标

        Parameters
        ----------
        period : int, default 12
            TRIX计算周期
        signal_period : int, default 20
            信号线周期
        """
        self.period = period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算TRIX指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含TRIX计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA1'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['EMA2'] = df['EMA1'].ewm(span=self.period, adjust=False).mean()
        df['EMA3'] = df['EMA2'].ewm(span=self.period, adjust=False).mean()

        ema3_prev = df['EMA3'].shift(1)
        df['TRIX'] = (df['EMA3'] - ema3_prev) / ema3_prev.replace(0, np.nan) * 100

        df['TRMA'] = df['TRIX'].rolling(window=self.signal_period).mean()

        df['trix_cross_trma_up'] = (df['TRIX'] > df['TRMA']) & (df['TRIX'].shift(1) <= df['TRMA'].shift(1))
        df['trix_cross_trma_down'] = (df['TRIX'] < df['TRMA']) & (df['TRIX'].shift(1) >= df['TRMA'].shift(1))
        df['trix_positive'] = df['TRIX'] > 0

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['EMA1', 'EMA2', 'EMA3', 'TRIX', 'TRMA', 'trix_cross_trma_up', 
                   'trix_cross_trma_down', 'trix_positive', 'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和TRIX数据的DataFrame
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
                trix_high_idx = window['TRIX'].idxmax()

                if price_high_idx > trix_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[trix_high_idx, 'Close'] and window.loc[price_high_idx, 'TRIX'] < window.loc[trix_high_idx, 'TRIX']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                trix_low_idx = window['TRIX'].idxmin()

                if price_low_idx > trix_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[trix_low_idx, 'Close'] and window.loc[price_low_idx, 'TRIX'] > window.loc[trix_low_idx, 'TRIX']:
                    divergence.iloc[i] = True

        return divergence
