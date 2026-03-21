import pandas as pd
import numpy as np
from ..base import BaseIndicator


class OBV(BaseIndicator):
    """
    OBV（On Balance Volume）指标，中文称为能量潮指标或平衡成交量指标。

    OBV指标由约瑟夫·格兰维尔（Joseph Granville）于1963年提出，是一种将成交量与价格结合的技术指标。
    OBV指标通过累积成交量来衡量资金流入和流出的情况，认为成交量是价格变化的先行指标。

    计算公式：
    1. 如果当日收盘价 > 前一日收盘价：OBV = 前一日OBV + 当日成交量
    2. 如果当日收盘价 < 前一日收盘价：OBV = 前一日OBV - 当日成交量
    3. 如果当日收盘价 = 前一日收盘价：OBV = 前一日OBV

    使用场景：
    - OBV上升时，表示资金流入，为买入信号
    - OBV下降时，表示资金流出，为卖出信号
    - OBV与价格同步上升时，趋势健康
    - OBV与价格同步下降时，趋势健康
    - 当价格创新高但OBV未创新高时，为顶背离，可能见顶
    - 当价格创新低但OBV未创新低时，为底背离，可能见底
    - OBV突破下降趋势线时，为买入信号
    - OBV跌破上升趋势线时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列

    输出参数：
    - OBV: 能量潮指标值
    - OBV_MA: OBV的移动平均线（默认周期为10）
    - obv_rising: OBV上升信号
    - obv_falling: OBV下降信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号
    - obv_cross_ma_up: OBV上穿均线信号
    - obv_cross_ma_down: OBV下穿均线信号

    注意事项：
    - OBV是成交量指标，不直接反映价格
    - OBV在成交量异常时需要谨慎解读
    - OBV的绝对值没有意义，主要看趋势和背离
    - 建议结合其他指标（如MACD、RSI）一起使用
    - OBV在低成交量股票上效果较差

    最佳实践建议：
    1. 趋势确认：OBV与价格趋势同步更可靠
    2. 背离确认：背离信号需要等待价格确认后再操作
    3. 组合使用：与价格指标配合使用
    4. 多周期使用：同时使用日线和周线的OBV
    5. 量价配合：结合价格和成交量变化验证信号
    6. 趋势线分析：绘制OBV的趋势线辅助判断
    7. 平滑处理：使用OBV均线减少噪音
    8. 相对比较：比较不同股票的OBV趋势
    """

    def __init__(self, ma_period=10):
        """
        初始化OBV指标

        Parameters
        ----------
        ma_period : int, default 10
            OBV均线周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算OBV指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含OBV计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        obv = np.zeros(len(df))
        obv[0] = df['Volume'].iloc[0]

        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]

        df['OBV'] = obv
        df['OBV_MA'] = df['OBV'].rolling(window=self.ma_period).mean()

        df['obv_rising'] = df['OBV'] > df['OBV'].shift(1)
        df['obv_falling'] = df['OBV'] < df['OBV'].shift(1)

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        df['obv_cross_ma_up'] = (df['OBV'] > df['OBV_MA']) & (df['OBV'].shift(1) <= df['OBV_MA'].shift(1))
        df['obv_cross_ma_down'] = (df['OBV'] < df['OBV_MA']) & (df['OBV'].shift(1) >= df['OBV_MA'].shift(1))

        return df[['OBV', 'OBV_MA', 'obv_rising', 'obv_falling', 'top_divergence', 
                   'bottom_divergence', 'obv_cross_ma_up', 'obv_cross_ma_down']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和OBV数据的DataFrame
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
                obv_high_idx = window['OBV'].idxmax()

                if price_high_idx > obv_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[obv_high_idx, 'Close'] and window.loc[price_high_idx, 'OBV'] < window.loc[obv_high_idx, 'OBV']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                obv_low_idx = window['OBV'].idxmin()

                if price_low_idx > obv_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[obv_low_idx, 'Close'] and window.loc[price_low_idx, 'OBV'] > window.loc[obv_low_idx, 'OBV']:
                    divergence.iloc[i] = True

        return divergence
