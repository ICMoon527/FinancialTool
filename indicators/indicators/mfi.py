import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MFI(BaseIndicator):
    """
    MFI（Money Flow Index）指标，中文称为资金流向指标或资金流量指标。

    MFI指标由Gene Quong和Avrum Soudack创立，是一种结合了成交量和价格的震荡指标。
    MFI指标通过计算典型价格和成交量的乘积，来衡量资金流入和流出的强度，可以看作是成交量加权的RSI。

    计算公式：
    1. 计算典型价格：TP = (High + Low + Close) / 3
    2. 计算资金流量：MF = TP × Volume
    3. 计算正资金流和负资金流：
       - 如果当日TP > 前一日TP，则PMF = MF
       - 如果当日TP < 前一日TP，则NMF = MF
    4. 计算N日内的PMF总和和NMF总和
    5. 计算资金比率：MR = PMF / NMF
    6. 计算MFI：MFI = 100 - (100 / (1 + MR))

    使用场景：
    - MFI在0到100之间波动
    - MFI > 80 时，处于超买区间，可能回落，为卖出信号
    - MFI < 20 时，处于超卖区间，可能反弹，为买入信号
    - MFI由下往上突破50时，为买入信号
    - MFI由上往下跌破50时，为卖出信号
    - 当股价创新高但MFI未创新高时，为顶背离，可能见顶
    - 当股价创新低但MFI未创新低时，为底背离，可能见底

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'、'Volume'列
    - period: 计算周期，默认14

    输出参数：
    - MFI: 资金流向指标值
    - TP: 典型价格
    - MF: 资金流量
    - overbought: 超买信号（MFI > 80）
    - oversold: 超卖信号（MFI < 20）
    - cross_50_up: 上穿50信号
    - cross_50_down: 下穿50信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - MFI结合了成交量信息，比单纯的价格指标更可靠
    - MFI在震荡市中表现较好，在单边市中可能长期处于超买或超卖区
    - 周期参数可以根据市场特性调整，14天是最常用的
    - 建议结合其他指标（如MACD、RSI）一起使用
    - MFI在成交量异常时需要谨慎解读

    最佳实践建议：
    1. 双重确认：等待MFI连续2天处于超买/超卖区再行动
    2. 趋势配合：结合均线判断大趋势，顺势操作
    3. 背离确认：背离信号需要等待价格确认后再操作
    4. 量价分析：结合成交量变化验证信号
    5. 多周期使用：同时使用日线和周线的MFI信号
    6. 组合使用：与RSI、KDJ等震荡指标配合使用
    7. 时间过滤：避免在重要数据发布前操作
    8. 止损设置：设置严格止损，控制风险
    """

    def __init__(self, period=14):
        """
        初始化MFI指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算MFI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MFI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['MF'] = df['TP'] * df['Volume']

        tp_change = df['TP'].diff()

        df['PMF'] = np.where(tp_change > 0, df['MF'], 0)
        df['NMF'] = np.where(tp_change < 0, df['MF'], 0)

        pmf_sum = df['PMF'].rolling(window=self.period).sum()
        nmf_sum = df['NMF'].rolling(window=self.period).sum()

        mr = pmf_sum / nmf_sum.replace(0, np.nan)

        df['MFI'] = 100 - (100 / (1 + mr))

        df['overbought'] = df['MFI'] > 80
        df['oversold'] = df['MFI'] < 20

        df['cross_50_up'] = (df['MFI'] > 50) & (df['MFI'].shift(1) <= 50)
        df['cross_50_down'] = (df['MFI'] < 50) & (df['MFI'].shift(1) >= 50)

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['MFI', 'TP', 'MF', 'overbought', 'oversold', 'cross_50_up', 
                   'cross_50_down', 'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和MFI数据的DataFrame
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
                mfi_high_idx = window['MFI'].idxmax()

                if price_high_idx > mfi_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[mfi_high_idx, 'Close'] and window.loc[price_high_idx, 'MFI'] < window.loc[mfi_high_idx, 'MFI']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                mfi_low_idx = window['MFI'].idxmin()

                if price_low_idx > mfi_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[mfi_low_idx, 'Close'] and window.loc[price_low_idx, 'MFI'] > window.loc[mfi_low_idx, 'MFI']:
                    divergence.iloc[i] = True

        return divergence
