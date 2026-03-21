import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EMV(BaseIndicator):
    """
    EMV（Ease of Movement）指标，中文称为简易波动指标或简易移动指标。

    EMV指标由理查德·阿姆斯（Richard Arms）提出，是一种结合了价格和成交量的技术指标。
    EMV指标通过计算价格波动的难易程度，来衡量资金推动价格的能力。

    计算公式：
    1. 计算中间价：M = (High + Low) / 2
    2. 计算前一日中间价：M_prev = (High_prev + Low_prev) / 2
    3. 计算距离比：Distance Ratio = M - M_prev
    4. 计算箱体比：Box Ratio = Volume / (High - Low)
    5. 计算EM：EM = Distance Ratio / Box Ratio
    6. 计算EMV：EMV = EM的N日简单移动平均

    使用场景：
    - EMV > 0时，表示价格容易向上波动，为买入信号
    - EMV < 0时，表示价格容易向下波动，为卖出信号
    - EMV由下往上穿越0线时，为买入信号
    - EMV由上往下穿越0线时，为卖出信号
    - EMV与价格同步上升时，趋势健康
    - EMV与价格同步下降时，趋势健康
    - 当价格创新高但EMV未创新高时，为顶背离，可能见顶
    - 当价格创新低但EMV未创新低时，为底背离，可能见底

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Volume'列
    - period: 计算周期，默认14
    - ma_period: EMV均线周期，默认9

    输出参数：
    - EM: 简易波动值
    - EMV: 简易波动指标
    - EMV_MA: EMV的移动平均线
    - emv_positive: EMV > 0信号
    - emv_cross_zero_up: EMV上穿0线信号
    - emv_cross_zero_down: EMV下穿0线信号
    - top_divergence: 顶背离信号
    - bottom_divergence: 底背离信号

    注意事项：
    - EMV在成交量异常时需要谨慎解读
    - EMV在横盘整理时效果较差
    - EMV在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：EMV和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的EMV
    4. 背离确认：背离信号需要等待价格确认后再操作
    5. 量价配合：结合成交量变化验证信号
    6. 趋势过滤：先判断大趋势，顺势操作
    7. 止损设置：设置严格止损，控制风险
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, period=14, ma_period=9):
        """
        初始化EMV指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        ma_period : int, default 9
            EMV均线周期
        """
        self.period = period
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算EMV指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含EMV计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        high_low = df['High'] - df['Low']
        high_low_prev = high_low.shift(1)

        m = (df['High'] + df['Low']) / 2
        m_prev = m.shift(1)

        distance_ratio = m - m_prev
        box_ratio = df['Volume'] / high_low.replace(0, np.nan)

        df['EM'] = distance_ratio / box_ratio.replace(0, np.nan)
        df['EMV'] = df['EM'].rolling(window=self.period).mean()
        df['EMV_MA'] = df['EMV'].rolling(window=self.ma_period).mean()

        df['emv_positive'] = df['EMV'] > 0
        df['emv_cross_zero_up'] = (df['EMV'] > 0) & (df['EMV'].shift(1) <= 0)
        df['emv_cross_zero_down'] = (df['EMV'] < 0) & (df['EMV'].shift(1) >= 0)

        df['top_divergence'] = self._detect_divergence(df, 'top')
        df['bottom_divergence'] = self._detect_divergence(df, 'bottom')

        return df[['EM', 'EMV', 'EMV_MA', 'emv_positive', 'emv_cross_zero_up', 
                   'emv_cross_zero_down', 'top_divergence', 'bottom_divergence']]

    def _detect_divergence(self, df, divergence_type):
        """
        检测背离信号

        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和EMV数据的DataFrame
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
                emv_high_idx = window['EMV'].idxmax()

                if price_high_idx > emv_high_idx and window.loc[price_high_idx, 'Close'] > window.loc[emv_high_idx, 'Close'] and window.loc[price_high_idx, 'EMV'] < window.loc[emv_high_idx, 'EMV']:
                    divergence.iloc[i] = True
            else:
                price_low_idx = window['Close'].idxmin()
                emv_low_idx = window['EMV'].idxmin()

                if price_low_idx > emv_low_idx and window.loc[price_low_idx, 'Close'] < window.loc[emv_low_idx, 'Close'] and window.loc[price_low_idx, 'EMV'] > window.loc[emv_low_idx, 'EMV']:
                    divergence.iloc[i] = True

        return divergence
