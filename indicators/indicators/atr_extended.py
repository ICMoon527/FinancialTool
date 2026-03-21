import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ATRExtended(BaseIndicator):
    """
    ATRExtended（ATR扩展），中文称为ATR扩展。

    ATRExtended指标是ATR指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    ATRExtended指标结合了真实波幅和平均真实波幅，来反应价格的波动程度。

    计算公式：
    1. 计算TR：TR = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
    2. 计算ATR：ATR = EMA(TR, N)
    3. 计算ATR_Percent：ATR_Percent = (ATR / Close) × 100
    4. 计算ATR_Ratio：ATR_Ratio = TR / ATR

    使用场景：
    - ATR上升时，表示波动加剧，可能出现趋势
    - ATR下降时，表示波动收窄，可能进入盘整
    - ATR_Percent高时，表示波动大
    - ATR_Percent低时，表示波动小
    - ATR_Ratio > 1 时，表示当前波动大于平均波动
    - ATR_Ratio < 1 时，表示当前波动小于平均波动

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14

    输出参数：
    - TR: 真实波幅
    - ATR: 平均真实波幅
    - ATR_Percent: ATR百分比
    - ATR_Ratio: TR与ATR的比率
    - atr_rising: ATR上升信号
    - atr_falling: ATR下降信号
    - atr_high: ATR_Percent高信号（>2）
    - atr_low: ATR_Percent低信号（<1）
    - atr_above_avg: ATR_Ratio > 1信号
    - atr_below_avg: ATR_Ratio < 1信号

    注意事项：
    - ATRExtended是波动指标，不直接给出买卖信号
    - ATRExtended主要用于衡量风险和设置止损
    - ATR的绝对数值与价格水平有关，需要用百分比比较
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：结合其他趋势指标判断方向
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ATRExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：ATR的倍数设置止损位置
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14):
        """
        初始化ATRExtended指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算ATRExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ATRExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - prev_close)
        tr3 = abs(df['Low'] - prev_close)
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].ewm(span=self.period, adjust=False).mean()
        df['ATR_Percent'] = (df['ATR'] / df['Close'].replace(0, np.nan)) * 100
        df['ATR_Percent'] = df['ATR_Percent'].fillna(0)
        df['ATR_Ratio'] = df['TR'] / df['ATR'].replace(0, np.nan)
        df['ATR_Ratio'] = df['ATR_Ratio'].fillna(1)

        df['atr_rising'] = df['ATR'] > df['ATR'].shift(1)
        df['atr_falling'] = df['ATR'] < df['ATR'].shift(1)
        df['atr_high'] = df['ATR_Percent'] > 2
        df['atr_low'] = df['ATR_Percent'] < 1
        df['atr_above_avg'] = df['ATR_Ratio'] > 1
        df['atr_below_avg'] = df['ATR_Ratio'] < 1

        return df[['TR', 'ATR', 'ATR_Percent', 'ATR_Ratio',
                   'atr_rising', 'atr_falling',
                   'atr_high', 'atr_low',
                   'atr_above_avg', 'atr_below_avg']]
