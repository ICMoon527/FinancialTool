import pandas as pd
import numpy as np
from ..base import BaseIndicator


class WR(BaseIndicator):
    """
    WR（Williams %R）指标，中文称为威廉指标或威廉超买超卖指标。

    WR指标由拉里·威廉斯（Larry Williams）于1973年提出，是一种利用振荡点来反映市场超买超卖现象，
    预测价格循环运动中的高点和低点的技术指标。

    计算公式：
    WR = (Hn - C) / (Hn - Ln) × (-100)
    其中：
    - Hn 为N日内的最高价
    - Ln 为N日内的最低价
    - C 为当日收盘价
    - N 为计算周期，通常为10日或14日

    使用场景：
    - WR值在0到-100之间波动
    - WR > -20 时，处于超买区间，可能回落，为卖出信号
    - WR < -80 时，处于超卖区间，可能反弹，为买入信号
    - WR由超卖区向上回升，突破-50中界线时，由弱转强，可以买入
    - WR由超买区向下跌落，跌破-50中界线时，由强转弱，可以卖出
    - 当WR进入超买区间后，继续上行，股价却不创新高，为顶背离
    - 当WR进入超卖区间后，继续下行，股价却不创新低，为底背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14

    输出参数：
    - WR: 威廉指标值
    - overbought: 超买信号（WR > -20）
    - oversold: 超卖信号（WR < -80）
    - cross_middle_up: 上穿中界线（-50）信号
    - cross_middle_down: 下穿中界线（-50）信号

    注意事项：
    - WR是震荡指标，在震荡市中表现较好，在单边市中容易产生伪信号
    - 超买超卖信号需要结合其他指标确认，避免单一指标决策
    - 周期参数可以根据市场特性调整，10-14天是常用范围
    - WR与RSI类似，但计算方法不同，可以配合使用
    - 在强势上升行情中，WR可能长期处于超买区，不应过早卖出

    最佳实践建议：
    1. 信号确认：等待WR连续2-3天处于超买/超卖区再行动
    2. 趋势配合：结合均线判断大趋势，顺势操作
    3. 背离确认：背离信号需要等待价格确认后再操作
    4. 量价分析：结合成交量变化过滤假信号
    5. 多周期使用：同时使用日线和周线的WR信号
    6. 组合使用：与RSI、KDJ等震荡指标配合使用
    7. 时间过滤：避免在重要数据发布前操作
    8. 止损设置：设置严格止损，控制风险
    """

    def __init__(self, period=14):
        """
        初始化WR指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算WR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含WR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        hh = df['High'].rolling(window=self.period).max()
        ll = df['Low'].rolling(window=self.period).min()

        df['WR'] = (hh - df['Close']) / (hh - ll) * (-100)

        df['overbought'] = df['WR'] > -20
        df['oversold'] = df['WR'] < -80

        df['cross_middle_up'] = (df['WR'] > -50) & (df['WR'].shift(1) <= -50)
        df['cross_middle_down'] = (df['WR'] < -50) & (df['WR'].shift(1) >= -50)

        return df[['WR', 'overbought', 'oversold', 'cross_middle_up', 'cross_middle_down']]
