import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MIKE(BaseIndicator):
    """
    MIKE（麦克指标），中文称为麦克指标。

    MIKE指标是一种路径分析指标，通过计算多个支撑阻力位，来判断价格的走势。
    MIKE指标通过计算初级、中级、强力的支撑和阻力位，来提供价格预测。

    计算公式：
    1. 计算TP（典型价）：TP = (High + Low + Close) / 3
    2. 计算初级支撑：WEAK_S = TP - (High - Low)
    3. 计算中级支撑：MID_S = 2 × TP - High
    4. 计算强力支撑：STRONG_S = TP - 2 × (High - Low)
    5. 计算初级阻力：WEAK_R = TP + (High - Low)
    6. 计算中级阻力：MID_R = 2 × TP - Low
    7. 计算强力阻力：STRONG_R = TP + 2 × (High - Low)

    使用场景：
    - 价格突破阻力位时，表示上升趋势，为买入信号
    - 价格跌破支撑位时，表示下降趋势，为卖出信号
    - 价格在支撑位反弹时，表示支撑有效
    - 价格在阻力位回落时，表示阻力有效
    - 多个阻力位被突破时，表示趋势强劲

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列

    输出参数：
    - TP: 典型价
    - WEAK_S: 初级支撑
    - MID_S: 中级支撑
    - STRONG_S: 强力支撑
    - WEAK_R: 初级阻力
    - MID_R: 中级阻力
    - STRONG_R: 强力阻力
    - close_above_weak_r: 价格突破初级阻力信号
    - close_above_mid_r: 价格突破中级阻力信号
    - close_above_strong_r: 价格突破强力阻力信号
    - close_below_weak_s: 价格跌破初级支撑信号
    - close_below_mid_s: 价格跌破中级支撑信号
    - close_below_strong_s: 价格跌破强力支撑信号

    注意事项：
    - MIKE是路径分析指标，提供多个支撑阻力位
    - MIKE在震荡市中会产生频繁的假信号
    - MIKE在单边市中表现最好
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 多层确认：多个阻力位被突破更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的MIKE
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self):
        """
        初始化MIKE指标
        """
        pass

    def calculate(self, data):
        """
        计算MIKE指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MIKE计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        hl = df['High'] - df['Low']
        df['WEAK_S'] = df['TP'] - hl
        df['MID_S'] = 2 * df['TP'] - df['High']
        df['STRONG_S'] = df['TP'] - 2 * hl
        df['WEAK_R'] = df['TP'] + hl
        df['MID_R'] = 2 * df['TP'] - df['Low']
        df['STRONG_R'] = df['TP'] + 2 * hl

        df['close_above_weak_r'] = df['Close'] > df['WEAK_R']
        df['close_above_mid_r'] = df['Close'] > df['MID_R']
        df['close_above_strong_r'] = df['Close'] > df['STRONG_R']
        df['close_below_weak_s'] = df['Close'] < df['WEAK_S']
        df['close_below_mid_s'] = df['Close'] < df['MID_S']
        df['close_below_strong_s'] = df['Close'] < df['STRONG_S']

        return df[['TP', 'WEAK_S', 'MID_S', 'STRONG_S', 'WEAK_R', 'MID_R', 'STRONG_R',
                   'close_above_weak_r', 'close_above_mid_r', 'close_above_strong_r',
                   'close_below_weak_s', 'close_below_mid_s', 'close_below_strong_s']]
