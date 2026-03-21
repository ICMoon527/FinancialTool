import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PUCU(BaseIndicator):
    """
    PUCU（瀑布线变种），中文称为瀑布线变种。

    PUCU指标是一种多周期移动平均线组合，通过多条不同周期的移动平均线，来判断价格的趋势。
    PUCU指标通过多条移动平均线的排列，来反应价格的趋势和支撑阻力位。

    计算公式：
    PUCU1 = EMA(Close, 5)
    PUCU2 = EMA(Close, 8)
    PUCU3 = EMA(Close, 13)
    PUCU4 = EMA(Close, 21)
    PUCU5 = EMA(Close, 34)
    PUCU6 = EMA(Close, 55)

    使用场景：
    - 多条瀑布线向上排列时，表示上升趋势
    - 多条瀑布线向下排列时，表示下降趋势
    - 价格在所有瀑布线上方时，为强势多头市场
    - 价格在所有瀑布线下方时，为强势空头市场
    - 短期瀑布线上穿长期瀑布线时，为买入信号（金叉）
    - 短期瀑布线下穿长期瀑布线时，为卖出信号（死叉）

    输入参数：
    - data: DataFrame，必须包含'Close'列

    输出参数：
    - PUCU1: 瀑布线1
    - PUCU2: 瀑布线2
    - PUCU3: 瀑布线3
    - PUCU4: 瀑布线4
    - PUCU5: 瀑布线5
    - PUCU6: 瀑布线6
    - pucu_above_all: 价格在所有瀑布线上方信号
    - pucu_below_all: 价格在所有瀑布线下方信号
    - pucu_upward: 瀑布线向上排列信号
    - pucu_downward: 瀑布线向下排列信号

    注意事项：
    - PUCU是多周期趋势指标，在震荡市中会产生频繁的假信号
    - PUCU在单边市中表现最好
    - PUCU反应较慢，不适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：多条瀑布线同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的PUCU
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self):
        """
        初始化PUCU指标
        """
        pass

    def calculate(self, data):
        """
        计算PUCU指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PUCU计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['PUCU1'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['PUCU2'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['PUCU3'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['PUCU4'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['PUCU5'] = df['Close'].ewm(span=34, adjust=False).mean()
        df['PUCU6'] = df['Close'].ewm(span=55, adjust=False).mean()

        df['pucu_above_all'] = (df['Close'] > df['PUCU1']) & (df['Close'] > df['PUCU2']) & \
                               (df['Close'] > df['PUCU3']) & (df['Close'] > df['PUCU4']) & \
                               (df['Close'] > df['PUCU5']) & (df['Close'] > df['PUCU6'])
        df['pucu_below_all'] = (df['Close'] < df['PUCU1']) & (df['Close'] < df['PUCU2']) & \
                               (df['Close'] < df['PUCU3']) & (df['Close'] < df['PUCU4']) & \
                               (df['Close'] < df['PUCU5']) & (df['Close'] < df['PUCU6'])
        df['pucu_upward'] = (df['PUCU1'] > df['PUCU2']) & (df['PUCU2'] > df['PUCU3']) & \
                            (df['PUCU3'] > df['PUCU4']) & (df['PUCU4'] > df['PUCU5']) & (df['PUCU5'] > df['PUCU6'])
        df['pucu_downward'] = (df['PUCU1'] < df['PUCU2']) & (df['PUCU2'] < df['PUCU3']) & \
                              (df['PUCU3'] < df['PUCU4']) & (df['PUCU4'] < df['PUCU5']) & (df['PUCU5'] < df['PUCU6'])

        return df[['PUCU1', 'PUCU2', 'PUCU3', 'PUCU4', 'PUCU5', 'PUCU6',
                   'pucu_above_all', 'pucu_below_all',
                   'pucu_upward', 'pucu_downward']]
