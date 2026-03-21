import pandas as pd
import numpy as np
from ..base import BaseIndicator


class PBX(BaseIndicator):
    """
    PBX（瀑布线），中文称为瀑布线。

    PBX指标是一种多周期移动平均线组合，通过多条不同周期的移动平均线，来判断价格的趋势。
    PBX指标通过多条移动平均线的排列，来反应价格的趋势和支撑阻力位。

    计算公式：
    PBX1 = EMA(Close, 4)
    PBX2 = EMA(Close, 6)
    PBX3 = EMA(Close, 9)
    PBX4 = EMA(Close, 13)
    PBX5 = EMA(Close, 18)
    PBX6 = EMA(Close, 24)

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
    - PBX1: 瀑布线1
    - PBX2: 瀑布线2
    - PBX3: 瀑布线3
    - PBX4: 瀑布线4
    - PBX5: 瀑布线5
    - PBX6: 瀑布线6
    - pbx_above_all: 价格在所有瀑布线上方信号
    - pbx_below_all: 价格在所有瀑布线下方信号
    - pbx_upward: 瀑布线向上排列信号
    - pbx_downward: 瀑布线向下排列信号

    注意事项：
    - PBX是多周期趋势指标，在震荡市中会产生频繁的假信号
    - PBX在单边市中表现最好
    - PBX反应较慢，不适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：多条瀑布线同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的PBX
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self):
        """
        初始化PBX指标
        """
        pass

    def calculate(self, data):
        """
        计算PBX指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含PBX计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['PBX1'] = df['Close'].ewm(span=4, adjust=False).mean()
        df['PBX2'] = df['Close'].ewm(span=6, adjust=False).mean()
        df['PBX3'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['PBX4'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['PBX5'] = df['Close'].ewm(span=18, adjust=False).mean()
        df['PBX6'] = df['Close'].ewm(span=24, adjust=False).mean()

        df['pbx_above_all'] = (df['Close'] > df['PBX1']) & (df['Close'] > df['PBX2']) & \
                             (df['Close'] > df['PBX3']) & (df['Close'] > df['PBX4']) & \
                             (df['Close'] > df['PBX5']) & (df['Close'] > df['PBX6'])
        df['pbx_below_all'] = (df['Close'] < df['PBX1']) & (df['Close'] < df['PBX2']) & \
                             (df['Close'] < df['PBX3']) & (df['Close'] < df['PBX4']) & \
                             (df['Close'] < df['PBX5']) & (df['Close'] < df['PBX6'])
        df['pbx_upward'] = (df['PBX1'] > df['PBX2']) & (df['PBX2'] > df['PBX3']) & \
                          (df['PBX3'] > df['PBX4']) & (df['PBX4'] > df['PBX5']) & (df['PBX5'] > df['PBX6'])
        df['pbx_downward'] = (df['PBX1'] < df['PBX2']) & (df['PBX2'] < df['PBX3']) & \
                            (df['PBX3'] < df['PBX4']) & (df['PBX4'] < df['PBX5']) & (df['PBX5'] < df['PBX6'])

        return df[['PBX1', 'PBX2', 'PBX3', 'PBX4', 'PBX5', 'PBX6',
                   'pbx_above_all', 'pbx_below_all',
                   'pbx_upward', 'pbx_downward']]
