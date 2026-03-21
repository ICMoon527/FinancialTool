import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DMAExtended(BaseIndicator):
    """
    DMAExtended（平均线差扩展），中文称为平均线差扩展。

    DMAExtended指标是DMA指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    DMAExtended指标结合了快速移动平均和慢速移动平均，来反应价格的趋势。

    计算公式：
    1. 计算MA1：MA1 = EMA(Close, N)
    2. 计算MA2：MA2 = EMA(Close, M)
    3. 计算DMA：DMA = MA1 - MA2
    4. 计算AMA：AMA = EMA(DMA, P)
    5. 计算DMA_OSC：DMA_OSC = (DMA - AMA) × 100

    使用场景：
    - DMA > 0 时，表示多头市场
    - DMA < 0 时，表示空头市场
    - DMA上升时，表示上升趋势
    - DMA下降时，表示下降趋势
    - DMA上穿AMA时，为买入信号（金叉）
    - DMA下穿AMA时，为卖出信号（死叉）
    - DMA_OSC > 0 时，表示量价配合良好
    - DMA_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period1: 快速MA周期，默认10
    - period2: 慢速MA周期，默认50
    - period3: AMA周期，默认10

    输出参数：
    - MA1: 快速移动平均线
    - MA2: 慢速移动平均线
    - DMA: 平均线差
    - AMA: AMA线
    - DMA_OSC: DMA的震荡指标
    - dma_positive: DMA > 0信号
    - dma_negative: DMA < 0信号
    - dma_rising: DMA上升信号
    - dma_falling: DMA下降信号
    - dma_above_ama: DMA在AMA上方信号
    - dma_below_ama: DMA在AMA下方信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - DMAExtended是趋势指标，在单边市中表现最好
    - DMAExtended在震荡市中会产生频繁的假信号
    - DMAExtended反应较慢，不适合短线操作
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：DMA和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DMAExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period1=10, period2=50, period3=10):
        """
        初始化DMAExtended指标

        Parameters
        ----------
        period1 : int, default 10
            快速MA周期
        period2 : int, default 50
            慢速MA周期
        period3 : int, default 10
            AMA周期
        """
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3

    def calculate(self, data):
        """
        计算DMAExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DMAExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA1'] = df['Close'].ewm(span=self.period1, adjust=False).mean()
        df['MA2'] = df['Close'].ewm(span=self.period2, adjust=False).mean()
        df['DMA'] = df['MA1'] - df['MA2']
        df['AMA'] = df['DMA'].ewm(span=self.period3, adjust=False).mean()
        df['DMA_OSC'] = (df['DMA'] - df['AMA']) * 100

        df['dma_positive'] = df['DMA'] > 0
        df['dma_negative'] = df['DMA'] < 0
        df['dma_rising'] = df['DMA'] > df['DMA'].shift(1)
        df['dma_falling'] = df['DMA'] < df['DMA'].shift(1)
        df['dma_above_ama'] = df['DMA'] > df['AMA']
        df['dma_below_ama'] = df['DMA'] < df['AMA']
        df['golden_cross'] = (df['DMA'] > df['AMA']) & (df['DMA'].shift(1) <= df['AMA'].shift(1))
        df['death_cross'] = (df['DMA'] < df['AMA']) & (df['DMA'].shift(1) >= df['AMA'].shift(1))

        return df[['MA1', 'MA2', 'DMA', 'AMA', 'DMA_OSC',
                   'dma_positive', 'dma_negative',
                   'dma_rising', 'dma_falling',
                   'dma_above_ama', 'dma_below_ama',
                   'golden_cross', 'death_cross']]
