import pandas as pd
import numpy as np
from ..base import BaseIndicator


class EhlerSuperSmoother(BaseIndicator):
    """
    Ehlers Super Smoother（埃勒斯超级平滑器），中文称为埃勒斯超级平滑器。

    Ehlers Super Smoother指标是由John Ehlers提出的，是一种高级平滑移动平均。
    Ehlers Super Smoother指标通过使用二次平滑，来减少滞后性同时保持平滑效果。

    计算公式：
    1. 计算α：α = 2 / (Period + 1)
    2. 计算Filt（过滤）：Filt = α × Price + (1 - α) × Filt_prev
    3. 计算SuperSmoother：SuperSmoother = 2 × Filt - Filt_prev

    使用场景：
    - SuperSmoother上升时，表示上升趋势
    - SuperSmoother下降时，表示下降趋势
    - 价格在SuperSmoother上方时，为多头市场
    - 价格在SuperSmoother下方时，为空头市场
    - 价格上穿SuperSmoother时，为买入信号
    - 价格下穿SuperSmoother时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认10

    输出参数：
    - Filt: 过滤器
    - SuperSmoother: 超级平滑器
    - price_above_smoother: 价格在平滑器上方信号
    - price_below_smoother: 价格在平滑器下方信号
    - smoother_rising: 平滑器上升信号
    - smoother_falling: 平滑器下降信号
    - price_cross_smoother_up: 价格上穿平滑器信号
    - price_cross_smoother_down: 价格下穿平滑器信号

    注意事项：
    - Ehlers Super Smoother是高级平滑移动平均，在震荡市中会产生频繁的假信号
    - Ehlers Super Smoother在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：价格和SuperSmoother同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Ehlers Super Smoother
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化Ehlers Super Smoother指标

        Parameters
        ----------
        period : int, default 10
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算Ehlers Super Smoother指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Ehlers Super Smoother计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        alpha = 2 / (self.period + 1)
        df['Filt'] = df['Close'].ewm(span=self.period, adjust=False).mean()
        df['SuperSmoother'] = 2 * df['Filt'] - df['Filt'].shift(1)

        df['price_above_smoother'] = df['Close'] > df['SuperSmoother']
        df['price_below_smoother'] = df['Close'] < df['SuperSmoother']
        df['smoother_rising'] = df['SuperSmoother'] > df['SuperSmoother'].shift(1)
        df['smoother_falling'] = df['SuperSmoother'] < df['SuperSmoother'].shift(1)
        df['price_cross_smoother_up'] = (df['Close'] > df['SuperSmoother']) & (df['Close'].shift(1) <= df['SuperSmoother'].shift(1))
        df['price_cross_smoother_down'] = (df['Close'] < df['SuperSmoother']) & (df['Close'].shift(1) >= df['SuperSmoother'].shift(1))

        return df[['Filt', 'SuperSmoother',
                   'price_above_smoother', 'price_below_smoother',
                   'smoother_rising', 'smoother_falling',
                   'price_cross_smoother_up', 'price_cross_smoother_down']]
