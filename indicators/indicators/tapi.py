import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TAPI(BaseIndicator):
    """
    TAPI（Trading Amount Per Index）指标，中文称为大盘加权成交量指标。

    TAPI指标是一种结合了价格和成交量的指标，通过计算成交量与收盘价的比值，
    来衡量市场的买卖力量和资金流向。

    计算公式：
    TAPI = Volume / Close

    使用场景：
    - TAPI上升时，表示成交量相对于价格上升，为买入信号
    - TAPI下降时，表示成交量相对于价格下降，为卖出信号
    - TAPI创新高时，表示资金流入，为买入信号
    - TAPI创新低时，表示资金流出，为卖出信号
    - TAPI与价格同步上升时，趋势健康
    - TAPI与价格同步下降时，趋势健康
    - TAPI与价格背离时，可能反转

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period: 移动平均周期，默认10

    输出参数：
    - TAPI: 大盘加权成交量指标
    - TAPI_MA: TAPI的移动平均线
    - tapi_rising: TAPI上升信号
    - tapi_falling: TAPI下降信号
    - tapi_new_high: TAPI创新高信号
    - tapi_new_low: TAPI创新低信号
    - tapi_above_ma: TAPI在MA上方信号
    - tapi_below_ma: TAPI在MA下方信号

    注意事项：
    - TAPI是成交量指标，反应成交量与价格的关系
    - TAPI在震荡市中会产生频繁的假信号
    - TAPI在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：TAPI和价格趋势同步更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的TAPI
    4. 背离判断：注意TAPI与价格的背离
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=10):
        """
        初始化TAPI指标

        Parameters
        ----------
        period : int, default 10
            移动平均周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算TAPI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含TAPI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TAPI'] = df['Volume'] / df['Close'].replace(0, np.nan)
        df['TAPI_MA'] = df['TAPI'].rolling(window=self.period).mean()

        df['tapi_rising'] = df['TAPI'] > df['TAPI'].shift(1)
        df['tapi_falling'] = df['TAPI'] < df['TAPI'].shift(1)
        df['tapi_new_high'] = df['TAPI'] == df['TAPI'].rolling(window=self.period).max()
        df['tapi_new_low'] = df['TAPI'] == df['TAPI'].rolling(window=self.period).min()
        df['tapi_above_ma'] = df['TAPI'] > df['TAPI_MA']
        df['tapi_below_ma'] = df['TAPI'] < df['TAPI_MA']

        return df[['TAPI', 'TAPI_MA', 'tapi_rising', 'tapi_falling', 
                   'tapi_new_high', 'tapi_new_low', 'tapi_above_ma', 'tapi_below_ma']]
