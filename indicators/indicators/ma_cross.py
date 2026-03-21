import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MACross(BaseIndicator):
    """
    MA（Moving Average）金叉死叉指标，中文称为移动平均线金叉死叉指标。

    MA金叉死叉指标是基于移动平均线的技术指标，通过计算短期移动平均线和长期移动平均线的交叉，
    来判断价格趋势的变化。金叉表示短期均线上穿长期均线，为买入信号；死叉表示短期均线下穿长期均线，为卖出信号。

    计算公式：
    1. 计算短期移动平均线：MA_Short = MA(Close, Short_Period)
    2. 计算长期移动平均线：MA_Long = MA(Close, Long_Period)
    3. 判断金叉：MA_Short上穿MA_Long
    4. 判断死叉：MA_Short下穿MA_Long

    使用场景：
    - 金叉出现时，为买入信号
    - 死叉出现时，为卖出信号
    - MA_Short在MA_Long上方时，为多头市场
    - MA_Short在MA_Long下方时，为空头市场
    - 金叉后MA_Short和MA_Long同时向上时，趋势健康
    - 死叉后MA_Short和MA_Long同时向下时，趋势健康

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - short_period: 短期均线周期，默认5
    - long_period: 长期均线周期，默认20
    - ma_type: 均线类型，'SMA'表示简单移动平均，'EMA'表示指数移动平均，默认'SMA'

    输出参数：
    - MA_Short: 短期移动平均线
    - MA_Long: 长期移动平均线
    - golden_cross: 金叉信号
    - death_cross: 死叉信号
    - ma_short_above: MA_Short在MA_Long上方信号
    - ma_short_below: MA_Short在MA_Long下方信号

    注意事项：
    - MA金叉死叉是趋势指标，在震荡市中会产生频繁的假信号
    - MA金叉死叉在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - MA金叉死叉反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：MA_Short和MA_Long同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的MA金叉死叉
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：金叉死叉后等待2-3个交易日确认
    """

    def __init__(self, short_period=5, long_period=20, ma_type='SMA'):
        """
        初始化MA金叉死叉指标

        Parameters
        ----------
        short_period : int, default 5
            短期均线周期
        long_period : int, default 20
            长期均线周期
        ma_type : str, default 'SMA'
            均线类型，'SMA'表示简单移动平均，'EMA'表示指数移动平均
        """
        self.short_period = short_period
        self.long_period = long_period
        self.ma_type = ma_type

    def calculate(self, data):
        """
        计算MA金叉死叉指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MA金叉死叉计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        if self.ma_type == 'EMA':
            df['MA_Short'] = df['Close'].ewm(span=self.short_period, adjust=False).mean()
            df['MA_Long'] = df['Close'].ewm(span=self.long_period, adjust=False).mean()
        else:
            df['MA_Short'] = df['Close'].rolling(window=self.short_period).mean()
            df['MA_Long'] = df['Close'].rolling(window=self.long_period).mean()

        df['golden_cross'] = (df['MA_Short'] > df['MA_Long']) & (df['MA_Short'].shift(1) <= df['MA_Long'].shift(1))
        df['death_cross'] = (df['MA_Short'] < df['MA_Long']) & (df['MA_Short'].shift(1) >= df['MA_Long'].shift(1))
        df['ma_short_above'] = df['MA_Short'] > df['MA_Long']
        df['ma_short_below'] = df['MA_Short'] < df['MA_Long']

        return df[['MA_Short', 'MA_Long', 'golden_cross', 'death_cross', 
                   'ma_short_above', 'ma_short_below']]
