import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BOLLBand(BaseIndicator):
    """
    BOLL（布林带）扩展指标，中文称为布林带扩展指标。

    BOLL扩展指标是基于布林带的技术指标，通过计算价格在布林带中的位置，
    来判断价格的超买超卖状态和趋势变化。

    计算公式：
    1. 计算MA（移动平均线）：MA = SMA(Close, N)
    2. 计算MD（标准差）：MD = Std(Close, N)
    3. 计算UP（上轨）：UP = MA + K × MD
    4. 计算DN（下轨）：DN = MA - K × MD
    5. 计算%b（布林带位置）：%b = (Close - DN) / (UP - DN)
    6. 计算带宽：Bandwidth = (UP - DN) / MA

    使用场景：
    - %b > 1 时，价格突破上轨，为超买信号，可能回落
    - %b < 0 时，价格跌破下轨，为超卖信号，可能反弹
    - 带宽扩大时，表示波动性增加，可能出现大行情
    - 带宽缩小时，表示波动性减少，可能出现突破
    - %b由下往上穿越0时，为买入信号
    - %b由上往下跌破1时，为卖出信号
    - 价格在中轨上方时，为多头市场
    - 价格在中轨下方时，为空头市场

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 计算周期，默认20
    - std_dev: 标准差倍数，默认2

    输出参数：
    - MA: 移动平均线（中轨）
    - UP: 上轨
    - DN: 下轨
    - percent_b: %b指标
    - bandwidth: 带宽
    - percent_b_overbought: %b > 1超买信号
    - percent_b_oversold: %b < 0超卖信号
    - close_above_up: 价格突破上轨信号
    - close_below_dn: 价格跌破下轨信号
    - close_above_ma: 价格在中轨上方信号
    - close_below_ma: 价格在中轨下方信号
    - bandwidth_expanding: 带宽扩大信号
    - bandwidth_shrinking: 带宽缩小信号

    注意事项：
    - BOLL是趋势指标，在震荡市中会产生频繁的假信号
    - BOLL在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用
    - BOLL反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：价格和中轨同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的BOLL
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=20, std_dev=2):
        """
        初始化BOLL扩展指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        std_dev : int, default 2
            标准差倍数
        """
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data):
        """
        计算BOLL扩展指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含BOLL扩展计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['MA'] = df['Close'].rolling(window=self.period).mean()
        md = df['Close'].rolling(window=self.period).std()

        df['UP'] = df['MA'] + self.std_dev * md
        df['DN'] = df['MA'] - self.std_dev * md

        df['percent_b'] = (df['Close'] - df['DN']) / (df['UP'] - df['DN']).replace(0, np.nan)
        df['bandwidth'] = (df['UP'] - df['DN']) / df['MA'].replace(0, np.nan)

        df['percent_b_overbought'] = df['percent_b'] > 1
        df['percent_b_oversold'] = df['percent_b'] < 0
        df['close_above_up'] = df['Close'] > df['UP']
        df['close_below_dn'] = df['Close'] < df['DN']
        df['close_above_ma'] = df['Close'] > df['MA']
        df['close_below_ma'] = df['Close'] < df['MA']
        df['bandwidth_expanding'] = df['bandwidth'] > df['bandwidth'].shift(1)
        df['bandwidth_shrinking'] = df['bandwidth'] < df['bandwidth'].shift(1)

        return df[['MA', 'UP', 'DN', 'percent_b', 'bandwidth', 
                   'percent_b_overbought', 'percent_b_oversold', 
                   'close_above_up', 'close_below_dn', 
                   'close_above_ma', 'close_below_ma', 
                   'bandwidth_expanding', 'bandwidth_shrinking']]
