import pandas as pd
import numpy as np
from ..base import BaseIndicator


class Ichimoku(BaseIndicator):
    """
    Ichimoku Kinko Hyo（一目均衡表），中文称为一目均衡表。

    Ichimoku Kinko Hyo指标是由日本记者Goichi Hosoda提出的，是一种综合的趋势指标。
    Ichimoku Kinko Hyo指标通过计算多个移动平均，来判断价格的趋势和支撑阻力位。

    计算公式：
    1. 计算Tenkan-sen（转换线）：Tenkan = (High_9 + Low_9) / 2
    2. 计算Kijun-sen（基准线）：Kijun = (High_26 + Low_26) / 2
    3. 计算Senkou Span A（先行线A）：Senkou_A = (Tenkan + Kijun) / 2，向前26期
    4. 计算Senkou Span B（先行线B）：Senkou_B = (High_52 + Low_52) / 2，向前26期
    5. 计算Chikou Span（迟行线）：Chikou = Close，向后26期

    使用场景：
    - 价格在云上方时，为多头市场
    - 价格在云下方时，为空头市场
    - 转换线上穿基准线时，为买入信号（金叉）
    - 转换线下穿基准线时，为卖出信号（死叉）
    - 迟行线在价格上方时，为买入信号
    - 迟行线在价格下方时，为卖出信号
    - 云向上时，表示上升趋势
    - 云向下时，表示下降趋势

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - tenkan_period: 转换线周期，默认9
    - kijun_period: 基准线周期，默认26
    - senkou_period: 先行线周期，默认52
    - shift_period: 移动周期，默认26

    输出参数：
    - Tenkan: 转换线
    - Kijun: 基准线
    - Senkou_A: 先行线A
    - Senkou_B: 先行线B
    - Chikou: 迟行线
    - cloud_top: 云顶
    - cloud_bottom: 云底
    - price_above_cloud: 价格在云上方信号
    - price_below_cloud: 价格在云下方信号
    - tenkan_above_kijun: 转换线在基准线上方信号
    - tenkan_below_kijun: 转换线在基准线下方信号
    - tenkan_cross_kijun_up: 转换线上穿基准线信号
    - tenkan_cross_kijun_down: 转换线下穿基准线信号
    - chikou_above_price: 迟行线在价格上方信号
    - chikou_below_price: 迟行线在价格下方信号
    - cloud_rising: 云上升信号
    - cloud_falling: 云下降信号

    注意事项：
    - Ichimoku是综合趋势指标，包含多个信号
    - Ichimoku反应较慢，不适合短线操作
    - Ichimoku在单边市中表现最好
    - 周期参数是基于日本市场设定的，可以调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：多个信号同时出现更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的Ichimoku
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场特性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, tenkan_period=9, kijun_period=26, senkou_period=52, shift_period=26):
        """
        初始化Ichimoku Kinko Hyo指标

        Parameters
        ----------
        tenkan_period : int, default 9
            转换线周期
        kijun_period : int, default 26
            基准线周期
        senkou_period : int, default 52
            先行线周期
        shift_period : int, default 26
            移动周期
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_period = senkou_period
        self.shift_period = shift_period

    def calculate(self, data):
        """
        计算Ichimoku Kinko Hyo指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含Ichimoku Kinko Hyo计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['Tenkan'] = (df['High'].rolling(window=self.tenkan_period).max() +
                       df['Low'].rolling(window=self.tenkan_period).min()) / 2
        df['Kijun'] = (df['High'].rolling(window=self.kijun_period).max() +
                      df['Low'].rolling(window=self.kijun_period).min()) / 2
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(self.shift_period)
        df['Senkou_B'] = ((df['High'].rolling(window=self.senkou_period).max() +
                          df['Low'].rolling(window=self.senkou_period).min()) / 2).shift(self.shift_period)
        df['Chikou'] = df['Close'].shift(-self.shift_period)

        df['cloud_top'] = df[['Senkou_A', 'Senkou_B']].max(axis=1)
        df['cloud_bottom'] = df[['Senkou_A', 'Senkou_B']].min(axis=1)
        df['price_above_cloud'] = df['Close'] > df['cloud_top']
        df['price_below_cloud'] = df['Close'] < df['cloud_bottom']
        df['tenkan_above_kijun'] = df['Tenkan'] > df['Kijun']
        df['tenkan_below_kijun'] = df['Tenkan'] < df['Kijun']
        df['tenkan_cross_kijun_up'] = (df['Tenkan'] > df['Kijun']) & (df['Tenkan'].shift(1) <= df['Kijun'].shift(1))
        df['tenkan_cross_kijun_down'] = (df['Tenkan'] < df['Kijun']) & (df['Tenkan'].shift(1) >= df['Kijun'].shift(1))
        df['chikou_above_price'] = df['Chikou'] > df['Close'].shift(self.shift_period)
        df['chikou_below_price'] = df['Chikou'] < df['Close'].shift(self.shift_period)
        df['cloud_rising'] = df['Senkou_A'] > df['Senkou_A'].shift(1)
        df['cloud_falling'] = df['Senkou_A'] < df['Senkou_A'].shift(1)

        return df[['Tenkan', 'Kijun', 'Senkou_A', 'Senkou_B', 'Chikou',
                   'cloud_top', 'cloud_bottom',
                   'price_above_cloud', 'price_below_cloud',
                   'tenkan_above_kijun', 'tenkan_below_kijun',
                   'tenkan_cross_kijun_up', 'tenkan_cross_kijun_down',
                   'chikou_above_price', 'chikou_below_price',
                   'cloud_rising', 'cloud_falling']]
