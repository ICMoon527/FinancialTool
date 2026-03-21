import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MACDCross(BaseIndicator):
    """
    MACD金叉死叉指标，中文称为MACD金叉死叉指标。

    MACD金叉死叉指标是基于MACD的技术指标，通过计算DIF和DEA的交叉，
    来判断价格趋势的变化。金叉表示DIF上穿DEA，为买入信号；死叉表示DIF下穿DEA，为卖出信号。

    计算公式：
    1. 计算快速EMA：EMA_Fast = EMA(Close, Fast_Period)
    2. 计算慢速EMA：EMA_Slow = EMA(Close, Slow_Period)
    3. 计算DIF（差离值）：DIF = EMA_Fast - EMA_Slow
    4. 计算DEA（讯号线）：DEA = EMA(DIF, Signal_Period)
    5. 计算MACD柱状图：MACD_Bar = 2 × (DIF - DEA)
    6. 判断金叉：DIF上穿DEA
    7. 判断死叉：DIF下穿DEA

    使用场景：
    - 金叉出现时，为买入信号
    - 死叉出现时，为卖出信号
    - DIF在DEA上方时，为多头市场
    - DIF在DEA下方时，为空头市场
    - 金叉后DIF和DEA同时向上时，趋势健康
    - 死叉后DIF和DEA同时向下时，趋势健康
    - MACD柱状图由负转正时，为买入信号
    - MACD柱状图由正转负时，为卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号线周期，默认9

    输出参数：
    - EMA_Fast: 快速EMA
    - EMA_Slow: 慢速EMA
    - DIF: 差离值
    - DEA: 讯号线
    - MACD_Bar: MACD柱状图
    - golden_cross: 金叉信号
    - death_cross: 死叉信号
    - dif_above_dea: DIF在DEA上方信号
    - dif_below_dea: DIF在DEA下方信号
    - macd_bar_positive: MACD柱状图正信号
    - macd_bar_negative: MACD柱状图负信号

    注意事项：
    - MACD金叉死叉是趋势指标，在震荡市中会产生频繁的假信号
    - MACD金叉死叉在单边市中表现最好
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如RSI、KDJ）一起使用
    - MACD金叉死叉反应较慢，不适合短线操作

    最佳实践建议：
    1. 趋势确认：DIF和DEA同时向上更可靠
    2. 组合使用：与RSI等震荡指标配合使用
    3. 多周期使用：同时使用日线和周线的MACD金叉死叉
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：金叉死叉后等待2-3个交易日确认
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化MACD金叉死叉指标

        Parameters
        ----------
        fast_period : int, default 12
            快速EMA周期
        slow_period : int, default 26
            慢速EMA周期
        signal_period : int, default 9
            信号线周期
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data):
        """
        计算MACD金叉死叉指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含MACD金叉死叉计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['EMA_Fast'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()

        df['DIF'] = df['EMA_Fast'] - df['EMA_Slow']
        df['DEA'] = df['DIF'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_Bar'] = 2 * (df['DIF'] - df['DEA'])

        df['golden_cross'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
        df['death_cross'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))
        df['dif_above_dea'] = df['DIF'] > df['DEA']
        df['dif_below_dea'] = df['DIF'] < df['DEA']
        df['macd_bar_positive'] = df['MACD_Bar'] > 0
        df['macd_bar_negative'] = df['MACD_Bar'] < 0

        return df[['EMA_Fast', 'EMA_Slow', 'DIF', 'DEA', 'MACD_Bar', 'golden_cross', 
                   'death_cross', 'dif_above_dea', 'dif_below_dea', 
                   'macd_bar_positive', 'macd_bar_negative']]
