import numpy as np
import pandas as pd

from ..base import BaseIndicator


class MACD(BaseIndicator):
    """
    MACD（Moving Average Convergence Divergence）指标，中文称为指数平滑异同移动平均线。

    MACD由杰拉尔·阿佩尔（Gerald Appel）于1979年提出，是一种利用短期（常用为12日）指数移动平均线与长期（常用为26日）
    指数移动平均线之间的聚合与分离状况，对买进、卖出时机作出研判的技术指标。

    计算公式：
    1. 计算快速EMA和慢速EMA：
       - EMA(12) = 前一日EMA(12) × 11/13 + 今日收盘价 × 2/13
       - EMA(26) = 前一日EMA(26) × 25/27 + 今日收盘价 × 2/27
    2. 计算DIF（差离值）：
       - DIF = EMA(12) - EMA(26)
    3. 计算DEA（讯号线）：
       - DEA = EMA(DIF, 9)
    4. 计算MACD柱状图：
       - MACD_Bar = 2 × (DIF - DEA)

    使用场景：
    - 金叉：DIF上穿DEA，为买入信号
    - 死叉：DIF下穿DEA，为卖出信号
    - 顶背离：价格创新高但DIF/DEA未创新高，可能见顶
    - 底背离：价格创新低但DIF/DEA未创新低，可能见底
    - 柱状图：柱状图由负转正，买入信号；由正转负，卖出信号

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - fast_period: 快速EMA周期，默认12
    - slow_period: 慢速EMA周期，默认26
    - signal_period: 信号线周期，默认9

    输出参数：
    - DIF: 差离值（EMA12 - EMA26）
    - DEA: 信号线（DIF的9日EMA）
    - MACD_Bar: MACD柱状图（2*(DIF-DEA)）
    - EMA12: 快速EMA
    - EMA26: 慢速EMA
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - MACD是趋势指标，在震荡市中容易产生伪信号
    - 建议结合其他指标（如RSI、KDJ）一起使用，避免单一指标决策
    - 周期参数可以根据市场特性调整，但12-26-9是最常用的组合
    - 在单边市中MACD表现最好，在震荡市中表现较差
    - 信号确认：建议等待3个交易日确认后再行动

    最佳实践建议：
    1. 多重确认：使用多个时间周期（日线+周线）的MACD信号互相确认
    2. 指标组合：结合成交量、布林带、RSI等指标过滤假信号
    3. 趋势判断：先判断大趋势，再用MACD寻找入场点
    4. 止损管理：即使MACD信号良好，也要设置止损
    5. 分批入场：金叉后不要一次性满仓，分批建仓更安全
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化MACD指标参数。

        Args:
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线周期，默认9
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算指数移动平均线（EMA）。

        EMA = 今日收盘价 × (2/(周期+1)) + 前一日EMA × (1-2/(周期+1))

        Args:
            data: 输入价格序列
            period: 周期参数

        Returns:
            EMA序列
        """
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame，必须包含'Close'列

        Returns:
            添加了MACD相关列的DataFrame，包括：
            - EMA12: 快速EMA
            - EMA26: 慢速EMA
            - DIF: 差离值
            - DEA: 信号线
            - MACD_Bar: MACD柱状图
            - golden_cross: 金叉信号
            - death_cross: 死叉信号
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]

        ema_fast = self._ema(close, self.fast_period)
        ema_slow = self._ema(close, self.slow_period)

        DIF = ema_fast - ema_slow

        DEA = self._ema(DIF, self.signal_period)

        MACD_Bar = 2 * (DIF - DEA)

        golden_cross = (DIF > DEA) & (DIF.shift(1) <= DEA.shift(1))
        death_cross = (DIF < DEA) & (DIF.shift(1) >= DEA.shift(1))

        result[f"EMA{self.fast_period}"] = ema_fast
        result[f"EMA{self.slow_period}"] = ema_slow
        result["DIF"] = DIF
        result["DEA"] = DEA
        result["MACD_Bar"] = MACD_Bar
        result["golden_cross"] = golden_cross
        result["death_cross"] = death_cross

        return result
