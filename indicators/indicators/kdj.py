import numpy as np
import pandas as pd

from ..base import BaseIndicator


class KDJ(BaseIndicator):
    """
    KDJ（随机指标）指标，由乔治·莱恩（George Lane）于1950年代提出，是期货和股票市场常用的技术分析工具。

    KDJ指标属于超买超卖类指标，主要用于衡量价格的波动幅度，通过特定周期内出现过的最高价、最低价及当日收盘价，
    计算出未成熟随机值RSV，然后根据平滑移动平均线的方法来计算K值、D值与J值，并绘成曲线图来研判股票走势。

    计算公式：
    1. 计算未成熟随机值RSV（Raw Stochastic Value）：
       - RSV = (今日收盘价 - N日内最低价) / (N日内最高价 - N日内最低价) × 100
       - 通常N=9
    2. 计算K值（快速确认线）：
       - K = SMA(RSV, M1)
       - 通常M1=3
    3. 计算D值（慢速确认线）：
       - D = SMA(K, M2)
       - 通常M2=3
    4. 计算J值（辅助线）：
       - J = 3 × K - 2 × D

    使用场景：
    - 超买区：K、D值在80以上，J值在100以上，可能见顶
    - 超卖区：K、D值在20以下，J值在0以下，可能见底
    - 金叉：K上穿D，且在低位（如20以下），为买入信号
    - 死叉：K下穿D，且在高位（如80以上），为卖出信号
    - 顶背离：价格创新高但K、D未创新高，可能见顶
    - 底背离：价格创新低但K、D未创新低，可能见底
    - J值：J值为正时K大于D，J值为负时K小于D

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - n_period: RSV计算周期，默认9
    - m1_period: K值平滑周期，默认3
    - m2_period: D值平滑周期，默认3

    输出参数：
    - RSV: 未成熟随机值
    - K: 快速确认线
    - D: 慢速确认线
    - J: 辅助线
    - overbought: 超买信号
    - oversold: 超卖信号
    - golden_cross: 金叉信号
    - death_cross: 死叉信号

    注意事项：
    - KDJ是震荡指标，在单边趋势市中容易钝化
    - KDJ在20-80之间的交叉信号可能不准确，建议只在超买超卖区使用
    - 不同市场参数应调整：短线可设为6-3-3，长线可设为14-3-3
    - J值可以为负或超过100，这是正常现象
    - 钝化问题：在极端行情中，KDJ可能长期在超买超卖区，此时应减少使用

    最佳实践建议：
    1. 参数适配：根据不同市场调整参数，A股常用9-3-3
    2. 背离确认：背离后等待二次确认再行动，避免假背离
    3. 多周期共振：日线KDJ+周线KDJ同时金叉，成功率更高
    4. 成交量配合：KDJ金叉时成交量放大，信号更可靠
    5. 趋势优先：先判断大趋势，顺势而为，逆势信号忽略
    6. 布林带结合：KDJ+BOLL，在布林带上下轨的KDJ信号更准确
    7. 分批操作：不要因一次KDJ信号就满仓，分批进场更安全
    """

    def __init__(self, n_period=9, m1_period=3, m2_period=3):
        """
        初始化KDJ指标参数。

        Args:
            n_period: RSV计算周期，默认9
            m1_period: K值平滑周期，默认3
            m2_period: D值平滑周期，默认3
        """
        self.n_period = n_period
        self.m1_period = m1_period
        self.m2_period = m2_period

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算简单移动平均线（SMA）。

        Args:
            data: 输入序列
            period: 周期参数

        Returns:
            SMA序列
        """
        return data.rolling(window=period, min_periods=1).mean()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算最低低值（Lowest Low Value）。

        Args:
            data: 输入序列
            period: 周期参数

        Returns:
            周期内最低值序列
        """
        return data.rolling(window=period, min_periods=1).min()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算最高高值（Highest High Value）。

        Args:
            data: 输入序列
            period: 周期参数

        Returns:
            周期内最高值序列
        """
        return data.rolling(window=period, min_periods=1).max()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算KDJ指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame，必须包含'High'、'Low'、'Close'列

        Returns:
            添加了KDJ相关列的DataFrame，包括：
            - RSV: 未成熟随机值
            - K: 快速确认线
            - D: 慢速确认线
            - J: 辅助线
            - overbought: 超买信号
            - oversold: 超卖信号
            - golden_cross: 金叉信号
            - death_cross: 死叉信号
        """
        self.validate_input(data)

        result = data.copy()

        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        llv_low_n = self._llv(low, self.n_period)
        hhv_high_n = self._hhv(high, self.n_period)

        denominator = hhv_high_n - llv_low_n
        denominator = denominator.replace(0, np.nan)
        RSV = (close - llv_low_n) / denominator * 100

        K = self._sma(RSV, self.m1_period)
        D = self._sma(K, self.m2_period)
        J = 3 * K - 2 * D

        overbought = (K >= 80) | (D >= 80)
        oversold = (K <= 20) | (D <= 20)

        golden_cross = (K > D) & (K.shift(1) <= D.shift(1))
        death_cross = (K < D) & (K.shift(1) >= D.shift(1))

        result["RSV"] = RSV
        result["K"] = K
        result["D"] = D
        result["J"] = J
        result["overbought"] = overbought
        result["oversold"] = oversold
        result["golden_cross"] = golden_cross
        result["death_cross"] = death_cross

        return result
