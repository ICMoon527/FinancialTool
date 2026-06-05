import numpy as np
import pandas as pd

from indicators.base import BaseIndicator


class Tiandao(BaseIndicator):
    """
    天道指标 - 完整版（趋势线 + 买入信号 + 资金流 + 金钻起涨）

    该指标包含：

    趋势线（4条）：
    - td_jinniu (金牛，红色): 2 * XMA²(H, n) - XMA²(L, n)，通道上轨
    - td_jinzuan (金钻趋势，金色): 2 * XMA²(L, n) - XMA²(H, n)，核心趋势线
    - td_jinniu2 (金牛2，绿色): EMA(金钻趋势, n)，慢速跟随线
    - td_bbi (BBI): (MA(C,3) + MA(C,6) + MA(C,12) + MA(C,24)) / 4

    买入信号（2个）：
    - td_xg (▲买入): 金钻趋势 > HIGH AND 回调买 AND L <= 金钻趋势
    - td_xg2 (↖金钻起涨): C>O AND DY2<0.02 AND MA(C,5)>MA(C,60) AND C/REF(C,1)>=1.02 AND H<金牛

    资金流指标：
    - td_ddx, td_v2, td_v5, td_v10, td_v20

    参数：
    - n: XMA/EMA 周期（默认 25）
    - n1/n2/n3/n4: BBI 各均线周期（默认 3/6/12/24）
    """

    def __init__(self, n: int = 25, n1: int = 3, n2: int = 6, n3: int = 12, n4: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

    # ==================== 通达信函数实现 ====================

    @staticmethod
    def _xma(series: pd.Series, n: int) -> pd.Series:
        """
        通达信 XMA 偏移移动平均（纯递推实现，无未来函数）。

        递推公式: XMA_t = (X_t + (n-1) * XMA_{t-1}) / n
        alpha = 1/n，权重衰减慢，均线更平滑，适合捕捉中期趋势。
        """
        values = series.values.astype(np.float64)
        result = np.empty_like(values)
        result[0] = values[0]
        alpha = 1.0 / n
        for i in range(1, len(values)):
            result[i] = (1.0 - alpha) * result[i - 1] + alpha * values[i]
        return pd.Series(result, index=series.index)

    @staticmethod
    def _sma(series: pd.Series, n: int, m: int) -> pd.Series:
        """
        通达信 SMA 加权移动平均。

        递推公式: SMA_t = (X_t * m + SMA_{t-1} * (n - m)) / n
        alpha = m/n
        """
        alpha = float(m) / n
        return series.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
        """
        通达信 CROSS(A, B): A 上穿 B。
        条件: A_{t-1} <= B_{t-1} AND A_t > B_t
        """
        return (a.shift(1) <= b.shift(1)) & (a > b)

    @staticmethod
    def _llv(series: pd.Series, n: int) -> pd.Series:
        """通达信 LLV(X, N): N 周期内最低值。"""
        return series.rolling(n).min()

    @staticmethod
    def _count(condition: pd.Series, n: int) -> pd.Series:
        """通达信 COUNT(condition, N): N 周期内满足条件的次数。"""
        return condition.rolling(n).sum()

    # ==================== 主计算 ====================

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算完整天道指标。"""
        self.validate_input(data)

        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        open_ = data["Open"]
        vol = data.get("Volume", pd.Series(0, index=data.index))

        n = self.n

        result = data.copy()

        # ========== 第一部分：四条趋势线 ==========

        # BBI: 四条均线的均值
        bbi = (
            close.rolling(self.n1).mean()
            + close.rolling(self.n2).mean()
            + close.rolling(self.n3).mean()
            + close.rolling(self.n4).mean()
        ) / 4.0

        # 两次 XMA 平滑: XMA(XMA(price, n), n)
        xma_high_2 = self._xma(self._xma(high, n), n)
        xma_low_2 = self._xma(self._xma(low, n), n)

        # 金钻趋势: 2 * XMA²(L, n) - XMA²(H, n)
        jinzuan = 2 * xma_low_2 - xma_high_2

        # 金牛: 2 * XMA²(H, n) - XMA²(L, n)
        jinniu = 2 * xma_high_2 - xma_low_2

        # 金牛2: EMA(金钻趋势, n)
        jinniu2 = jinzuan.ewm(span=n, adjust=False).mean()

        # ========== 第二部分：VAR23 回调买 & XG 买入信号 ==========

        # VAR23 = 100 * XMA(XMA(C-REF(C,1), 6), 6) / XMA(XMA(ABS(C-REF(C,1)), 6), 6)
        delta = close - close.shift(1)
        var23_num = self._xma(self._xma(delta, 6), 6)
        var23_den = self._xma(self._xma(delta.abs(), 6), 6)
        var23 = 100 * var23_num / var23_den.replace(0, np.nan)

        # 回调买 = LLV(VAR23, 2) == LLV(VAR23, 7) AND COUNT(VAR23 < 0, 2) AND CROSS(VAR23, MA(VAR23, 2))
        llv2 = self._llv(var23, 2)
        llv7 = self._llv(var23, 7)
        count_neg = self._count(var23 < 0, 2)
        ma_var23_2 = var23.rolling(2).mean()
        cross_var23 = self._cross(var23, ma_var23_2)
        hui_diao_mai = (llv2 == llv7) & (count_neg >= 1) & cross_var23

        # XG = 金钻趋势 > HIGH AND 回调买 AND L <= 金钻趋势
        xg = (jinzuan > high) & hui_diao_mai & (low <= jinzuan)

        # ========== 第三部分：DDX 资金流 ==========

        # 检测是否有 CAPITAL（流通股本）字段，无则按 CAPITAL=0 处理
        has_capital = "Capital" in data.columns and data["Capital"].notna().any()

        jj = (high + low + close) / 3.0
        qj0 = vol / np.where(high == low, 4, high - low)

        min_co = np.minimum(close, open_)
        max_co = np.maximum(close, open_)

        if has_capital:
            capital = data["Capital"]
            qj1 = np.where(
                high == low,
                qj0,
                qj0 * (min_co - low),
            )
            qj2 = np.where(
                high == low,
                qj0,
                qj0 * (jj - np.minimum(close, open_)),
            )
            qj3 = np.where(
                high == low,
                qj0,
                qj0 * (high - max_co),
            )
            qj4 = np.where(
                high == low,
                qj0,
                qj0 * (max_co - jj),
            )
        else:
            # CAPITAL = 0 分支
            qj1 = qj0 * (jj - min_co)  # IF(CAPITAL=0, QJ0*(JJ-MIN(CLOSE,OPEN)), ...)
            qj2 = qj0 * (min_co - low)  # IF(CAPITAL=0, QJ0*(MIN(OPEN,CLOSE)-LOW), ...)
            qj3 = qj0 * (high - max_co)  # IF(CAPITAL=0, QJ0*(HIGH-MAX(OPEN,CLOSE)), ...)
            qj4 = qj0 * (max_co - jj)  # IF(CAPITAL=0, QJ0*(MAX(CLOSE,OPEN)-JJ), ...)

        ddx = ((qj1 + qj2) - (qj3 + qj4)) / 10000.0

        # V2 = SMA(IF(C >= REF(C,1), DDX, -DDX/100), 2, 1)
        v2_input = pd.Series(np.where(close >= close.shift(1), ddx, -ddx / 100.0), index=data.index)
        v2 = self._sma(v2_input, 2, 1)

        # V5 = SMA(V2 * 120 / FROMOPEN * 5, 2, 1)
        # 日线数据 FROMOPEN = 240（全天交易分钟数）
        fromopen = 240
        v5_input = v2 * 120.0 / fromopen * 5.0
        v5 = self._sma(v5_input, 2, 1)

        # V10 = SMA(V5, 5, 1)
        v10 = self._sma(v5, 5, 1)

        # V20 = SMA(V10, 5, 1)
        v20 = self._sma(v10, 5, 1)

        # ========== 第四部分：XG2 金钻起涨信号 ==========

        # DY = CURRBARSCOUNT == 1 AND C < REF(C, 1)
        # CURRBARSCOUNT: 从最后一根 K 线向前计数，最后一根为 1
        currbarscount = pd.Series(range(len(data), 0, -1), index=data.index)
        dy = (currbarscount == 1) & (close < close.shift(1))

        # DY2 = REF(V2, 1) - DY
        dy2 = v2.shift(1) - dy.astype(float)

        # XG2 = C > O AND DY2 < 0.02 AND MA(C, 5) > MA(C, 60) AND C/REF(C, 1) >= 1.02 AND H < 金牛
        ma5 = close.rolling(5).mean()
        ma60 = close.rolling(60).mean()
        xg2 = (
            (close > open_)
            & (dy2 < 0.02)
            & (ma5 > ma60)
            & (close / close.shift(1) >= 1.02)
            & (high < jinniu)
        )

        # ========== 输出所有列 ==========

        # 趋势线
        result["td_bbi"] = bbi
        result["td_jinzuan"] = jinzuan
        result["td_jinniu"] = jinniu
        result["td_jinniu2"] = jinniu2

        # 买入信号
        result["td_var23"] = var23
        result["td_huidiaomai"] = hui_diao_mai.astype(int)
        result["td_xg"] = xg.astype(int)

        # 资金流
        result["td_ddx"] = ddx
        result["td_v2"] = v2
        result["td_v5"] = v5
        result["td_v10"] = v10
        result["td_v20"] = v20

        # 金钻起涨
        result["td_xg2"] = xg2.astype(int)

        return result