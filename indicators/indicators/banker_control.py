import pandas as pd

from indicators.base import BaseIndicator


class BankerControl(BaseIndicator):
    """
    庄家控盘指标

    该指标基于EMA计算和价格基准来测量庄家对股票价格走势的控盘程度。

    公式：
    - AAA：(3 * Close + Open + High + Low) / 6（价格基准）
    - MA12：AAA的12周期EMA
    - MA36：AAA的36周期EMA
    - 控盘度：(MA12 - REF(MA36, 1)) / REF(MA36, 1) * 100 + 50

    控盘级别：
    - 低控盘：control_degree >= 50
    - 中控盘：control_degree >= 60
    - 高控盘：control_degree >= 80
    """

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算指数移动平均（EMA）。

        Args:
            data: 输入序列
            period: EMA周期

        Returns:
            EMA值
        """
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算庄家控盘指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            包含'control_degree', 'low_control', 'medium_control', 'high_control'列的DataFrame
        """
        self.validate_input(data)

        close = data["Close"].copy()
        open_ = data["Open"].copy()
        high = data["High"].copy()
        low = data["Low"].copy()

        aaa = (3 * close + open_ + high + low) / 6

        ma12 = self._ema(aaa, 12)
        ma36 = self._ema(aaa, 36)
        ma36_prev = ma36.shift(1)

        control_degree = (ma12 - ma36_prev) / ma36_prev * 100 + 50

        low_control = (control_degree >= 50).astype(int)
        medium_control = (control_degree >= 60).astype(int)
        high_control = (control_degree >= 80).astype(int)

        result = data.copy()
        result["control_degree"] = control_degree
        result["low_control"] = low_control
        result["medium_control"] = medium_control
        result["high_control"] = high_control

        return result
