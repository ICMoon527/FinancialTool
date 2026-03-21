import numpy as np
import pandas as pd

from indicators.base import BaseIndicator


class MainTrading(BaseIndicator):
    """
    主力操盘指标

    该指标提供三条线（攻击线、操盘线、防守线），用于识别市场趋势，并根据它们的聚合和发散模式生成买卖信号。

    公式：
    - EMA26：收盘价的26周期指数移动平均
    - A：EMA26 + ABS(EMA26 - REF(EMA26, 1))
    - B：EMA26 + EMA26 - REF(EMA26, 1)
    - 操盘线：IF(EMA26 < B, B, EMA26)
    - 防守线：操盘线 - (EMA6 - 操盘线)
    - 攻击线：EMA6（6周期指数移动平均）

    信号：
    - 买入信号：三条线向上发散（攻击线 > 操盘线 > 防守线且三条线都在上升）
    - 卖出信号：三条线向下发散（攻击线 < 操盘线 < 防守线且三条线都在下降）
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
        计算主力操盘指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'attack_line', 'trading_line', 'defense_line', 'buy_signal', 'sell_signal'列的DataFrame
        """
        self.validate_input(data)

        close = data["Close"].copy()

        ema6 = self._ema(close, 6)
        ema26 = self._ema(close, 26)

        ema26_prev = ema26.shift(1)
        ema26_diff = ema26 - ema26_prev

        A = ema26 + ema26_diff.abs()
        B = ema26 + ema26_diff

        trading_line = np.where(ema26 < B, B, ema26)
        trading_line = pd.Series(trading_line, index=data.index)

        defense_line = trading_line - (ema6 - trading_line)

        attack_line = ema6

        attack_line_prev = attack_line.shift(1)
        trading_line_prev = trading_line.shift(1)
        defense_line_prev = defense_line.shift(1)

        is_rising = (attack_line > attack_line_prev) & (trading_line > trading_line_prev) & (defense_line > defense_line_prev)
        is_falling = (attack_line < attack_line_prev) & (trading_line < trading_line_prev) & (defense_line < defense_line_prev)

        is_diverging_up = (attack_line > trading_line) & (trading_line > defense_line)
        is_diverging_down = (attack_line < trading_line) & (trading_line < defense_line)

        buy_signal = (is_rising & is_diverging_up).astype(int)
        sell_signal = (is_falling & is_diverging_down).astype(int)

        result = data.copy()
        result["attack_line"] = attack_line
        result["trading_line"] = trading_line
        result["defense_line"] = defense_line
        result["buy_signal"] = buy_signal
        result["sell_signal"] = sell_signal

        return result
