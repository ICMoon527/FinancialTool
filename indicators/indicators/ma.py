import numpy as np
import pandas as pd

from ..base import BaseIndicator


class MA(BaseIndicator):
    """
    MA（Moving Average）移动平均线指标，是股票技术分析中最基础、最常用的指标之一。

    移动平均线是将某一段时间的收盘价之和除以该周期，它是道氏理论的形象化表述，
    能够反映价格趋势的运行方向，帮助投资者识别当前趋势并判断趋势的反转。

    计算公式：
    1. 简单移动平均线（SMA）：
       - SMA(N) = (今日收盘价 + 昨日收盘价 + ... + 前N-1日收盘价) / N
    2. 指数移动平均线（EMA）：
       - EMA(N) = 今日收盘价 × (2/(N+1)) + 前一日EMA × (1-2/(N+1))
    3. 加权移动平均线（WMA）：
       - WMA(N) = (今日收盘价×N + 昨日收盘价×(N-1) + ... + 前N-1日收盘价×1) / (N×(N+1)/2)

    使用场景：
    - 趋势判断：价格在MA之上为上升趋势，价格在MA之下为下降趋势
    - 支撑阻力：MA可以作为支撑线和阻力线
    - 金叉死叉：短期MA上穿长期MA为金叉（买入信号），短期MA下穿长期MA为死叉（卖出信号）
    - 多头排列：短期MA > 中期MA > 长期MA，且都向上，强烈看涨信号
    - 空头排列：短期MA < 中期MA < 长期MA，且都向下，强烈看跌信号
    - 乖离率：价格偏离MA过远可能会回调

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - periods: 周期列表，默认[5, 10, 20, 30, 60]
    - ma_type: 移动平均类型，可选'sma'、'ema'、'wma'，默认'sma'

    输出参数：
    - MA5, MA10, MA20, ...: 对应周期的移动平均线
    - golden_cross: 金叉信号（5日均线上穿10日均线）
    - death_cross: 死叉信号（5日均线下穿10日均线）
    - above_ma: 价格在MA之上
    - below_ma: 价格在MA之下
    - trend_up: 上升趋势
    - trend_down: 下降趋势

    注意事项：
    - MA是滞后指标，信号会比价格晚
    - 在震荡市中，MA容易产生伪信号
    - 参数选择：短期（5,10）适合短线，中期（20,30）适合中线，长期（60,120,250）适合长线
    - 均线数量：建议使用3-5条均线，过多反而混乱
    - 不同市场参数应调整：A股常用5-10-20-60-250，美股常用50-200

    最佳实践建议：
    1. 多周期组合：短期MA+中期MA+长期MA，组合效果最好
    2. 趋势优先：先判断大趋势，顺势而为，逆势信号谨慎
    3. 金叉确认：金叉后等待3个交易日，且成交量放大，信号更可靠
    4. 多头排列：出现多头排列后坚定持有，直到排列破坏
    5. 支撑阻力：在上升趋势中，回调到MA附近是买入机会
    6. 止损设置：有效跌破重要MA（如20日、60日）要止损
    7. 成交量配合：均线金叉时成交量放大，信号更可靠
    8. 布林带结合：MA+BOLL，在布林带中轨的MA信号更准确
    9. MACD结合：MA+MACD，趋势共振时成功率更高
    10. 动态调整：根据市场状态动态调整均线参数
    """

    def __init__(self, periods=None, ma_type='sma'):
        """
        初始化MA指标参数。

        Args:
            periods: 周期列表，默认[5, 10, 20, 30, 60]
            ma_type: 移动平均类型，可选'sma'、'ema'、'wma'，默认'sma'
        """
        if periods is None:
            periods = [5, 10, 20, 30, 60]
        self.periods = periods
        self.ma_type = ma_type.lower()

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

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算指数移动平均线（EMA）。

        Args:
            data: 输入序列
            period: 周期参数

        Returns:
            EMA序列
        """
        return data.ewm(span=period, adjust=False).mean()

    def _wma(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算加权移动平均线（WMA）。

        Args:
            data: 输入序列
            period: 周期参数

        Returns:
            WMA序列
        """
        weights = np.arange(1, period + 1)
        return data.rolling(window=period, min_periods=1).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum()
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算MA指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame，必须包含'Close'列

        Returns:
            添加了MA相关列的DataFrame，包括：
            - MA5, MA10, MA20, ...: 对应周期的移动平均线
            - golden_cross: 金叉信号
            - death_cross: 死叉信号
            - trend_up: 上升趋势
            - trend_down: 下降趋势
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]

        ma_func = self._sma
        if self.ma_type == 'ema':
            ma_func = self._ema
        elif self.ma_type == 'wma':
            ma_func = self._wma

        ma_values = {}
        for period in self.periods:
            ma_col = f"{self.ma_type.upper()}{period}"
            ma_values[ma_col] = ma_func(close, period)
            result[ma_col] = ma_values[ma_col]

        if 5 in self.periods and 10 in self.periods:
            ma5_col = f"{self.ma_type.upper()}5"
            ma10_col = f"{self.ma_type.upper()}10"
            ma5 = ma_values[ma5_col]
            ma10 = ma_values[ma10_col]

            golden_cross = (ma5 > ma10) & (ma5.shift(1) <= ma10.shift(1))
            death_cross = (ma5 < ma10) & (ma5.shift(1) >= ma10.shift(1))

            result["golden_cross"] = golden_cross
            result["death_cross"] = death_cross

        if len(self.periods) >= 3:
            periods_sorted = sorted(self.periods)
            short_col = f"{self.ma_type.upper()}{periods_sorted[0]}"
            mid_col = f"{self.ma_type.upper()}{periods_sorted[1]}"
            long_col = f"{self.ma_type.upper()}{periods_sorted[2]}"

            short_ma = ma_values[short_col]
            mid_ma = ma_values[mid_col]
            long_ma = ma_values[long_col]

            trend_up = (short_ma > mid_ma) & (mid_ma > long_ma) & \
                       (short_ma > short_ma.shift(1)) & (mid_ma > mid_ma.shift(1))
            trend_down = (short_ma < mid_ma) & (mid_ma < long_ma) & \
                         (short_ma < short_ma.shift(1)) & (mid_ma < mid_ma.shift(1))

            result["trend_up"] = trend_up
            result["trend_down"] = trend_down

        return result
