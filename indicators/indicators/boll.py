import numpy as np
import pandas as pd

from ..base import BaseIndicator


class BOLL(BaseIndicator):
    """
    BOLL（Bollinger Bands）布林带指标，由约翰·布林格（John Bollinger）于20世纪80年代提出。

    布林带是一种简单但实用的技术分析工具，它利用统计学原理，确定股价的波动范围及未来走势，
    利用波带显示股价的安全高低价位，因而也被称为布林带。布林带由上轨、中轨、下轨三条线组成。

    计算公式：
    1. 计算中轨（MA）：
       - 中轨 = MA(N)，通常N=20
    2. 计算标准差（Std）：
       - 标准差 = 过去N日收盘价的标准差
    3. 计算上轨和下轨：
       - 上轨 = 中轨 + K × 标准差，通常K=2
       - 下轨 = 中轨 - K × 标准差，通常K=2

    使用场景：
    - 突破上轨：价格突破上轨，可能超买，考虑卖出
    - 跌破下轨：价格跌破下轨，可能超卖，考虑买入
    - 中轨支撑/阻力：上升趋势中，中轨是支撑；下降趋势中，中轨是阻力
    - 开口收窄：布林带开口收窄，预示即将有大行情
    - 开口放大：布林带开口放大，趋势开始或加速
    - 喇叭口形态：判断趋势的开始、加速、结束
    - 价格在布林带内运行：正常波动范围，高抛低吸

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: 周期参数，默认20
    - std_dev: 标准差倍数，默认2

    输出参数：
    - BOLL_MID: 布林带中轨（MA）
    - BOLL_UP: 布林带上轨
    - BOLL_LOW: 布林带下轨
    - BOLL_WIDTH: 布林带宽度
    - BOLL_PCT: 布林带百分比位置（价格在布林带中的相对位置）
    - above_upper: 突破上轨
    - below_lower: 跌破下轨
    - squeeze: 布林带收窄信号

    注意事项：
    - 布林带是震荡指标，在单边趋势市中容易失效
    - 参数选择：通常20-2是标准组合，短线可设10-2，长线可设50-2.5
    - 开口收窄后不一定马上有大行情，需要等待突破确认
    - 在震荡市中，布林带效果最好；在单边市中，效果较差
    - 上轨下轨只是统计概念，不是绝对的压力支撑

    最佳实践建议：
    1. 参数选择：A股常用20-2，短线可调整为10-2或15-1.8
    2. 突破确认：突破上轨或下轨后，等待2-3个交易日确认
    3. 成交量配合：突破时成交量放大，信号更可靠
    4. 喇叭口形态：识别三种喇叭口形态，判断趋势阶段
    5. 多周期验证：日线布林带+周线布林带，共振信号更可靠
    6. RSI结合：布林带+RSI，超买超卖判断更准确
    7. KDJ结合：布林带+KDJ，在布林带上下轨的KDJ信号更准确
    8. 趋势优先：先判断大趋势，顺势而为，逆势信号谨慎
    9. 止损设置：有效突破布林带后，反向穿越要止损
    10. 分批操作：不要因一次布林带信号就满仓，分批进场更安全
    11. 中轨运用：上升趋势中，回踩中轨是买入机会；下降趋势中，反弹中轨是卖出机会
    12. 开口收窄：布林带开口收窄后，关注后续突破方向，顺势而为
    """

    def __init__(self, period=20, std_dev=2):
        """
        初始化布林带指标参数。

        Args:
            period: 周期参数，默认20
            std_dev: 标准差倍数，默认2
        """
        self.period = period
        self.std_dev = std_dev

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

    def _std(self, data: pd.Series, period: int) -> pd.Series:
        """
        计算标准差。

        Args:
            data: 输入序列
            period: 周期参数

        Returns:
            标准差序列
        """
        return data.rolling(window=period, min_periods=1).std(ddof=0)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame，必须包含'Close'列

        Returns:
            添加了布林带相关列的DataFrame，包括：
            - BOLL_MID: 中轨
            - BOLL_UP: 上轨
            - BOLL_LOW: 下轨
            - BOLL_WIDTH: 布林带宽度
            - BOLL_PCT: 价格在布林带中的位置百分比
            - above_upper: 突破上轨
            - below_lower: 跌破下轨
            - squeeze: 布林带收窄信号
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]

        mid = self._sma(close, self.period)
        std = self._std(close, self.period)
        up = mid + self.std_dev * std
        low = mid - self.std_dev * std

        width = (up - low) / mid
        width = width.fillna(0)

        boll_range = up - low
        boll_range = boll_range.replace(0, np.nan)
        pct = (close - low) / boll_range * 100
        pct = pct.fillna(50)

        above_upper = close > up
        below_lower = close < low

        width_ma = self._sma(width, 10)
        squeeze = (width < width_ma.shift(1) * 0.9) & (width_ma < width_ma.shift(5) * 0.85)

        result["BOLL_MID"] = mid
        result["BOLL_UP"] = up
        result["BOLL_LOW"] = low
        result["BOLL_WIDTH"] = width
        result["BOLL_PCT"] = pct
        result["above_upper"] = above_upper
        result["below_lower"] = below_lower
        result["squeeze"] = squeeze

        return result
