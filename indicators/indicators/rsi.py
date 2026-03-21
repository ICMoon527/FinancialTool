import numpy as np
import pandas as pd

from ..base import BaseIndicator


class RSI(BaseIndicator):
    """
    RSI（Relative Strength Index）相对强弱指标，由威尔斯·威尔德（J. Welles Wilder）于1978年提出。

    RSI是通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买盘和卖盘的意向和实力，
    从而作出未来市场的走势判断。RSI指标的数值在0到100之间波动，是最经典的超买超卖类指标之一。

    计算公式：
    1. 计算涨跌幅：
       - 涨跌 = 今日收盘价 - 昨日收盘价
    2. 分离涨幅和跌幅：
       - 涨幅 = max(涨跌, 0)
       - 跌幅 = max(-涨跌, 0)
    3. 计算平均涨幅和平均跌幅（通常用SMA或EMA）：
       - 平均涨幅 = 平均(涨幅, N)
       - 平均跌幅 = 平均(跌幅, N)
    4. 计算相对强弱：
       - RS = 平均涨幅 / 平均跌幅
    5. 计算RSI：
       - RSI = 100 - 100 / (1 + RS)
       - 其中N通常为14、6或24

    使用场景：
    - 超买区：RSI > 70（或80），可能见顶，考虑卖出
    - 超卖区：RSI < 30（或20），可能见底，考虑买入
    - 中线：RSI = 50，多空平衡点
    - 金叉：短期RSI（如6日）上穿长期RSI（如14日），买入信号
    - 死叉：短期RSI下穿长期RSI，卖出信号
    - 顶背离：价格创新高但RSI未创新高，可能见顶
    - 底背离：价格创新低但RSI未创新低，可能见底

    输入参数：
    - data: DataFrame，必须包含'Close'列
    - period: RSI周期，默认14
    - use_ema: 是否使用EMA计算平均涨跌，默认False（使用SMA）

    输出参数：
    - RSI: RSI指标值
    - gain: 每日涨幅（正值）
    - loss: 每日跌幅（正值）
    - avg_gain: 平均涨幅
    - avg_loss: 平均跌幅
    - RS: 相对强弱
    - overbought: 超买信号
    - oversold: 超卖信号

    注意事项：
    - RSI是震荡指标，在单边趋势市中容易出现钝化
    - 不同市场周期参数应调整：A股常用14、6，外汇常用9
    - RSI在30-70之间时信号可能不准确，建议只在超买超卖区使用
    - 钝化问题：在极端行情中，RSI可能长期在超买超卖区
    - 背离信号需要确认：出现背离后等待二次确认再行动

    最佳实践建议：
    1. 参数选择：根据不同市场选择参数，A股14日、6日常用
    2. 多周期验证：日线RSI+周线RSI，共振信号更可靠
    3. 背离确认：背离后等待3-5个交易日确认，避免假背离
    4. 趋势判断：先判断大趋势，顺势而为，逆势信号谨慎
    5. 成交量配合：RSI信号时成交量放大，信号更可靠
    6. 布林带结合：RSI+BOLL，在布林带上下轨的RSI信号更准确
    7. 分批操作：不要因一次RSI信号就满仓，分批进场更安全
    8. 止损设置：即使RSI信号良好，也要设置止损
    9. 指标组合：RSI+KDJ+MACD，多重指标确认成功率更高
    10. 动态调整：根据市场状态动态调整超买超卖阈值（牛市区80-20，熊市区70-30）
    """

    def __init__(self, period=14, use_ema=False):
        """
        初始化RSI指标参数。

        Args:
            period: RSI周期，默认14
            use_ema: 是否使用EMA计算平均涨跌，默认False（使用SMA）
        """
        self.period = period
        self.use_ema = use_ema

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

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame，必须包含'Close'列

        Returns:
            添加了RSI相关列的DataFrame，包括：
            - gain: 每日涨幅
            - loss: 每日跌幅
            - avg_gain: 平均涨幅
            - avg_loss: 平均跌幅
            - RS: 相对强弱
            - RSI: RSI指标值
            - overbought: 超买信号
            - oversold: 超卖信号
        """
        self.validate_input(data)

        result = data.copy()

        close = data["Close"]

        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        if self.use_ema:
            avg_gain = self._ema(gain, self.period)
            avg_loss = self._ema(loss, self.period)
        else:
            avg_gain = self._sma(gain, self.period)
            avg_loss = self._sma(loss, self.period)

        RS = avg_gain / avg_loss.replace(0, np.nan)

        RSI = 100 - 100 / (1 + RS)

        overbought = RSI >= 70
        oversold = RSI <= 30

        result["gain"] = gain
        result["loss"] = loss
        result["avg_gain"] = avg_gain
        result["avg_loss"] = avg_loss
        result["RS"] = RS
        result["RSI"] = RSI
        result["overbought"] = overbought
        result["oversold"] = oversold

        return result
