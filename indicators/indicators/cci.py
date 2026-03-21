import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CCI(BaseIndicator):
    """
    CCI（Commodity Channel Index）指标，中文称为顺势指标或商品通道指标。

    CCI指标由唐纳德·蓝伯特（Donald Lambert）于1980年提出，最初用于期货市场，后来广泛应用于股票市场。
    CCI指标是一种超买超卖指标，但与其他超买超卖指标不同的是，它的波动区间不受限制，可以达到正无穷或负无穷。

    计算公式：
    1. 计算典型价格：TP = (High + Low + Close) / 3
    2. 计算典型价格的N日简单移动平均：MA(TP, N)
    3. 计算平均绝对偏差：MD = Σ(|TP - MA(TP, N)|) / N
    4. 计算CCI：CCI = (TP - MA(TP, N)) / (0.015 × MD)

    使用场景：
    - CCI > 100 时，进入超买区间，可能回落，为卖出信号
    - CCI < -100 时，进入超卖区间，可能反弹，为买入信号
    - CCI由下往上突破+100时，为买入信号
    - CCI由上往下跌破+100时，为卖出信号
    - CCI由下往上突破-100时，为买入信号
    - CCI由上往下跌破-100时，为卖出信号
    - CCI在+100至-100之间时，为观望信号
    - 当CCI与价格出现背离时，为反转信号

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认20

    输出参数：
    - CCI: 顺势指标值
    - TP: 典型价格
    - overbought: 超买信号（CCI > 100）
    - oversold: 超卖信号（CCI < -100）
    - cross_100_up: 上穿+100信号
    - cross_100_down: 下穿+100信号
    - cross_neg100_up: 上穿-100信号
    - cross_neg100_down: 下穿-100信号

    注意事项：
    - CCI不受区间限制，波动可以很大，这是其特点也是缺点
    - CCI适合捕捉短期反转，但不适合判断长期趋势
    - 周期参数可以根据市场特性调整，20天是最常用的
    - 在单边市中，CCI可能长期处于超买或超卖区
    - 建议结合其他指标（如MA、MACD）一起使用

    最佳实践建议：
    1. 双重确认：等待CCI连续2天处于超买/超卖区再行动
    2. 趋势过滤：先判断大趋势，只在趋势方向上操作
    3. 背离确认：背离信号需要等待价格确认后再操作
    4. 多周期使用：同时使用日线和周线的CCI信号
    5. 组合使用：与RSI、KDJ等震荡指标配合使用
    6. 量价分析：结合成交量变化过滤假信号
    7. 止损设置：设置严格止损，控制风险
    8. 时间过滤：避免在重要数据发布前操作
    """

    def __init__(self, period=20):
        """
        初始化CCI指标

        Parameters
        ----------
        period : int, default 20
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算CCI指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含CCI计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3

        tp_ma = df['TP'].rolling(window=self.period).mean()

        md = df['TP'].rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        df['CCI'] = (df['TP'] - tp_ma) / (0.015 * md)

        df['overbought'] = df['CCI'] > 100
        df['oversold'] = df['CCI'] < -100

        df['cross_100_up'] = (df['CCI'] > 100) & (df['CCI'].shift(1) <= 100)
        df['cross_100_down'] = (df['CCI'] < 100) & (df['CCI'].shift(1) >= 100)

        df['cross_neg100_up'] = (df['CCI'] > -100) & (df['CCI'].shift(1) <= -100)
        df['cross_neg100_down'] = (df['CCI'] < -100) & (df['CCI'].shift(1) >= -100)

        return df[['CCI', 'TP', 'overbought', 'oversold', 'cross_100_up', 
                   'cross_100_down', 'cross_neg100_up', 'cross_neg100_down']]
