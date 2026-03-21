import pandas as pd
import numpy as np
from ..base import BaseIndicator


class VR(BaseIndicator):
    """
    VR（Volume Ratio）指标，中文称为成交量比率指标或容量比率指标。

    VR指标是一种将成交量与价格结合的技术指标，通过计算一定周期内上涨日成交量总和与下跌日成交量总和的比率，
    来衡量市场的买卖力量和活跃度。

    计算公式：
    VR = (AV + 0.5 × CV) / (BV + 0.5 × CV) × 100
    其中：
    - AV：N日内上涨日的成交量总和（当日收盘价 > 前一日收盘价）
    - BV：N日内下跌日的成交量总和（当日收盘价 < 前一日收盘价）
    - CV：N日内平盘日的成交量总和（当日收盘价 = 前一日收盘价）
    - N：计算周期，通常为12日、24日或26日

    使用场景：
    - VR在70-150之间为正常区间，市场处于平衡状态
    - VR > 150 时，处于超买区间，可能回落
    - VR < 70 时，处于超卖区间，可能反弹
    - VR > 350 时，严重超买，行情可能反转
    - VR < 40 时，严重超卖，行情可能反转
    - VR由低往上穿越70时，为买入信号
    - VR由高往下跌破150时，为卖出信号
    - VR与价格同步上升时，趋势健康
    - VR与价格同步下降时，趋势健康

    输入参数：
    - data: DataFrame，必须包含'Close'、'Volume'列
    - period: 计算周期，默认26

    输出参数：
    - VR: 成交量比率指标值
    - AV: 上涨日成交量总和
    - BV: 下跌日成交量总和
    - CV: 平盘日成交量总和
    - overbought: 超买信号
    - oversold: 超卖信号
    - cross_70_up: 上穿70信号
    - cross_150_down: 下穿150信号

    注意事项：
    - VR是成交量指标，不直接反映价格
    - VR在成交量异常时需要谨慎解读
    - 超买超卖阈值需要根据市场特性调整
    - 周期参数可以根据市场特性调整
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：VR与价格趋势同步更可靠
    2. 组合使用：与价格指标配合使用
    3. 多周期使用：同时使用短期和长期VR
    4. 量价配合：结合价格和成交量变化验证信号
    5. 时间过滤：避免在重要数据发布前操作
    6. 平滑处理：使用VR均线减少噪音
    7. 相对比较：比较不同股票的VR水平
    8. 灵活调整：根据市场波动性调整参数
    """

    def __init__(self, period=26):
        """
        初始化VR指标

        Parameters
        ----------
        period : int, default 26
            计算周期
        """
        self.period = period

    def calculate(self, data):
        """
        计算VR指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'Close'、'Volume'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含VR计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        close_prev = df['Close'].shift(1)

        av = np.where(df['Close'] > close_prev, df['Volume'], 0)
        bv = np.where(df['Close'] < close_prev, df['Volume'], 0)
        cv = np.where(df['Close'] == close_prev, df['Volume'], 0)

        df['AV'] = pd.Series(av, index=df.index).rolling(window=self.period).sum()
        df['BV'] = pd.Series(bv, index=df.index).rolling(window=self.period).sum()
        df['CV'] = pd.Series(cv, index=df.index).rolling(window=self.period).sum()

        df['VR'] = (df['AV'] + 0.5 * df['CV']) / (df['BV'] + 0.5 * df['CV']).replace(0, np.nan) * 100

        df['overbought'] = df['VR'] > 150
        df['oversold'] = df['VR'] < 70

        df['cross_70_up'] = (df['VR'] > 70) & (df['VR'].shift(1) <= 70)
        df['cross_150_down'] = (df['VR'] < 150) & (df['VR'].shift(1) >= 150)

        return df[['VR', 'AV', 'BV', 'CV', 'overbought', 'oversold', 'cross_70_up', 'cross_150_down']]
