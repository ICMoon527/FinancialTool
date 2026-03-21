import pandas as pd
import numpy as np
from ..base import BaseIndicator


class ASIExtended(BaseIndicator):
    """
    ASIExtended（振动升降指标扩展），中文称为振动升降指标扩展。

    ASIExtended指标是ASI指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    ASIExtended指标结合了振动升降指标和平均振动升降指标，来反应价格的趋势。

    计算公式：
    1. 计算A：A = max(High - Close_prev, Low - Close_prev)
    2. 计算B：B = abs(High - Low)
    3. 计算C：C = abs(Close_prev - Open_prev)
    4. 计算D：D = Close - Open + 0.5 × (Close_prev - Open_prev) + 0.25 × (Close_prev_prev - Open_prev_prev)
    5. 计算E：E = max(A, B, C)
    6. 计算R：R = E + 0.5 × D
    7. 计算X：X = (Close - Close_prev) + 0.5 × (Close - Open) + 0.25 × (Close_prev - Open_prev)
    8. 计算SI：SI = X / R
    9. 计算ASI：ASI = cumsum(SI)
    10. 计算ASI_MA：ASI_MA = EMA(ASI, N)
    11. 计算ASI_OSC：ASI_OSC = (ASI - ASI_MA) × 100

    使用场景：
    - ASI上升时，表示上升趋势，为买入信号
    - ASI下降时，表示下降趋势，为卖出信号
    - ASI与价格同步时，趋势健康
    - ASI与价格背离时，可能反转
    - ASI创新高时，表示趋势强劲
    - ASI创新低时，表示趋势疲软
    - ASI_OSC > 0 时，表示量价配合良好
    - ASI_OSC < 0 时，表示量价背离

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Open'、'Close'列
    - ma_period: ASI移动平均周期，默认10

    输出参数：
    - SI: 振动升降值
    - ASI: 振动升降指标
    - ASI_MA: ASI的移动平均线
    - ASI_OSC: ASI的震荡指标
    - asi_rising: ASI上升信号
    - asi_falling: ASI下降信号
    - asi_above_ma: ASI在MA上方信号
    - asi_below_ma: ASI在MA下方信号
    - asi_new_high: ASI创新高信号
    - asi_new_low: ASI创新低信号

    注意事项：
    - ASIExtended是趋势指标，在单边市中表现最好
    - ASIExtended在震荡市中会产生频繁的假信号
    - ASIExtended计算较为复杂
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ASI和价格同时向上更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的ASIExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, ma_period=10):
        """
        初始化ASIExtended指标

        Parameters
        ----------
        ma_period : int, default 10
            ASI移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, data):
        """
        计算ASIExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Open'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含ASIExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_close = df['Close'].shift(1)
        prev_open = df['Open'].shift(1)
        prev_prev_close = df['Close'].shift(2)
        prev_prev_open = df['Open'].shift(2)

        A = np.maximum(df['High'] - prev_close, prev_close - df['Low'])
        B = abs(df['High'] - df['Low'])
        C = abs(prev_close - prev_open)
        D = (df['Close'] - df['Open']) + 0.5 * (prev_close - prev_open) + 0.25 * (prev_prev_close - prev_prev_open)
        E = pd.concat([A, B, C], axis=1).max(axis=1)
        R = E + 0.5 * D
        X = (df['Close'] - prev_close) + 0.5 * (df['Close'] - df['Open']) + 0.25 * (prev_close - prev_open)
        df['SI'] = X / R.replace(0, np.nan)
        df['SI'] = df['SI'].fillna(0)
        df['ASI'] = df['SI'].cumsum()
        df['ASI_MA'] = df['ASI'].ewm(span=self.ma_period, adjust=False).mean()
        df['ASI_OSC'] = (df['ASI'] - df['ASI_MA']) * 100

        df['asi_rising'] = df['ASI'] > df['ASI'].shift(1)
        df['asi_falling'] = df['ASI'] < df['ASI'].shift(1)
        df['asi_above_ma'] = df['ASI'] > df['ASI_MA']
        df['asi_below_ma'] = df['ASI'] < df['ASI_MA']
        df['asi_new_high'] = df['ASI'] == df['ASI'].rolling(window=self.ma_period * 2).max()
        df['asi_new_low'] = df['ASI'] == df['ASI'].rolling(window=self.ma_period * 2).min()

        return df[['SI', 'ASI', 'ASI_MA', 'ASI_OSC',
                   'asi_rising', 'asi_falling',
                   'asi_above_ma', 'asi_below_ma',
                   'asi_new_high', 'asi_new_low']]
