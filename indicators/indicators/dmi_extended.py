import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DMIExtended(BaseIndicator):
    """
    DMIExtended（动向指标扩展），中文称为动向指标扩展。

    DMIExtended指标是DMI指标的扩展版本，通过增加更多的平滑和信号检测，来提高指标的可靠性。
    DMIExtended指标结合了+DI、-DI、ADX和ADXR，来判断价格的趋势和强度。

    计算公式：
    1. 计算+DM：+DM = High - High_prev if High - High_prev > Low_prev - Low else 0
    2. 计算-DM：-DM = Low_prev - Low if Low_prev - Low > High - High_prev else 0
    3. 计算TR：TR = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
    4. 计算+DI：+DI = 100 × SMA(+DM, N) / SMA(TR, N)
    5. 计算-DI：-DI = 100 × SMA(-DM, N) / SMA(TR, N)
    6. 计算DX：DX = 100 × |+DI - -DI| / (|+DI + -DI|)
    7. 计算ADX：ADX = SMA(DX, M)
    8. 计算ADXR：ADXR = (ADX + ADX_prev_M) / 2

    使用场景：
    - +DI上穿-DI时，为买入信号
    - +DI下穿-DI时，为卖出信号
    - ADX上升时，表示趋势增强
    - ADX下降时，表示趋势减弱
    - ADX > 25 时，表示有明显趋势
    - ADX < 20 时，表示无明显趋势
    - ADX > 40 时，表示趋势极强

    输入参数：
    - data: DataFrame，必须包含'High'、'Low'、'Close'列
    - period: 计算周期，默认14
    - adx_period: ADX周期，默认14

    输出参数：
    - +DM: 正动向变动值
    - -DM: 负动向变动值
    - TR: 真实波幅
    - +DI: 正方向指标
    - -DI: 负方向指标
    - DX: 动向指标
    - ADX: 平均动向指标
    - ADXR: 平均动向指标比率
    - plus_di_above_minus_di: +DI在-DI上方信号
    - plus_di_below_minus_di: +DI在-DI下方信号
    - adx_rising: ADX上升信号
    - adx_falling: ADX下降信号
    - adx_above_25: ADX > 25信号
    - adx_below_20: ADX < 20信号
    - golden_cross: +DI上穿-DI信号
    - death_cross: +DI下穿-DI信号

    注意事项：
    - DMIExtended是趋势指标，在单边市中表现最好
    - DMIExtended在震荡市中会产生频繁的假信号
    - ADX的高低比+DI和-DI的交叉更重要
    - 建议结合其他指标（如MACD、RSI）一起使用

    最佳实践建议：
    1. 趋势确认：ADX上升且+DI在-DI上方更可靠
    2. 组合使用：与MACD等趋势指标配合使用
    3. 多周期使用：同时使用日线和周线的DMIExtended
    4. 量价配合：结合成交量变化验证信号
    5. 趋势过滤：先判断大趋势，顺势操作
    6. 止损设置：设置严格止损，控制风险
    7. 灵活调整：根据市场波动性调整参数
    8. 等待确认：信号出现后等待1-2个交易日确认
    """

    def __init__(self, period=14, adx_period=14):
        """
        初始化DMIExtended指标

        Parameters
        ----------
        period : int, default 14
            计算周期
        adx_period : int, default 14
            ADX周期
        """
        self.period = period
        self.adx_period = adx_period

    def calculate(self, data):
        """
        计算DMIExtended指标

        Parameters
        ----------
        data : pandas.DataFrame
            包含'High'、'Low'、'Close'列的OHLCV数据

        Returns
        -------
        pandas.DataFrame
            包含DMIExtended计算结果的DataFrame
        """
        self.validate_input(data)

        df = data.copy()

        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        prev_close = df['Close'].shift(1)

        up_move = df['High'] - prev_high
        down_move = prev_low - df['Low']

        df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - prev_close)
        tr3 = abs(df['Low'] - prev_close)
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df['+DI'] = 100 * (df['+DM'].rolling(window=self.period).sum() / df['TR'].rolling(window=self.period).sum().replace(0, np.nan))
        df['-DI'] = 100 * (df['-DM'].rolling(window=self.period).sum() / df['TR'].rolling(window=self.period).sum().replace(0, np.nan))
        df['+DI'] = df['+DI'].fillna(0)
        df['-DI'] = df['-DI'].fillna(0)

        df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan))
        df['DX'] = df['DX'].fillna(0)
        df['ADX'] = df['DX'].rolling(window=self.adx_period).mean()
        df['ADXR'] = (df['ADX'] + df['ADX'].shift(self.adx_period)) / 2

        df['plus_di_above_minus_di'] = df['+DI'] > df['-DI']
        df['plus_di_below_minus_di'] = df['+DI'] < df['-DI']
        df['adx_rising'] = df['ADX'] > df['ADX'].shift(1)
        df['adx_falling'] = df['ADX'] < df['ADX'].shift(1)
        df['adx_above_25'] = df['ADX'] > 25
        df['adx_below_20'] = df['ADX'] < 20
        df['golden_cross'] = (df['+DI'] > df['-DI']) & (df['+DI'].shift(1) <= df['-DI'].shift(1))
        df['death_cross'] = (df['+DI'] < df['-DI']) & (df['+DI'].shift(1) >= df['-DI'].shift(1))

        return df[['+DM', '-DM', 'TR', '+DI', '-DI', 'DX', 'ADX', 'ADXR',
                   'plus_di_above_minus_di', 'plus_di_below_minus_di',
                   'adx_rising', 'adx_falling',
                   'adx_above_25', 'adx_below_20',
                   'golden_cross', 'death_cross']]
