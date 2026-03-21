import pandas as pd
import numpy as np
from ..base import BaseIndicator


class DoubleDragonSniper(BaseIndicator):
    """
    双龙狙击指标
    紫色主力操盘线向上时紫线变粗，表示短线多头行情
    紫色主力操盘线向下时紫线变细，表示短线空头行情
    黄色主力决策线向上时黄线变粗，表示中线多头行情
    黄色主力决策线向下时黄线变细，表示中线空头行情
    红色实心K线表示股价涨停
    紫色柱体表示跳空缺口
    蓝色钻石表示主力操盘线之下的低吸信号
    红色钻石表示主力操盘线之上的低吸信号
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """Lowest Low Value"""
        return data.rolling(window=period).min()

    def _ref(self, data: pd.Series, periods: int = 1) -> pd.Series:
        """Reference to previous period"""
        return data.shift(periods)

    def _count(self, condition: pd.Series, period: int) -> pd.Series:
        """Count of condition being true over period"""
        return condition.astype(int).rolling(window=period).sum()

    def _barslast(self, condition: pd.Series) -> pd.Series:
        """Bars since last occurrence of condition"""
        result = pd.Series(np.nan, index=condition.index)
        last_bar = -1
        for i in range(len(condition)):
            if condition.iloc[i]:
                last_bar = i
                result.iloc[i] = 0
            elif last_bar != -1:
                result.iloc[i] = i - last_bar
        return result

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Double Dragon Sniper indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 实体板: (CLOSE/REF(CLOSE,1)>=1.097) AND (CLOSE=HIGH) AND (CLOSE>OPEN)
        limit_up_body = (
            (df['Close'] / self._ref(df['Close']) >= 1.097) &
            (df['Close'] == df['High']) &
            (df['Close'] > df['Open'])
        )
        result['limit_up_body'] = limit_up_body

        # 跳空缺口: LOW>REF(HIGH,1)
        gap_up = df['Low'] > self._ref(df['High'])
        result['gap_up'] = gap_up

        # 主力操盘线: (MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4
        ma3 = self._sma(df['Close'], 3)
        ma6 = self._sma(df['Close'], 6)
        ma12 = self._sma(df['Close'], 12)
        ma24 = self._sma(df['Close'], 24)
        result['main_trading_line'] = (ma3 + ma6 + ma12 + ma24) / 4

        # 主力决策线: MA(CLOSE,45)
        result['main_decision_line'] = self._sma(df['Close'], 45)

        # 主力决策线向上/向下
        result['decision_line_up'] = result['main_decision_line'] > self._ref(result['main_decision_line'])
        result['decision_line_down'] = result['main_decision_line'] < self._ref(result['main_decision_line'])

        # 主力操盘线向上/向下
        result['trading_line_up'] = result['main_trading_line'] > self._ref(result['main_trading_line'])
        result['trading_line_down'] = result['main_trading_line'] < self._ref(result['main_trading_line'])

        # 涨停: CLOSE/REF(CLOSE,1)>=1.090 AND HIGH=CLOSE
        limit_up = (df['Close'] / self._ref(df['Close']) >= 1.090) & (df['High'] == df['Close'])
        result['limit_up'] = limit_up

        # 连板数: COUNT(涨停,6)
        result['consecutive_limit_up_count'] = self._count(limit_up, 6)

        # 双龙周期: BARSLAST(COUNT(涨停,2)=2)<=12
        two_limit_up = self._count(limit_up, 2) == 2
        result['double_dragon_period'] = self._barslast(two_limit_up) <= 12

        # 回踩: HIGH>主力操盘线 AND LOW<=主力操盘线
        pullback = (df['High'] > result['main_trading_line']) & (df['Low'] <= result['main_trading_line'])
        result['pullback'] = pullback

        # 当日涨幅: CLOSE/REF(CLOSE,1)>=0.91 AND CLOSE/REF(CLOSE,1)<=1.05
        change_pct = df['Close'] / self._ref(df['Close'])
        daily_change_range = (change_pct >= 0.91) & (change_pct <= 1.05)
        result['daily_change_range'] = daily_change_range

        # 十字星实体: ABS(CLOSE-OPEN)/CLOSE<=0.03
        cross_star_body = abs(df['Close'] - df['Open']) / df['Close'] <= 0.03
        result['cross_star_body'] = cross_star_body

        # 双龙条件: 双龙周期 AND 回踩 AND 当日涨幅 AND 十字星实体 AND 连板数<=6
        double_dragon = (
            result['double_dragon_period'] &
            result['pullback'] &
            result['daily_change_range'] &
            result['cross_star_body'] &
            (result['consecutive_limit_up_count'] <= 6)
        )
        result['double_dragon'] = double_dragon

        # 上影线: HIGH-MAX(CLOSE,OPEN)
        result['upper_shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])

        # 下影线: MIN(CLOSE,OPEN)-LOW
        result['lower_shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']

        # 实体部分: ABS(CLOSE-OPEN)
        result['body_size'] = abs(df['Close'] - df['Open'])

        # 买1: 下影线/CLOSE>=0.02 AND 实体部分/CLOSE<=0.03 AND 下影线>上影线
        buy1 = (
            (result['lower_shadow'] / df['Close'] >= 0.02) &
            (result['body_size'] / df['Close'] <= 0.03) &
            (result['lower_shadow'] > result['upper_shadow'])
        )
        result['buy1'] = buy1

        # 买2: 下影线/CLOSE>=0.01 AND 上影线/CLOSE>=0.01 AND 十字星实体<=0.015
        buy2 = (
            (result['lower_shadow'] / df['Close'] >= 0.01) &
            (result['upper_shadow'] / df['Close'] >= 0.01) &
            (cross_star_body <= 0.015)
        )
        result['buy2'] = buy2

        # 买3: 上影线/CLOSE>=0.02 AND 实体部分/CLOSE<=0.02
        buy3 = (
            (result['upper_shadow'] / df['Close'] >= 0.02) &
            (result['body_size'] / df['Close'] <= 0.02)
        )
        result['buy3'] = buy3

        # 买4: LOW=OPEN AND 实体部分/CLOSE>=0.02
        buy4 = (df['Low'] == df['Open']) & (result['body_size'] / df['Close'] >= 0.02)
        result['buy4'] = buy4

        # 买5: OPEN<REF(CLOSE,1) AND CLOSE>OPEN AND 实体部分/CLOSE>=0.03
        buy5 = (
            (df['Open'] < self._ref(df['Close'])) &
            (df['Close'] > df['Open']) &
            (result['body_size'] / df['Close'] >= 0.03)
        )
        result['buy5'] = buy5

        # 买6: CLOSE>OPEN AND CLOSE>REF(MAX(CLOSE,OPEN),1)
        buy6 = (
            (df['Close'] > df['Open']) &
            (df['Close'] > self._ref(np.maximum(df['Close'], df['Open'])))
        )
        result['buy6'] = buy6

        # 双龙狙击1: 双龙 AND (买1 OR 买2 OR 买3 OR 买4 OR 买5 OR 买6) AND (CLOSE>主力操盘线)
        double_dragon_sniper1 = (
            result['double_dragon'] &
            (buy1 | buy2 | buy3 | buy4 | buy5 | buy6) &
            (df['Close'] > result['main_trading_line'])
        )
        result['double_dragon_sniper1'] = double_dragon_sniper1

        # 双龙狙击2: 双龙 AND (买1 OR 买2 OR 买3 OR 买4 OR 买5 OR 买6) AND (CLOSE<=主力操盘线)
        double_dragon_sniper2 = (
            result['double_dragon'] &
            (buy1 | buy2 | buy3 | buy4 | buy5 | buy6) &
            (df['Close'] <= result['main_trading_line'])
        )
        result['double_dragon_sniper2'] = double_dragon_sniper2

        # FILTER(双龙狙击1=1, 5) - simplified with rolling max
        rolling_max_1 = double_dragon_sniper1.rolling(window=5, min_periods=1).max().shift(1).fillna(False)
        result['a1'] = double_dragon_sniper1 & (~rolling_max_1.astype(bool))

        # FILTER(双龙狙击2=1, 5)
        rolling_max_2 = double_dragon_sniper2.rolling(window=5, min_periods=1).max().shift(1).fillna(False)
        result['a2'] = double_dragon_sniper2 & (~rolling_max_2.astype(bool))

        # 红色钻石信号
        result['red_diamond_signal'] = result['a1']
        # 蓝色钻石信号
        result['blue_diamond_signal'] = result['a2']

        return result
