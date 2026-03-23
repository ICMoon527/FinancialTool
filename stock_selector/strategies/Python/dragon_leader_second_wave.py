import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from indicators.base import BaseIndicator


class DragonLeaderSecondWave(BaseIndicator):
    """
    龙头二波策略
    整合双龙狙击、龙头量能、动能二号三个指标
    用于识别龙头股的第二波上涨信号
    
    策略组成：
    1. 双龙狙击：识别龙头二波的核心信号（回踩、低吸）
    2. 龙头量能：监控龙头成交量的变化
    3. 动能二号：监控价格动能变化
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """Lowest Low Value"""
        return data.rolling(window=period).min()

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
        Calculate the Dragon Leader Second Wave strategy
        
        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume
        
        Returns:
            DataFrame with strategy values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        result = self._calculate_double_dragon_sniper(df, result)
        result = self._calculate_dragon_leader_volume(df, result)
        result = self._calculate_momentum_2(df, result)
        result = self._calculate_combined_signals(result)

        return result

    def _calculate_double_dragon_sniper(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Double Dragon Sniper indicator components"""
        
        open_ = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        limit_up_body = (
            (close / close.shift(1) >= 1.097) &
            (close == high) &
            (close > open_)
        )
        result['limit_up_body'] = limit_up_body

        gap_up = low > high.shift(1)
        result['gap_up'] = gap_up

        ma3 = self._sma(close, 3)
        ma6 = self._sma(close, 6)
        ma12 = self._sma(close, 12)
        ma24 = self._sma(close, 24)
        result['main_trading_line'] = (ma3 + ma6 + ma12 + ma24) / 4

        result['main_decision_line'] = self._sma(close, 45)

        result['decision_line_up'] = result['main_decision_line'] > result['main_decision_line'].shift(1)
        result['decision_line_down'] = result['main_decision_line'] < result['main_decision_line'].shift(1)

        result['trading_line_up'] = result['main_trading_line'] > result['main_trading_line'].shift(1)
        result['trading_line_down'] = result['main_trading_line'] < result['main_trading_line'].shift(1)

        limit_up = (close / close.shift(1) >= 1.090) & (high == close)
        result['limit_up'] = limit_up

        result['consecutive_limit_up_count'] = self._count(limit_up, 6)

        two_limit_up = self._count(limit_up, 2) == 2
        result['double_dragon_period'] = self._barslast(two_limit_up) <= 12

        pullback = (high > result['main_trading_line']) & (low <= result['main_trading_line'])
        result['pullback'] = pullback

        change_pct = close / close.shift(1)
        daily_change_range = (change_pct >= 0.91) & (change_pct <= 1.05)
        result['daily_change_range'] = daily_change_range

        cross_star_body = abs(close - open_) / close <= 0.03
        result['cross_star_body'] = cross_star_body

        double_dragon = (
            result['double_dragon_period'] &
            result['pullback'] &
            result['daily_change_range'] &
            result['cross_star_body'] &
            (result['consecutive_limit_up_count'] <= 6)
        )
        result['double_dragon'] = double_dragon

        result['upper_shadow'] = high - np.maximum(close, open_)
        result['lower_shadow'] = np.minimum(close, open_) - low
        result['body_size'] = abs(close - open_)

        buy1 = (
            (result['lower_shadow'] / close >= 0.02) &
            (result['body_size'] / close <= 0.03) &
            (result['lower_shadow'] > result['upper_shadow'])
        )
        result['buy1'] = buy1

        buy2 = (
            (result['lower_shadow'] / close >= 0.01) &
            (result['upper_shadow'] / close >= 0.01) &
            (cross_star_body <= 0.015)
        )
        result['buy2'] = buy2

        buy3 = (
            (result['upper_shadow'] / close >= 0.02) &
            (result['body_size'] / close <= 0.02)
        )
        result['buy3'] = buy3

        buy4 = (low == open_) & (result['body_size'] / close >= 0.02)
        result['buy4'] = buy4

        buy5 = (
            (open_ < close.shift(1)) &
            (close > open_) &
            (result['body_size'] / close >= 0.03)
        )
        result['buy5'] = buy5

        buy6 = (
            (close > open_) &
            (close > np.maximum(close.shift(1), open_.shift(1)))
        )
        result['buy6'] = buy6

        double_dragon_sniper1 = (
            result['double_dragon'] &
            (buy1 | buy2 | buy3 | buy4 | buy5 | buy6) &
            (close > result['main_trading_line'])
        )
        result['double_dragon_sniper1'] = double_dragon_sniper1

        double_dragon_sniper2 = (
            result['double_dragon'] &
            (buy1 | buy2 | buy3 | buy4 | buy5 | buy6) &
            (close <= result['main_trading_line'])
        )
        result['double_dragon_sniper2'] = double_dragon_sniper2

        rolling_max_1 = double_dragon_sniper1.rolling(window=5, min_periods=1).max().shift(1).fillna(False)
        result['a1'] = double_dragon_sniper1 & (~rolling_max_1.astype(bool))

        rolling_max_2 = double_dragon_sniper2.rolling(window=5, min_periods=1).max().shift(1).fillna(False)
        result['a2'] = double_dragon_sniper2 & (~rolling_max_2.astype(bool))

        result['red_diamond_signal'] = result['a1']
        result['blue_diamond_signal'] = result['a2']

        return result

    def _calculate_dragon_leader_volume(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Dragon Leader Volume indicator components"""
        
        if 'Amount' in df.columns:
            turnover_billion = df['Amount'] / 100000000
        else:
            turnover_billion = df['Volume'] * df['Close'] / 100000000
        result['turnover_billion'] = turnover_billion

        result['volume_ratio'] = df['Volume'] / df['Volume'].shift(1)

        result['dlv_ma5'] = self._sma(turnover_billion, 5)
        result['dlv_ma40'] = self._sma(turnover_billion, 40)
        result['dlv_ma135'] = self._sma(turnover_billion, 135)

        volume_ratio_change = turnover_billion / turnover_billion.shift(1)
        result['sky_high_volume'] = volume_ratio_change > 3.5
        result['double_volume'] = (volume_ratio_change >= 1.5) & (volume_ratio_change <= 3.5)

        six_day_high_volume = self._hhv(turnover_billion, 6)
        low_volume_0 = (turnover_billion / six_day_high_volume) <= 0.6
        rolling_max_low_volume = low_volume_0.rolling(window=5, min_periods=1).max().shift(1).fillna(False)
        result['low_volume'] = low_volume_0 & (~rolling_max_low_volume.astype(bool))

        result['false_yin_true_yang'] = (df['Close'] > df['Close'].shift(1)) & (df['Close'] < df['Open'])

        return result

    def _calculate_momentum_2(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Momentum 2 indicator components"""
        
        tt = (2 * df['Close'] + df['Open'] + df['High'] + df['Low'])
        momentum_price = 100 * (tt / self._ema(tt, 5) - 1)
        result['momentum_price'] = momentum_price

        strong_momentum_1 = (momentum_price > 0) & (momentum_price > momentum_price.shift(1))
        result['strong_momentum'] = strong_momentum_1

        medium_momentum_1 = (momentum_price > 0) & (momentum_price < momentum_price.shift(1))
        result['medium_momentum'] = medium_momentum_1

        no_momentum_1 = (momentum_price < 0) & (momentum_price < momentum_price.shift(1))
        result['no_momentum'] = no_momentum_1

        weak_momentum_1 = (momentum_price < 0) & (momentum_price > momentum_price.shift(1))
        result['weak_momentum'] = weak_momentum_1

        return result

    def _calculate_combined_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined strategy signals"""
        
        result['second_wave_strong'] = (
            (result['red_diamond_signal'] | result['blue_diamond_signal']) &
            (result['strong_momentum'] | result['double_volume'])
        )
        
        result['second_wave_moderate'] = (
            (result['red_diamond_signal'] | result['blue_diamond_signal']) &
            (result['medium_momentum'] | result['pullback'])
        )
        
        result['second_wave_signal'] = result['second_wave_strong'] | result['second_wave_moderate']
        
        result['signal_type'] = ''
        result.loc[result['red_diamond_signal'], 'signal_type'] = '红色钻石'
        result.loc[result['blue_diamond_signal'], 'signal_type'] = '蓝色钻石'
        
        return result
