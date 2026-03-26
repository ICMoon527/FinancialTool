import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from indicators.base import BaseIndicator

pd.set_option('future.no_silent_downcasting', True)


class DragonLeaderStart(BaseIndicator):
    """
    龙头启动策略
    整合启动追踪、量能异动、换手热度、主力流向四个指标
    用于识别龙头股的启动信号
    
    策略组成：
    1. 启动追踪：识别龙头启动的核心信号（超跌板、趋势板、放量板）
    2. 量能异动：监控成交量的异常变化
    3. 换手热度：监控换手率活跃度
    4. 主力流向：监控主力资金流向
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
        Calculate the Dragon Leader Start strategy
        
        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume
        
        Returns:
            DataFrame with strategy values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        result = self._calculate_start_tracking(df, result)
        result = self._calculate_volume_anomaly(df, result)
        result = self._calculate_turnover_heat(df, result)
        result = self._calculate_main_capital_flow(df, result)
        result = self._calculate_combined_signals(result)

        return result

    def _calculate_start_tracking(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Start Tracking indicator components"""
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        ma3 = self._sma(close, 3)
        ma6 = self._sma(close, 6)
        ma9 = self._sma(close, 9)
        result['main_attack_line'] = (ma3 + ma6 + ma9) / 3

        ma5 = self._sma(close, 5)
        ma10 = self._sma(close, 10)
        ma15 = self._sma(close, 15)
        ma20 = self._sma(close, 20)
        ma25 = self._sma(close, 25)
        ma30 = self._sma(close, 30)
        result['main_control_line'] = (ma5 + ma10 + ma15 + ma20 + ma25 + ma30) / 6

        result['attack_line_red'] = (
            (result['main_attack_line'] > result['main_attack_line'].shift(1)) &
            (result['main_attack_line'] > result['main_control_line'])
        )

        change_pct = close / close.shift(1) - 1
        limit_up = (change_pct >= 0.09) & (close == high)
        result['limit_up'] = limit_up

        result['has_body'] = close > low

        if 'Amount' in df.columns:
            hhv_amount_10 = self._hhv(df['Amount'], 10)
            result['turnover_condition'] = df['Amount'] >= hhv_amount_10.shift(1)
        else:
            result['turnover_condition'] = True

        llv_low_34 = self._llv(low, 34)
        hhv_high_34 = self._hhv(high, 34)
        rsv = ((close - llv_low_34) / (hhv_high_34 - llv_low_34 + 1e-10)) * 100
        k = self._sma(rsv, 3)
        result['k'] = k

        result['bias_rate'] = (close / result['main_attack_line'] - 1) * 100

        consecutive_limit_up = self._count(limit_up, 2) == 2
        result['consecutive_limit_up'] = consecutive_limit_up

        a2zq = self._barslast(consecutive_limit_up)
        result['a2zq'] = a2zq
        result['continuous_limit_up_candle'] = (a2zq == 0)

        attack_above_control = result['main_attack_line'] > result['main_control_line']
        attack_cross_control = (attack_above_control & ~attack_above_control.shift(1).fillna(False).astype(bool))
        bars_since_cross = self._barslast(attack_cross_control)
        
        golden_cross = (
            (bars_since_cross <= 8) &
            (result['bias_rate'] <= 10) &
            attack_above_control &
            (close > result['main_attack_line'])
        )
        result['golden_cross'] = golden_cross

        if 'Amount' in df.columns:
            amount_increasing = self._count(df['Amount'] > df['Amount'].shift(1), 3) == 3
            result['sustained_volume'] = amount_increasing
        else:
            result['sustained_volume'] = True

        super_drop_limit_up = (
            limit_up &
            result['has_body'] &
            (self._count(limit_up, 13) == 1) &
            (k.shift(1) <= 20) &
            result['turnover_condition']
        )
        result['super_drop_limit_up'] = super_drop_limit_up

        trend_limit_up = (
            limit_up &
            result['has_body'] &
            (self._count(limit_up, 13) == 1) &
            golden_cross &
            result['turnover_condition']
        )
        result['trend_limit_up'] = trend_limit_up

        volume_limit_up = (
            limit_up &
            result['has_body'] &
            (self._count(limit_up, 13) == 1) &
            result['sustained_volume'] &
            result['turnover_condition']
        )
        result['volume_limit_up'] = volume_limit_up

        result['xg'] = volume_limit_up | trend_limit_up | super_drop_limit_up
        result['super_drop_signal'] = super_drop_limit_up
        result['trend_signal'] = trend_limit_up
        result['volume_signal'] = volume_limit_up

        return result

    def _calculate_volume_anomaly(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Anomaly indicator components"""
        
        if 'Amount' in df.columns:
            turnover_billion = df['Amount'] / 100000000
        else:
            turnover_billion = df['Volume'] * df['Close'] / 100000000
        result['turnover_billion'] = turnover_billion

        ma5 = self._sma(turnover_billion, 5)
        result['va_ma5'] = ma5

        ma10 = self._sma(turnover_billion, 10)
        result['va_ma10'] = ma10

        h10 = self._hhv(turnover_billion, 10).shift(1)
        result['va_h10'] = h10

        l10 = self._llv(turnover_billion, 10).shift(1)
        result['va_l10'] = l10

        result['high_volume_anomaly'] = turnover_billion >= h10
        result['low_volume_anomaly'] = turnover_billion <= l10

        return result

    def _calculate_turnover_heat(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Turnover Heat indicator components"""
        
        turnover_rate = (df['Volume'] / df['Volume'].rolling(window=100).mean()) * 100
        result['turnover_rate'] = turnover_rate

        result['inactive'] = turnover_rate < 3
        result['moderately_active'] = (turnover_rate >= 3) & (turnover_rate < 7)
        result['very_active'] = (turnover_rate >= 7) & (turnover_rate < 10)
        result['highly_active'] = turnover_rate >= 10

        return result

    def _calculate_main_capital_flow(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate Main Capital Flow indicator components"""
        
        price_change = df['Close'] - df['Open']
        
        main_buy = df['Volume'] * df['Close'] / 10000
        main_buy = main_buy.where(price_change > 0, 0)
        result['main_buy_wan'] = main_buy

        main_sell = df['Volume'] * df['Close'] / 10000
        main_sell = main_sell.where(price_change < 0, 0)
        result['main_sell_wan'] = main_sell

        main_net_buy = main_buy - main_sell
        result['main_net_buy_wan'] = main_net_buy

        result['main_capital_inflow'] = main_net_buy > 0
        result['main_capital_outflow'] = main_net_buy < 0

        return result

    def _calculate_combined_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined strategy signals"""
        
        result['dragon_start_strong'] = (
            result['xg'] &
            result['high_volume_anomaly'] &
            (result['very_active'] | result['highly_active']) &
            result['main_capital_inflow']
        )
        
        result['dragon_start_moderate'] = (
            result['xg'] &
            (result['high_volume_anomaly'] | result['moderately_active'])
        )
        
        result['dragon_start_signal'] = result['dragon_start_strong'] | result['dragon_start_moderate']
        
        result['signal_type'] = ''
        result.loc[result['super_drop_signal'], 'signal_type'] = '超跌板'
        result.loc[result['trend_signal'], 'signal_type'] = '趋势板'
        result.loc[result['volume_signal'], 'signal_type'] = '放量板'
        
        return result
