import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from logger import log

class SignalGenerator:
    BUY = 1
    SELL = -1
    HOLD = 0
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, strategy_name: str = 'ma_cross', **kwargs) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        df['signal'] = SignalGenerator.HOLD
        
        strategies = {
            'ma_cross': SignalGenerator.ma_cross_signal,
            'macd': SignalGenerator.macd_signal,
            'rsi': SignalGenerator.rsi_signal,
            'bollinger': SignalGenerator.bollinger_signal,
            'kdj': SignalGenerator.kdj_signal,
            'dual_thrust': SignalGenerator.dual_thrust_signal,
            'turtle': SignalGenerator.turtle_signal
        }
        
        if strategy_name in strategies:
            df = strategies[strategy_name](df, **kwargs)
        else:
            log.warning(f"策略 {strategy_name} 不存在，使用默认均线交叉策略")
            df = SignalGenerator.ma_cross_signal(df, **kwargs)
        
        df['position'] = df['signal'].shift(1).fillna(0)
        log.info(f"策略 {strategy_name} 信号生成完成")
        return df
    
    @staticmethod
    def ma_cross_signal(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
        if f'MA{short_period}' not in df.columns or f'MA{long_period}' not in df.columns:
            df[f'MA{short_period}'] = df['close'].rolling(window=short_period).mean()
            df[f'MA{long_period}'] = df['close'].rolling(window=long_period).mean()
        
        condition_buy = (df[f'MA{short_period}'] > df[f'MA{long_period}']) & \
                        (df[f'MA{short_period}'].shift(1) <= df[f'MA{long_period}'].shift(1))
        
        condition_sell = (df[f'MA{short_period}'] < df[f'MA{long_period}']) & \
                         (df[f'MA{short_period}'].shift(1) >= df[f'MA{long_period}'].shift(1))
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def macd_signal(df: pd.DataFrame) -> pd.DataFrame:
        if 'MACD' not in df.columns or 'MACD_signal' not in df.columns:
            from strategy.factors import factor_library
            df = factor_library.calculate_macd(df)
        
        condition_buy = (df['MACD'] > df['MACD_signal']) & \
                        (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)) & \
                        (df['MACD'] > 0)
        
        condition_sell = (df['MACD'] < df['MACD_signal']) & \
                         (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)) & \
                         (df['MACD'] < 0)
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def rsi_signal(df: pd.DataFrame, lower: float = 30, upper: float = 70) -> pd.DataFrame:
        if 'RSI' not in df.columns:
            from strategy.factors import factor_library
            df = factor_library.calculate_rsi(df)
        
        condition_buy = (df['RSI'] < lower) & (df['RSI'].shift(1) >= lower)
        condition_sell = (df['RSI'] > upper) & (df['RSI'].shift(1) <= upper)
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def bollinger_signal(df: pd.DataFrame) -> pd.DataFrame:
        if 'BB_lower' not in df.columns or 'BB_upper' not in df.columns:
            from strategy.factors import factor_library
            df = factor_library.calculate_bollinger_bands(df)
        
        condition_buy = (df['close'] < df['BB_lower']) & (df['close'].shift(1) >= df['BB_lower'].shift(1))
        condition_sell = (df['close'] > df['BB_upper']) & (df['close'].shift(1) <= df['BB_upper'].shift(1))
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def kdj_signal(df: pd.DataFrame) -> pd.DataFrame:
        if 'K' not in df.columns or 'D' not in df.columns:
            from strategy.factors import factor_library
            df = factor_library.calculate_kdj(df)
        
        condition_buy = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)) & (df['K'] < 30)
        condition_sell = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)) & (df['K'] > 70)
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def dual_thrust_signal(df: pd.DataFrame, n: int = 20, k1: float = 0.5, k2: float = 0.5) -> pd.DataFrame:
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return df
        
        hh = df['high'].rolling(window=n).max().shift(1)
        hc = df['close'].rolling(window=n).max().shift(1)
        lc = df['close'].rolling(window=n).min().shift(1)
        ll = df['low'].rolling(window=n).min().shift(1)
        
        range_up = hh - lc
        range_down = hc - ll
        
        up_thrust = df['open'] + k1 * range_up
        down_thrust = df['open'] - k2 * range_down
        
        condition_buy = df['close'] > up_thrust
        condition_sell = df['close'] < down_thrust
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def turtle_signal(df: pd.DataFrame, entry_period: int = 20, exit_period: int = 10) -> pd.DataFrame:
        if not all(col in df.columns for col in ['high', 'low']):
            return df
        
        df['entry_high'] = df['high'].rolling(window=entry_period).max().shift(1)
        df['entry_low'] = df['low'].rolling(window=entry_period).min().shift(1)
        df['exit_high'] = df['high'].rolling(window=exit_period).max().shift(1)
        df['exit_low'] = df['low'].rolling(window=exit_period).min().shift(1)
        
        condition_buy = df['close'] > df['entry_high']
        condition_sell = df['close'] < df['entry_low']
        
        df.loc[condition_buy, 'signal'] = SignalGenerator.BUY
        df.loc[condition_sell, 'signal'] = SignalGenerator.SELL
        
        return df
    
    @staticmethod
    def custom_signal(df: pd.DataFrame, signal_func, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df['signal'] = signal_func(df, **kwargs)
        df['position'] = df['signal'].shift(1).fillna(0)
        return df

signal_generator = SignalGenerator()
