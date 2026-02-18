import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from logger import log

class FactorLibrary:
    @staticmethod
    def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        
        df = FactorLibrary.calculate_ma(df)
        df = FactorLibrary.calculate_ema(df)
        df = FactorLibrary.calculate_macd(df)
        df = FactorLibrary.calculate_rsi(df)
        df = FactorLibrary.calculate_bollinger_bands(df)
        df = FactorLibrary.calculate_kdj(df)
        df = FactorLibrary.calculate_atr(df)
        df = FactorLibrary.calculate_obv(df)
        df = FactorLibrary.calculate_momentum(df)
        df = FactorLibrary.calculate_volume_factors(df)
        
        return df
    
    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
        if 'close' not in df.columns:
            return df
        
        for period in periods:
            df[f'MA{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
        if 'close' not in df.columns:
            return df
        
        for period in periods:
            df[f'EMA{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        if 'close' not in df.columns:
            return df
        
        df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        if 'close' not in df.columns:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        if 'close' not in df.columns:
            return df
        
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std_dev)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        return df
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return df
        
        low_list = df['low'].rolling(window=n, min_periods=1).min()
        high_list = df['high'].rolling(window=n, min_periods=1).max()
        
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return df
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['TR'] = tr
        df['ATR'] = tr.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in ['close', 'vol']):
            return df
        
        obv = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        df['OBV'] = obv
        
        return df
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        if 'close' not in df.columns:
            return df
        
        for period in periods:
            df[f'MOM{period}'] = df['close'].pct_change(periods=period) * 100
        
        df['ROC'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100
        
        return df
    
    @staticmethod
    def calculate_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
        if 'vol' not in df.columns:
            return df
        
        df['VOL_MA5'] = df['vol'].rolling(window=5).mean()
        df['VOL_MA10'] = df['vol'].rolling(window=10).mean()
        df['VOL_RATIO'] = df['vol'] / df['VOL_MA5']
        
        return df
    
    @staticmethod
    def custom_factor(df: pd.DataFrame, factor_func, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df = factor_func(df, **kwargs)
        return df

factor_library = FactorLibrary()
