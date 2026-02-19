import pandas as pd
import numpy as np
from typing import Optional, Union
from logger import log

class DataCleaner:
    @staticmethod
    def clean_daily_data(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        
        log.info(f"开始清洗数据，原始数据条数: {len(df)}")
        
        df = DataCleaner.handle_missing_values(df)
        df = DataCleaner.handle_outliers(df)
        df = DataCleaner.validate_data_ranges(df)
        
        log.info(f"数据清洗完成，清洗后数据条数: {len(df)}")
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        existing_cols = [col for col in numeric_cols if col in df.columns]
        
        if existing_cols:
            df[existing_cols] = df[existing_cols].fillna(method='ffill').fillna(method='bfill')
        
        if 'vol' in df.columns:
            df['vol'] = df['vol'].fillna(0)
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0)
        
        return df
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, method: str = 'iqr', sigma: float = 3.0) -> pd.DataFrame:
        if method not in ['iqr', 'sigma']:
            method = 'iqr'
        
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in df.columns]
        
        if not existing_price_cols:
            return df
        
        for col in existing_price_cols:
            if method == 'sigma':
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - sigma * std
                upper = mean + sigma * std
            else:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        if 'pct_chg' in df.columns:
            df['pct_chg'] = df['pct_chg'].clip(lower=-20, upper=20)
        
        return df
    
    @staticmethod
    def validate_data_ranges(df: pd.DataFrame) -> pd.DataFrame:
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df = df[(df['high'] >= df['low']) & 
                    (df['high'] >= df['open']) & 
                    (df['high'] >= df['close']) & 
                    (df['low'] <= df['open']) & 
                    (df['low'] <= df['close'])]
        
        if 'vol' in df.columns:
            df = df[df['vol'] >= 0]
        if 'amount' in df.columns:
            df = df[df['amount'] >= 0]
        
        return df
    
    @staticmethod
    def adjust_for_splits_and_dividends(df: pd.DataFrame, 
                                         adjust_factor: Optional[pd.Series] = None,
                                         adjust_type: str = 'qfq') -> pd.DataFrame:
        if adjust_factor is None:
            return df
        
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close', 'pre_close']
        
        for col in price_cols:
            if col in df.columns:
                if adjust_type == 'qfq':
                    df[col] = df[col] * adjust_factor
                else:
                    df[col] = df[col] / adjust_factor
        
        return df
    
    @staticmethod
    def resample_data(df: pd.DataFrame, freq: str = 'D', date_col: str = 'trade_date') -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum',
            'amount': 'sum'
        }
        
        existing_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
        resampled = df.resample(freq).agg(existing_agg).dropna()
        
        return resampled.reset_index()
    
    @staticmethod
    def merge_dataframes(main_df: pd.DataFrame, 
                         add_df: pd.DataFrame, 
                         on: Union[str, list], 
                         how: str = 'left') -> pd.DataFrame:
        merged = pd.merge(main_df, add_df, on=on, how=how)
        log.info(f"合并数据后条数: {len(merged)}")
        return merged

data_cleaner = DataCleaner()
