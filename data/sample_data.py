"""样本数据生成模块"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta


class SampleDataGenerator:
    """样本数据生成器"""
    
    def __init__(self):
        """初始化样本数据生成器"""
        pass
    
    def generate_single_stock_data(self, ts_code: str = '000000.SZ', name: str = '演示股票', days: int = 252) -> pd.DataFrame:
        """
        生成单支股票的样本数据
        
        Args:
            ts_code: 股票代码
            name: 股票名称
            days: 数据天数
        
        Returns:
            包含股票数据的DataFrame
        """
        # 生成日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)
        
        # 生成交易日历（跳过周末）
        dates = []
        current_date = start_date
        while current_date <= end_date and len(dates) < days:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        n_days = len(dates)
        
        # 生成价格数据
        base_price = 10.0
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base_price * (1 + returns).cumprod()
        
        # 生成OHLC数据
        open_prices = prices * (1 + np.random.normal(0, 0.005, n_days))
        high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        close_prices = prices
        
        # 生成成交量
        volume = np.random.lognormal(10, 1, n_days).astype(int)
        
        # 生成其他字段
        pre_close = np.roll(close_prices, 1)
        pre_close[0] = close_prices[0]
        change = close_prices - pre_close
        pct_chg = (change / pre_close) * 100
        amount = close_prices * volume
        
        # 创建DataFrame
        df = pd.DataFrame({
            'ts_code': ts_code,
            'name': name,
            'trade_date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'pre_close': pre_close,
            'change': change,
            'pct_chg': pct_chg,
            'vol': volume,
            'volume': volume,
            'amount': amount
        })
        
        return df
    
    def generate_sample_stock_data(self, num_stocks: int = 10, days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        生成多支股票的样本数据
        
        Args:
            num_stocks: 股票数量
            days: 每支股票的数据天数
        
        Returns:
            股票数据字典
        """
        stock_data = {}
        
        for i in range(num_stocks):
            ts_code = f'{i:06d}.SZ'
            name = f'样本股票{i+1}'
            df = self.generate_single_stock_data(ts_code, name, days)
            stock_data[ts_code] = df
        
        return stock_data
    
    def generate_backtest_demo(self, days: int = 252) -> pd.DataFrame:
        """
        生成回测演示数据
        
        Args:
            days: 数据天数
        
        Returns:
            演示数据DataFrame
        """
        return self.generate_single_stock_data(days=days)


sample_data_generator = SampleDataGenerator()
