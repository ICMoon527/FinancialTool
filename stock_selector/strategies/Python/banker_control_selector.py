import pandas as pd
import numpy as np
from ..base import BaseIndicator


class BankerControlSelector(BaseIndicator):
    """
    庄家控盘选股模块
    
    该模块基于庄家控盘指标筛选股票，筛选出最新控盘度大于50的个股。
    
    公式：
    - AAA：(3 * Close + Open + High + Low) / 6（价格基准）
    - MA12：AAA的12周期EMA
    - MA36：AAA的36周期EMA
    - 控盘度：(MA12 - REF(MA36, 1)) / REF(MA36, 1) * 100 + 50
    
    控盘级别：
    - 低控盘：control_degree >= 50
    - 中控盘：control_degree >= 60
    - 高控盘：control_degree >= 80
    """

    def __init__(self):
        super().__init__()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算庄家控盘指标。
        
        Args:
            data: 包含OHLCV数据的输入DataFrame
        
        Returns:
            添加了庄家控盘相关列的DataFrame
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        result = self._calculate_control_degree(df, result)

        return result

    def _calculate_control_degree(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """计算控盘度指标"""
        
        close = df['Close']
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        
        aaa = (3 * close + open_price + high + low) / 6
        result['aaa'] = aaa
        
        ma12 = self._ema(aaa, 12)
        ma36 = self._ema(aaa, 36)
        result['ma12'] = ma12
        result['ma36'] = ma36
        
        ma36_prev = ma36.shift(1)
        control_degree = (ma12 - ma36_prev) / ma36_prev * 100 + 50
        result['control_degree'] = control_degree
        
        result['low_control'] = (control_degree >= 50).astype(int)
        result['medium_control'] = (control_degree >= 60).astype(int)
        result['high_control'] = (control_degree >= 80).astype(int)
        
        return result

    @staticmethod
    def select_control_stocks(data_dict: dict, min_control: float = 50.0) -> list:
        """
        从多只股票中筛选控盘度大于指定阈值的股票
        
        Args:
            data_dict: 字典，key为股票代码，value为该股票的DataFrame（包含计算结果）
            min_control: 最小控盘度阈值，默认为50
        
        Returns:
            筛选后的股票列表，包含股票代码和最新控盘度
        """
        stock_scores = []
        
        for stock_code, df in data_dict.items():
            if 'control_degree' in df.columns and len(df) > 0:
                latest_control = df['control_degree'].iloc[-1]
                if pd.notna(latest_control) and latest_control >= min_control:
                    stock_scores.append({
                        'stock_code': stock_code,
                        'control_degree': latest_control,
                        'low_control': bool(df['low_control'].iloc[-1]),
                        'medium_control': bool(df['medium_control'].iloc[-1]),
                        'high_control': bool(df['high_control'].iloc[-1]),
                        'latest_data': df.iloc[-1:].copy()
                    })
        
        stock_scores.sort(key=lambda x: x['control_degree'], reverse=True)
        
        return stock_scores
