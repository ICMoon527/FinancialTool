import pandas as pd
import numpy as np
from ..base import BaseIndicator


class TopRiseStockSelector(BaseIndicator):
    """
    领涨选股模块
    
    该模块整合多个技术分析维度，筛选潜在的领涨股票，包括：
    1. 涨幅排行 - 短期涨幅表现
    2. 量能异动 - 成交量放大确认
    3. 资金流向 - 主力资金净流入
    4. 技术形态 - 突破关键压力位
    5. 龙头信号 - 整合龙头启动、主升、二波信号
    
    使用方式：
    - 对单只股票数据：返回该股票的领涨评分和信号
    - 对多只股票组合：可进行综合排序筛选
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).min()

    def _count(self, condition: pd.Series, period: int) -> pd.Series:
        return condition.astype(int).rolling(window=period).sum()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算领涨选股评分和信号
        
        Args:
            data: 包含OHLCV数据的输入DataFrame
        
        Returns:
            添加了领涨选股相关列的DataFrame
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        result = self._calculate_price_ranking(df, result)
        result = self._calculate_volume_analysis(df, result)
        result = self._calculate_capital_flow(df, result)
        result = self._calculate_technical_breakout(df, result)
        result = self._calculate_dragon_signals(df, result)
        result = self._calculate_composite_score(result)

        return result

    def _calculate_price_ranking(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """计算价格涨幅相关指标"""
        
        close = df['Close']
        
        change_pct = (close / close.shift(1) - 1) * 100
        result['change_pct'] = change_pct
        
        result['change_1d'] = change_pct
        result['change_3d'] = (close / close.shift(3) - 1) * 100
        result['change_5d'] = (close / close.shift(5) - 1) * 100
        result['change_10d'] = (close / close.shift(10) - 1) * 100
        
        result['price_ranking_score'] = (
            result['change_1d'] * 0.4 +
            result['change_3d'] * 0.3 +
            result['change_5d'] * 0.2 +
            result['change_10d'] * 0.1
        )
        
        return result

    def _calculate_volume_analysis(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """计算成交量相关指标"""
        
        volume = df['Volume']
        
        ma5_volume = self._sma(volume, 5)
        ma10_volume = self._sma(volume, 10)
        ma20_volume = self._sma(volume, 20)
        
        result['volume_ma5'] = ma5_volume
        result['volume_ma10'] = ma10_volume
        result['volume_ma20'] = ma20_volume
        
        result['volume_ratio_5'] = volume / ma5_volume
        result['volume_ratio_10'] = volume / ma10_volume
        
        hhv_volume_20 = self._hhv(volume, 20)
        result['volume_new_high'] = volume >= hhv_volume_20
        
        volume_increasing = (volume > volume.shift(1)) & (volume.shift(1) > volume.shift(2))
        result['volume_increasing_3d'] = volume_increasing
        
        result['volume_score'] = np.where(
            result['volume_ratio_5'] >= 2, 30,
            np.where(result['volume_ratio_5'] >= 1.5, 20,
            np.where(result['volume_ratio_5'] >= 1.2, 10, 0))
        )
        
        result['volume_score'] += np.where(result['volume_new_high'], 20, 0)
        result['volume_score'] += np.where(result['volume_increasing_3d'], 10, 0)
        
        return result

    def _calculate_capital_flow(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """计算资金流向相关指标"""
        
        close = df['Close']
        volume = df['Volume']
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        
        price_change = close - open_price
        
        if 'Amount' in df.columns:
            amount = df['Amount']
        else:
            amount = volume * close
        
        main_buy = amount.where(price_change > 0, 0)
        main_sell = amount.where(price_change < 0, 0)
        main_net = main_buy - main_sell
        
        result['main_buy'] = main_buy
        result['main_sell'] = main_sell
        result['main_net'] = main_net
        
        main_net_ma5 = self._sma(main_net, 5)
        result['main_net_ma5'] = main_net_ma5
        
        result['capital_inflow'] = main_net > 0
        result['capital_inflow_3d'] = (self._count(result['capital_inflow'], 3) == 3)
        
        result['capital_score'] = np.where(
            result['capital_inflow_3d'], 30,
            np.where(result['capital_inflow'], 15, 0)
        )
        
        return result

    def _calculate_technical_breakout(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """计算技术突破相关指标"""
        
        close = df['Close']
        high = df['High']
        
        ema20 = self._ema(close, 20)
        ema60 = self._ema(close, 60)
        
        result['ema20'] = ema20
        result['ema60'] = ema60
        
        hhv_high_20 = self._hhv(high, 20)
        hhv_high_60 = self._hhv(high, 60)
        
        result['breakout_20'] = close >= hhv_high_20.shift(1)
        result['breakout_60'] = close >= hhv_high_60.shift(1)
        
        result['above_ema20'] = close > ema20
        result['above_ema60'] = close > ema60
        
        result['ema_golden_cross'] = (ema20 > ema60) & (ema20.shift(1) <= ema60.shift(1))
        
        result['technical_score'] = np.where(result['breakout_60'], 40,
                                        np.where(result['breakout_20'], 25, 0))
        result['technical_score'] += np.where(result['above_ema60'], 15,
                                        np.where(result['above_ema20'], 10, 0))
        result['technical_score'] += np.where(result['ema_golden_cross'], 20, 0)
        
        return result

    def _calculate_dragon_signals(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """计算龙头信号相关指标"""
        
        close = df['Close']
        high = df['High']
        
        change_pct = (close / close.shift(1) - 1) * 100
        limit_up = change_pct >= 9.0
        
        result['limit_up'] = limit_up
        
        consecutive_limit_up = (self._count(limit_up, 2) == 2)
        result['consecutive_limit_up'] = consecutive_limit_up
        
        has_body = close > df['Low']
        result['has_body'] = has_body
        
        result['dragon_leader_signal'] = limit_up & has_body
        
        result['dragon_score'] = np.where(consecutive_limit_up, 40,
                                      np.where(limit_up, 25, 0))
        
        return result

    def _calculate_composite_score(self, result: pd.DataFrame) -> pd.DataFrame:
        """计算综合领涨评分"""
        
        result['top_rise_score'] = (
            result['price_ranking_score'] * 0.2 +
            result['volume_score'] * 0.25 +
            result['capital_score'] * 0.2 +
            result['technical_score'] * 0.2 +
            result['dragon_score'] * 0.15
        )
        
        result['top_rise_signal'] = result['top_rise_score'] >= 50
        result['top_rise_strong'] = result['top_rise_score'] >= 70
        
        result['signal_level'] = ''
        result.loc[result['top_rise_strong'], 'signal_level'] = '强信号'
        result.loc[(result['top_rise_signal'] & ~result['top_rise_strong']), 'signal_level'] = '中信号'
        
        return result

    @staticmethod
    def select_top_stocks(data_dict: dict, top_n: int = 10, sort_by: str = 'top_rise_score') -> list:
        """
        从多只股票中筛选领涨股票
        
        Args:
            data_dict: 字典，key为股票代码，value为该股票的DataFrame（包含计算结果）
            top_n: 返回前N只股票
            sort_by: 排序字段，默认为'top_rise_score'
        
        Returns:
            排序后的股票列表，包含股票代码和最新评分
        """
        stock_scores = []
        
        for stock_code, df in data_dict.items():
            if sort_by in df.columns and len(df) > 0:
                latest_score = df[sort_by].iloc[-1]
                if pd.notna(latest_score):
                    stock_scores.append({
                        'stock_code': stock_code,
                        'score': latest_score,
                        'latest_data': df.iloc[-1:].copy()
                    })
        
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return stock_scores[:top_n]
