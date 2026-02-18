import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from logger import log
from strategy.factors import factor_library
from strategy.signals import signal_generator
from strategy.strategy_config import strategy_config
from utils.progress_bar import create_progress_bar


class RecommendationEngine:
    
    def __init__(self, use_config: bool = True):
        self.use_config = use_config
    
    def generate_short_term_recommendations(self, stock_data: Dict[str, pd.DataFrame], top_n: int = 5) -> List[Dict]:
        recommendations = []
        total = len(stock_data)
        pb = create_progress_bar(total, '计算短线推荐')
        
        for i, (ts_code, df) in enumerate(stock_data.items()):
            if df.empty or len(df) < 60:
                pb.update(i + 1)
                continue
            
            if 'name' in df.columns and not df['name'].empty:
                stock_name = df['name'].iloc[0]
                if 'ST' in stock_name or '*ST' in stock_name:
                    pb.update(i + 1)
                    continue
            
            score = self._calculate_short_term_score(df)
            reason = self._explain_short_term_reason(df, ts_code)
            
            recommendations.append({
                'ts_code': ts_code,
                'name': df['name'].iloc[0] if 'name' in df.columns and not df['name'].empty else ts_code,
                'score': score,
                'reason': reason,
                'current_price': df['close'].iloc[-1],
                'analysis': self._generate_buy_analysis(df, 'short')
            })
            pb.update(i + 1)
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def generate_medium_long_term_recommendations(self, stock_data: Dict[str, pd.DataFrame], top_n: int = 5) -> List[Dict]:
        recommendations = []
        total = len(stock_data)
        pb = create_progress_bar(total, '计算中线推荐')
        
        for i, (ts_code, df) in enumerate(stock_data.items()):
            if df.empty or len(df) < 90:
                pb.update(i + 1)
                continue
            
            if 'name' in df.columns and not df['name'].empty:
                stock_name = df['name'].iloc[0]
                if 'ST' in stock_name or '*ST' in stock_name:
                    pb.update(i + 1)
                    continue
            
            score = self._calculate_medium_long_score(df)
            reason = self._explain_medium_long_reason(df, ts_code)
            
            recommendations.append({
                'ts_code': ts_code,
                'name': df['name'].iloc[0] if 'name' in df.columns and not df['name'].empty else ts_code,
                'score': score,
                'reason': reason,
                'current_price': df['close'].iloc[-1],
                'analysis': self._generate_buy_analysis(df, 'medium')
            })
            pb.update(i + 1)
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def generate_long_term_recommendations(self, stock_data: Dict[str, pd.DataFrame], top_n: int = 5) -> List[Dict]:
        recommendations = []
        total = len(stock_data)
        pb = create_progress_bar(total, '计算长线推荐')
        
        for i, (ts_code, df) in enumerate(stock_data.items()):
            if df.empty or len(df) < 120:
                pb.update(i + 1)
                continue
            
            if 'name' in df.columns and not df['name'].empty:
                stock_name = df['name'].iloc[0]
                if 'ST' in stock_name or '*ST' in stock_name:
                    pb.update(i + 1)
                    continue
            
            score = self._calculate_long_score(df)
            reason = self._explain_long_reason(df, ts_code)
            
            recommendations.append({
                'ts_code': ts_code,
                'name': df['name'].iloc[0] if 'name' in df.columns and not df['name'].empty else ts_code,
                'score': score,
                'reason': reason,
                'current_price': df['close'].iloc[-1],
                'analysis': self._generate_buy_analysis(df, 'long')
            })
            pb.update(i + 1)
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def _calculate_short_term_score(self, df: pd.DataFrame) -> float:
        score = 0.0
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        ma5 = df['MA5'].iloc[-1] if 'MA5' in df.columns else current_price
        ma10 = df['MA10'].iloc[-1] if 'MA10' in df.columns else current_price
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        
        if self.use_config:
            weights = strategy_config.get_weights('short')
        else:
            weights = {}
        
        if current_price > ma5:
            score += weights.get('ma5_weight', 15)
        if ma5 > ma10:
            score += weights.get('ma5_ma10_weight', 10)
        if ma10 > ma20:
            score += weights.get('ma10_ma20_weight', 10)
        
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            if macd > macd_signal and macd > 0:
                score += weights.get('macd_positive_weight', 20)
            elif macd > macd_signal:
                score += weights.get('macd_cross_weight', 10)
        
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if 30 <= rsi <= 50:
                score += weights.get('rsi_low_weight', 15)
            elif 50 < rsi <= 70:
                score += weights.get('rsi_mid_weight', 10)
        
        if 'K' in df.columns and 'D' in df.columns:
            k = df['K'].iloc[-1]
            d = df['D'].iloc[-1]
            if k > d and k < 50:
                score += weights.get('kdj_weight', 15)
        
        recent_volume = df['volume'].iloc[-5:].mean()
        prev_volume = df['volume'].iloc[-10:-5].mean()
        if recent_volume > prev_volume * 1.2:
            score += 5
        
        return min(score, 100)
    
    def _calculate_medium_long_score(self, df: pd.DataFrame) -> float:
        score = 0.0
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
        
        if self.use_config:
            weights = strategy_config.get_weights('medium')
        else:
            weights = {}
        
        if current_price > ma20:
            score += weights.get('ma20_weight', 10)
        if ma20 > ma60:
            score += weights.get('ma20_ma60_weight', 15)
        
        if 'MACD' in df.columns:
            macd = df['MACD'].iloc[-1]
            if macd > 0:
                score += weights.get('macd_weight', 15)
        
        prices_60 = df['close'].iloc[-60:]
        if prices_60.iloc[-1] > prices_60.iloc[0]:
            score += weights.get('trend_weight', 20)
        
        if 'OBV' in df.columns:
            obv_trend = df['OBV'].iloc[-20:].mean() > df['OBV'].iloc[-40:-20].mean()
            if obv_trend:
                score += weights.get('obv_weight', 15)
        
        volatility = df['close'].pct_change().iloc[-60:].std()
        if volatility < 0.03:
            score += weights.get('volatility_weight', 15)
        
        return min(score, 100)
    
    def _calculate_long_score(self, df: pd.DataFrame) -> float:
        score = 0.0
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        data_len = len(df)
        
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
        ma120 = df['MA120'].iloc[-1] if ('MA120' in df.columns and data_len >= 120) else ma60
        
        if self.use_config:
            weights = strategy_config.get_weights('long')
        else:
            weights = {}
        
        if current_price > ma20:
            score += weights.get('ma20_weight', 10)
        if ma20 > ma60:
            score += weights.get('ma20_ma60_weight', 15)
        if data_len >= 120 and ma60 > ma120:
            score += weights.get('ma60_ma120_weight', 20)
        
        lookback_days = min(data_len, 120)
        prices_period = df['close'].iloc[-lookback_days:]
        if prices_period.iloc[-1] > prices_period.iloc[0]:
            score += weights.get('trend_weight', 25)
        
        max_period = prices_period.max()
        if current_price > max_period * 0.8:
            score += weights.get('near_high_weight', 15)
        
        if 'OBV' in df.columns:
            obv_short = min(data_len, 30)
            obv_long = min(data_len, 60)
            if obv_long > obv_short:
                obv_trend = df['OBV'].iloc[-obv_short:].mean() > df['OBV'].iloc[-obv_long:-obv_short].mean()
                if obv_trend:
                    score += weights.get('obv_weight', 15)
        
        return min(score, 100)
    
    def _explain_short_term_reason(self, df: pd.DataFrame, ts_code: str) -> str:
        reasons = []
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        ma5 = df['MA5'].iloc[-1] if 'MA5' in df.columns else current_price
        ma10 = df['MA10'].iloc[-1] if 'MA10' in df.columns else current_price
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        
        if current_price > ma5:
            reasons.append("股价站上5日均线，短期强势")
        if ma5 > ma10:
            reasons.append("5日均线上穿10日均线，形成金叉")
        if ma10 > ma20:
            reasons.append("均线系统呈多头排列")
        
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            if macd > macd_signal and macd > 0:
                reasons.append("MACD金叉且位于零轴上方，上涨动能充足")
            elif macd > macd_signal:
                reasons.append("MACD金叉，短期有望反弹")
        
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if 30 <= rsi <= 50:
                reasons.append(f"RSI={rsi:.1f}，处于合理区间，有上涨空间")
        
        if 'K' in df.columns and 'D' in df.columns:
            k = df['K'].iloc[-1]
            d = df['D'].iloc[-1]
            if k > d and k < 50:
                reasons.append("KDJ低位金叉，超卖反弹信号")
        
        recent_volume = df['volume'].iloc[-5:].mean()
        prev_volume = df['volume'].iloc[-10:-5].mean()
        if recent_volume > prev_volume * 1.2:
            reasons.append("近期成交量放大，资金关注度提升")
        
        if not reasons:
            reasons.append("技术面综合评分较高，值得关注")
        
        return "；".join(reasons)
    
    def _explain_medium_long_reason(self, df: pd.DataFrame, ts_code: str) -> str:
        reasons = []
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
        
        if current_price > ma20 and ma20 > ma60:
            reasons.append("中期均线多头排列，趋势向上")
        
        prices_60 = df['close'].iloc[-60:]
        if prices_60.iloc[-1] > prices_60.iloc[0]:
            gain = (prices_60.iloc[-1] / prices_60.iloc[0] - 1) * 100
            reasons.append(f"近60日上涨{gain:.1f}%，中期走势稳健")
        
        if 'OBV' in df.columns:
            obv_trend = df['OBV'].iloc[-20:].mean() > df['OBV'].iloc[-40:-20].mean()
            if obv_trend:
                reasons.append("OBV能量潮持续上升，资金持续流入")
        
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            price_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            if atr / price_range < 0.5:
                reasons.append("波动率适中，走势相对平稳")
        
        volatility = df['close'].pct_change().iloc[-60:].std()
        if volatility < 0.03:
            reasons.append("波动率较低，风险可控")
        
        if not reasons:
            reasons.append("中期技术面表现良好，具备投资价值")
        
        return "；".join(reasons)
    
    def _explain_long_reason(self, df: pd.DataFrame, ts_code: str) -> str:
        reasons = []
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        data_len = len(df)
        
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
        ma120 = df['MA120'].iloc[-1] if ('MA120' in df.columns and data_len >= 120) else ma60
        
        if ma20 > ma60 and (data_len < 120 or ma60 > ma120):
            reasons.append("均线系统多头排列，趋势向上")
        
        lookback_days = min(data_len, 120)
        prices_period = df['close'].iloc[-lookback_days:]
        if prices_period.iloc[-1] > prices_period.iloc[0]:
            gain = (prices_period.iloc[-1] / prices_period.iloc[0] - 1) * 100
            period_desc = f"近{lookback_days}日" if lookback_days < 120 else "近半年"
            reasons.append(f"{period_desc}上涨{gain:.1f}%，走势稳健")
        
        max_period = prices_period.max()
        if current_price > max_period * 0.8:
            reasons.append("股价接近阶段新高，上涨空间打开")
        
        if 'OBV' in df.columns:
            obv_short = min(data_len, 30)
            obv_long = min(data_len, 60)
            if obv_long > obv_short:
                obv_trend = df['OBV'].iloc[-obv_short:].mean() > df['OBV'].iloc[-obv_long:-obv_short].mean()
                if obv_trend:
                    reasons.append("资金持续流入，主力看好")
        
        if not reasons:
            reasons.append("技术面表现良好，适合投资")
        
        return "；".join(reasons)
    
    def _generate_buy_analysis(self, df: pd.DataFrame, period: str) -> Dict:
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        
        if period == 'short':
            target_price = current_price * 1.08
            stop_loss = current_price * 0.95
            holding_period = "1-5个交易日"
        elif period == 'medium':
            target_price = current_price * 1.20
            stop_loss = current_price * 0.90
            holding_period = "1-3个月"
        else:
            target_price = current_price * 1.50
            stop_loss = current_price * 0.85
            holding_period = "6个月以上"
        
        return {
            'target_price': target_price,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'entry_suggestion': f"建议在{current_price:.2f}附近分批买入",
            'risk_control': f"跌破{stop_loss:.2f}建议止损"
        }
    
    def backtest_strategy(self, df: pd.DataFrame, strategy_name: str = 'ma_cross', **kwargs) -> Dict:
        from backtest.engine import backtest_engine
        
        df = df.copy()
        df = signal_generator.generate_signals(df, strategy_name, **kwargs)
        
        results = backtest_engine.run(df)
        return results


recommendation_engine = RecommendationEngine()
