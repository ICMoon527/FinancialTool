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
    
    def _calculate_support_resistance(self, df: pd.DataFrame, period: str) -> Dict[str, float]:
        """
        计算支撑位和阻力位（改进版）
        
        Args:
            df: 股票历史数据
            period: 周期 ('short', 'medium', 'long')
        
        Returns:
            包含支撑位和阻力位的字典
        """
        current_price = df['close'].iloc[-1]
        
        if period == 'short':
            # 短线：使用最近20天数据
            window = 20
            recent_df = df.iloc[-window:]
        elif period == 'medium':
            # 中长线：使用最近60天数据
            window = 60
            recent_df = df.iloc[-window:]
        else:  # long
            # 长线：使用最近120天数据
            window = 120
            recent_df = df.iloc[-window:]
        
        # 1. 基于最近高低点的支撑/阻力
        recent_high = recent_df['high'].max()
        recent_low = recent_df['low'].min()
        price_range = recent_high - recent_low
        
        # 2. 结合成交量判断关键价位
        volume_weighted_prices = []
        for i in range(len(recent_df)):
            if recent_df['volume'].iloc[i] > recent_df['volume'].mean():
                volume_weighted_prices.append(recent_df['close'].iloc[i])
        
        # 3. 计算斐波那契回撤位
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        fib_levels = sorted(fib_levels, reverse=True)
        
        # 4. 考虑均线作为动态支撑/阻力
        ma_levels = {}
        if 'MA20' in df.columns:
            ma_levels['MA20'] = df['MA20'].iloc[-1]
        if 'MA60' in df.columns:
            ma_levels['MA60'] = df['MA60'].iloc[-1]
        if 'MA120' in df.columns and len(df) >= 120:
            ma_levels['MA120'] = df['MA120'].iloc[-1]
        
        # 5. 收集所有潜在支撑/阻力位
        potential_levels = set()
        potential_levels.add(recent_high)
        potential_levels.add(recent_low)
        
        # 添加斐波那契水平
        for level in fib_levels:
            potential_levels.add(recent_low + price_range * level)
            potential_levels.add(recent_high - price_range * level)
        
        # 添加成交量加权价格
        for price in volume_weighted_prices[:5]:  # 取成交量最大的5个价位
            potential_levels.add(price)
        
        # 添加均线水平
        for ma_name, ma_value in ma_levels.items():
            potential_levels.add(ma_value)
        
        # 6. 过滤和排序支撑/阻力位
        potential_levels = sorted(list(potential_levels))
        
        # 找出当前价格附近的支撑和阻力位
        resistance_levels = [level for level in potential_levels if level > current_price]
        support_levels = [level for level in potential_levels if level < current_price]
        
        # 排序并选择最近的支撑/阻力位
        resistance_levels.sort()
        support_levels.sort(reverse=True)
        
        # 确保有足够的支撑/阻力位
        while len(resistance_levels) < 2:
            # 如果阻力位不足，基于最近高点和价格范围生成
            if resistance_levels:
                next_resistance = resistance_levels[-1] + price_range * 0.236
            else:
                next_resistance = recent_high + price_range * 0.236
            resistance_levels.append(next_resistance)
        
        while len(support_levels) < 2:
            # 如果支撑位不足，基于最近低点和价格范围生成
            if support_levels:
                next_support = support_levels[-1] - price_range * 0.236
            else:
                next_support = recent_low - price_range * 0.236
            support_levels.append(next_support)
        
        # 选择最近的支撑/阻力位
        resistance1 = resistance_levels[0]
        resistance2 = resistance_levels[1]
        support1 = support_levels[0]
        support2 = support_levels[1]
        
        return {
            'resistance1': resistance1,
            'resistance2': resistance2,
            'support1': support1,
            'support2': support2
        }
    
    def _explain_short_term_reason(self, df: pd.DataFrame, ts_code: str) -> str:
        reasons = []
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        ma5 = df['MA5'].iloc[-1] if 'MA5' in df.columns else current_price
        ma10 = df['MA10'].iloc[-1] if 'MA10' in df.columns else current_price
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        
        # 计算支撑位和阻力位
        sr_levels = self._calculate_support_resistance(df, 'short')
        
        # 均线系统详细分析
        if current_price > ma5:
            price_ma5_diff = (current_price - ma5) / ma5 * 100
            if price_ma5_diff > 2:
                reasons.append(f"股价强势站上5日均线{price_ma5_diff:.1f}%，短期走势强劲")
            else:
                reasons.append(f"股价站上5日均线，短期企稳")
        if ma5 > ma10:
            ma5_ma10_diff = (ma5 - ma10) / ma10 * 100
            reasons.append(f"5日均线上穿10日均线{ma5_ma10_diff:.2f}%，形成金叉")
        if ma10 > ma20:
            ma10_ma20_diff = (ma10 - ma20) / ma20 * 100
            reasons.append(f"10日均线上穿20日均线{ma10_ma20_diff:.2f}%，均线系统呈多头排列")
        
        # MACD详细分析
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            macd_diff = macd - macd_signal
            if macd > macd_signal and macd > 0:
                reasons.append(f"MACD金叉且位于零轴上方，差值为{macd_diff:.3f}，上涨动能充足")
            elif macd > macd_signal:
                reasons.append(f"MACD金叉，差值为{macd_diff:.3f}，短期有望反弹")
        
        # KDJ详细分析
        if 'K' in df.columns and 'D' in df.columns and 'J' in df.columns:
            k = df['K'].iloc[-1]
            d = df['D'].iloc[-1]
            j = df['J'].iloc[-1]
            if k > d and k < 50:
                reasons.append(f"KDJ低位金叉，K值={k:.1f}，D值={d:.1f}，J值={j:.1f}，超卖反弹信号")
        
        # 成交量详细分析
        recent_volume = df['volume'].iloc[-5:].mean()
        prev_volume = df['volume'].iloc[-10:-5].mean()
        if recent_volume > prev_volume * 1.2:
            volume_ratio = recent_volume / prev_volume
            reasons.append(f"近期成交量放大{volume_ratio:.1f}倍，资金关注度显著提升")
        
        # 波动率分析
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            atr_ratio = atr / current_price * 100
            if atr_ratio < 2:
                reasons.append(f"波动率较低，ATR={atr:.2f}，适合稳健操作")
            elif atr_ratio < 5:
                reasons.append(f"波动率适中，ATR={atr:.2f}，有一定操作空间")
        
        # 支撑位和阻力位分析
        reasons.append(f"短期阻力位1: {sr_levels['resistance1']:.2f}，阻力位2: {sr_levels['resistance2']:.2f}")
        reasons.append(f"短期支撑位1: {sr_levels['support1']:.2f}，支撑位2: {sr_levels['support2']:.2f}")
        
        # 风险评估
        risk_factors = []
        if len(df) > 20:
            recent_volatility = df['close'].pct_change().iloc[-20:].std() * 100
            if recent_volatility > 3:
                risk_factors.append(f"短期波动率较高({recent_volatility:.1f}%)，注意风险控制")
        
        if not reasons:
            reasons.append("技术面综合评分较高，值得关注")
        
        # 合并风险因素
        if risk_factors:
            reasons.extend(risk_factors)
        
        return "；".join(reasons)
    
    def _explain_medium_long_reason(self, df: pd.DataFrame, ts_code: str) -> str:
        reasons = []
        
        df = df.copy()
        df = factor_library.calculate_all_factors(df)
        
        current_price = df['close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        ma60 = df['MA60'].iloc[-1] if 'MA60' in df.columns else current_price
        
        # 计算支撑位和阻力位
        sr_levels = self._calculate_support_resistance(df, 'medium')
        
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
        
        # 支撑位和阻力位分析
        reasons.append(f"中期阻力位1: {sr_levels['resistance1']:.2f}，阻力位2: {sr_levels['resistance2']:.2f}")
        reasons.append(f"中期支撑位1: {sr_levels['support1']:.2f}，支撑位2: {sr_levels['support2']:.2f}")
        
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
        
        # 计算支撑位和阻力位
        sr_levels = self._calculate_support_resistance(df, 'long')
        
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
        
        # 支撑位和阻力位分析
        reasons.append(f"长期阻力位1: {sr_levels['resistance1']:.2f}，阻力位2: {sr_levels['resistance2']:.2f}")
        reasons.append(f"长期支撑位1: {sr_levels['support1']:.2f}，支撑位2: {sr_levels['support2']:.2f}")
        
        if not reasons:
            reasons.append("技术面表现良好，适合投资")
        
        return "；".join(reasons)
    
    def _generate_buy_analysis(self, df: pd.DataFrame, period: str) -> Dict:
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        
        if period == 'short':
            # 基于支撑位、阻力位和波动率计算
            buy_price = current_price * 0.995  # 略微低于收盘价
            
            # 基于ATR的止损位
            if atr > 0:
                stop_loss = current_price - atr * 1.5
            else:
                stop_loss = current_price * 0.95
            
            # 基于目标收益率的止盈位
            target_price = current_price * 1.08
            
            # 基于阻力位的调整
            if target_price > recent_high:
                target_price = recent_high * 0.99
            
            holding_period = "1-2个交易日"
            execution_time = "下一个交易日开盘后15分钟内"
            exit_time = "下下个交易日收盘前"
        elif period == 'medium':
            buy_price = current_price * 0.99
            if atr > 0:
                stop_loss = current_price - atr * 2.0
            else:
                stop_loss = current_price * 0.90
            target_price = current_price * 1.20
            holding_period = "1-3个月"
            execution_time = "下一个交易日"
            exit_time = "目标价附近或持有期结束"
        else:
            buy_price = current_price * 0.98
            if atr > 0:
                stop_loss = current_price - atr * 3.0
            else:
                stop_loss = current_price * 0.85
            target_price = current_price * 1.50
            holding_period = "6个月以上"
            execution_time = "分批买入"
            exit_time = "目标价附近或基本面变化"
        
        # 计算潜在收益和风险
        potential_gain = (target_price - buy_price) / buy_price * 100
        potential_loss = (buy_price - stop_loss) / buy_price * 100
        risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
        
        return {
            'buy_price': buy_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'execution_time': execution_time,
            'exit_time': exit_time,
            'potential_gain': potential_gain,
            'potential_loss': potential_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'entry_suggestion': f"建议在{buy_price:.2f}附近分批买入",
            'risk_control': f"跌破{stop_loss:.2f}建议止损",
            'take_profit': f"达到{target_price:.2f}建议止盈"
        }
    
    def backtest_strategy(self, df: pd.DataFrame, strategy_name: str = 'ma_cross', **kwargs) -> Dict:
        from backtest.engine import backtest_engine
        
        df = df.copy()
        df = signal_generator.generate_signals(df, strategy_name, **kwargs)
        
        results = backtest_engine.run(df)
        return results


recommendation_engine = RecommendationEngine()
