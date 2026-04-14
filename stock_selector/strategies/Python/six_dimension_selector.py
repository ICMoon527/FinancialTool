# -*- coding: utf-8 -*-
"""
六维选股策略 - Python实现。
"""

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy
from stock_selector.config import get_config
from indicators.indicators.main_trading import MainTrading
from indicators.indicators.momentum_2 import Momentum2

logger = logging.getLogger(__name__)


class MarketDataCache:
    """大盘数据缓存管理器"""

    _CACHE_DIR = None
    _CACHE_VALID_HOURS = 24
    _cache = {}

    @classmethod
    def _get_cache_dir(cls) -> Path:
        """获取缓存目录路径"""
        if cls._CACHE_DIR is None:
            project_root = Path(__file__).parent.parent.parent.parent
            cls._CACHE_DIR = project_root / "data" / "cache"
            cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls._CACHE_DIR

    @classmethod
    def _get_cache_file_path(cls, symbol: str) -> Path:
        """获取缓存文件路径"""
        cache_dir = cls._get_cache_dir()
        return cache_dir / f"market_{symbol}.pkl"

    @classmethod
    def _is_cache_valid(cls, cache_file: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_file.exists():
            return False
        try:
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            return datetime.now() - file_mtime < timedelta(hours=cls._CACHE_VALID_HOURS)
        except Exception:
            return False

    @classmethod
    def load(cls, symbol: str) -> Optional[pd.DataFrame]:
        """从缓存加载大盘数据"""
        if symbol in cls._cache:
            return cls._cache[symbol]
        cache_file = cls._get_cache_file_path(symbol)
        if not cls._is_cache_valid(cache_file):
            return None
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"[大盘数据缓存] 从缓存加载 {symbol} 数据成功")
            cls._cache[symbol] = data
            return data
        except Exception as e:
            logger.warning(f"[大盘数据缓存] 加载缓存失败: {e}")
            return None

    @classmethod
    def save(cls, symbol: str, data: pd.DataFrame) -> None:
        """保存大盘数据到缓存"""
        cls._cache[symbol] = data
        cache_file = cls._get_cache_file_path(symbol)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[大盘数据缓存] 保存 {symbol} 数据到缓存成功")
        except Exception as e:
            logger.warning(f"[大盘数据缓存] 保存缓存失败: {e}")


@register_strategy
class SixDimensionSelectorStrategy(StockSelectorStrategy):
    """
    六维选股策略 - Python实现。

    该策略整合主力操盘、庄家控盘、动能二号、共振追涨、强势起爆五个策略的筛选逻辑。
    当任一子策略匹配时，六维策略即匹配，各维度评分独立，最后总分直接相加，排名按分数高低。
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="six_dimension_selector",
            name="six_dimension_selector",
            display_name="六维选股(Python)",
            description="Python实现的六维选股策略，整合主力操盘、庄家控盘、动能二号、共振追涨、强势起爆五个策略的筛选逻辑。",
            strategy_type=StrategyType.PYTHON,
            category="comprehensive",
            source="builtin",
            version="2.1.0",
            score_multiplier=1.0,
            max_raw_score=500.0,
        )
        super().__init__(metadata)
        self._main_trading_indicator = MainTrading()
        self._momentum2_indicator = Momentum2()
        
        # 注意：每次使用配置时都调用 get_config() 获取最新配置，而不是保存为实例变量
        # 这样配置更新后能立即生效
        logger.info("六维选股策略初始化完成")

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period, min_periods=1).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period, min_periods=1).min()

    def _calculate_main_trading(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df is None or len(df) < 30:
            return None
        try:
            df = df.copy()
            if 'Open' not in df.columns and 'open' in df.columns:
                df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            result_df = self._main_trading_indicator.calculate(df)
            latest = result_df.iloc[-1]
            latest_buy = bool(latest['buy_signal'] == 1) if pd.notna(latest['buy_signal']) else False
            score = 100 if latest_buy else 0
            return {
                'matched': latest_buy,
                'score': score,
                'buy_signal': latest_buy,
            }
        except Exception:
            return None

    def _calculate_banker_control(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df is None or len(df) < 40:
            return None
        try:
            df = df.copy()
            close = df['close']
            open_price = df['open']
            high = df['high']
            low = df['low']
            aaa = (3 * close + open_price + high + low) / 6
            ma12 = self._ema(aaa, 12)
            ma36 = self._ema(aaa, 36)
            ma36_prev = ma36.shift(1)
            control_degree = (ma12 - ma36_prev) / ma36_prev * 100 + 50
            latest_control = float(control_degree.iloc[-1]) if pd.notna(control_degree.iloc[-1]) else None
            matched = latest_control >= 50 if latest_control is not None else False
            if latest_control is not None and latest_control >= 50:
                # score = 200 - (2 * latest_control)
                score = 100 - abs(53 - latest_control) * 5
            else:
                score = 0
            return {
                'matched': matched,
                'score': score,
                'control_degree': latest_control,
            }
        except Exception:
            return None

    def _calculate_momentum2(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df is None or len(df) < 6:
            return None
        try:
            df = df.copy()
            if 'Open' not in df.columns and 'open' in df.columns:
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            result_df = self._momentum2_indicator.calculate(df)
            
            if len(result_df) < 2:
                return None
            
            latest = result_df.iloc[-1]
            prev = result_df.iloc[-2]
            
            today_is_red = bool(latest['strong_momentum1']) if pd.notna(latest['strong_momentum1']) else False
            today_height = float(latest['momentum_price']) if pd.notna(latest['momentum_price']) else 0
            
            if not today_is_red:
                return {
                    'matched': False,
                    'score': 0,
                    'red_pillar': False,
                }
            
            prev_is_red = bool(prev['strong_momentum1']) if pd.notna(prev['strong_momentum1']) else False
            prev_is_yellow = bool(prev['medium_momentum1']) if pd.notna(prev['medium_momentum1']) else False
            prev_is_green = bool(prev['weak_momentum1']) if pd.notna(prev['weak_momentum1']) else False
            prev_is_blue = bool(prev['recovery_momentum1']) if pd.notna(prev['recovery_momentum1']) else False
            prev_height = float(prev['momentum_price']) if pd.notna(prev['momentum_price']) else 0
            
            score = 80
            score_reason = "默认分"
            height_change_pct = None
            prev_color = None
            
            if prev_is_green:
                score = 90
                score_reason = "今天红，昨天绿"
                prev_color = "绿"
            elif prev_is_blue:
                score = 90
                score_reason = "今天红，昨天蓝"
                prev_color = "蓝"
            elif prev_is_red and prev_height > 0:
                height_change_pct = ((today_height - prev_height) / prev_height) * 100
                prev_color = "红"
                if height_change_pct > 100:
                    score = 100
                    score_reason = f"昨天红，今天红，高度+{height_change_pct:.1f}%"
                elif height_change_pct >= 60:
                    score = 90
                    score_reason = f"昨天红，今天红，高度+{height_change_pct:.1f}%"
                else:
                    score = 80
                    score_reason = f"昨天红，今天红，高度+{height_change_pct:.1f}%"
            elif prev_is_yellow:
                score = 80
                score_reason = "今天红，昨天黄"
                prev_color = "黄"
            
            return {
                'matched': True,
                'score': score,
                'red_pillar': True,
                'score_reason': score_reason,
                'today_height': today_height,
                'prev_height': prev_height,
                'height_change_pct': height_change_pct,
                'prev_color': prev_color,
            }
        except Exception:
            return None

    def _calculate_resonance_chase(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df is None or len(df) < 60:
            return None
        try:
            df = df.copy()
            close = df['close']
            open_price = df['open']
            high = df['high']
            low = df['low']
            AAA = (3 * close + high + low + open_price) / 6
            VAR1 = self._ema(AAA, 35)
            VAR2 = (self._hhv(VAR1, 10) + self._hhv(VAR1, 30) + self._hhv(VAR1, min(60, len(df)))) / 3
            VAR3 = (self._llv(VAR1, 10) + self._llv(VAR1, 30) + self._llv(VAR1, min(60, len(df)))) / 3
            bull_line = (self._hhv(VAR2, 5) + self._hhv(VAR2, 10) + self._hhv(VAR2, 20)) / 3
            bear_line = (self._llv(VAR3, 5) + self._llv(VAR3, 10) + self._llv(VAR3, 20)) / 3
            ema_aaa_2 = self._ema(self._ema(AAA, 2), 2)
            mid_bullish = ema_aaa_2 > bear_line
            exp6 = self._ema(close, 6)
            exp18 = self._ema(close, 18)
            OUT1 = 500 * (exp6 - exp18) / exp18 + 2
            OUT2 = self._ema(OUT1, 3)
            resonance = OUT1 > 2
            purple_resonance = mid_bullish & resonance
            latest_purple = bool(purple_resonance.iloc[-1]) if pd.notna(purple_resonance.iloc[-1]) else False
            latest_mid = bool(mid_bullish.iloc[-1]) if pd.notna(mid_bullish.iloc[-1]) else False
            latest_res = bool(resonance.iloc[-1]) if pd.notna(resonance.iloc[-1]) else False
            score = 100 if latest_purple else 0
            if latest_mid and not latest_purple:
                score = 50
            if latest_res and not latest_purple:
                score = max(score, 60)
            return {
                'matched': latest_purple,
                'score': score,
                'purple_resonance': latest_purple,
                'mid_bullish': latest_mid,
                'resonance': latest_res,
            }
        except Exception:
            return None

    def _get_market_index_code(self, stock_code: str) -> str:
        """根据股票代码获取对应的大盘指数代码"""
        if stock_code.startswith(('60', '688')):
            return 'sh000001'
        elif stock_code.startswith(('00', '30')):
            return 'sz399001'
        else:
            return 'sh000001'

    def _get_market_data(self, index_symbol: str) -> Optional[pd.DataFrame]:
        """获取大盘数据（带缓存）"""
        cached_data = MarketDataCache.load(index_symbol)
        if cached_data is not None:
            return cached_data

        if self._data_provider:
            try:
                market_data = self._data_provider.get_index_daily_data(index_symbol)
                if market_data is not None and not market_data.empty:
                    MarketDataCache.save(index_symbol, market_data)
                    return market_data
            except Exception as e:
                logger.warning(f"获取大盘数据失败 {index_symbol}: {e}")

        return None

    def _calculate_strong_detonation(self, df: pd.DataFrame, stock_code: str) -> Optional[Dict[str, Any]]:
        if df is None or len(df) < 60:
            return None
        try:
            df = df.copy()
            index_symbol = self._get_market_index_code(stock_code)
            market_data = self._get_market_data(index_symbol)

            if market_data is not None and not market_data.empty:
                market_df = market_data.copy()
                if "date" in market_df.columns:
                    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.date

                market_df_renamed = market_df[["date", "close"]].rename(
                    columns={"close": "close_index"}
                )
                merged_df = df.merge(market_df_renamed, on="date", how="left")

                if "close_index" in merged_df.columns:
                    merged_df["close_index"] = merged_df["close_index"].ffill().bfill()
                    if not merged_df["close_index"].isna().any():
                        # 从 merged_df 中获取所有数据，确保索引一致
                        close = merged_df["close"]
                        open_price = merged_df["open"]
                        high = merged_df["high"]
                        low = merged_df["low"]
                        index_close = merged_df["close_index"]
                        
                        aaa = (3 * close + open_price + high + low) / 6
                        var1 = self._ema(aaa, 35)
                        var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
                        var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
                        bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
                        ema120_stock = self._ema(close, 120)
                        
                        ema120_index = self._ema(index_close, 120)
                        a1 = index_close / ema120_index
                        market_midline = self._ema(ema120_stock * a1, 2)
                    else:
                        return {
                            'matched': False,
                            'score': 0,
                            'purple_box': False,
                            'consecutive_count': 0,
                        }
                else:
                    return {
                        'matched': False,
                        'score': 0,
                        'purple_box': False,
                        'consecutive_count': 0,
                    }
            else:
                return {
                    'matched': False,
                    'score': 0,
                    'purple_box': False,
                    'consecutive_count': 0,
                }

            purple_box = (aaa > bull_line) & (aaa > market_midline)
            latest_purple = bool(purple_box.iloc[-1]) if pd.notna(purple_box.iloc[-1]) else False
            
            consecutive_count = 0
            if latest_purple:
                for i in range(len(purple_box) - 1, -1, -1):
                    if bool(purple_box.iloc[i]) if pd.notna(purple_box.iloc[i]) else False:
                        consecutive_count += 1
                    else:
                        break
            
            if 1 <= consecutive_count < 5:
                score = 100
            elif 5 <= consecutive_count < 10:
                score = 95
            elif 10 <= consecutive_count < 20:
                score = 85
            elif 20 <= consecutive_count <= 50:
                score = 70
            elif consecutive_count > 20:
                score = 50
            else:
                score = 0
            
            return {
                'matched': latest_purple,
                'score': score,
                'purple_box': latest_purple,
                'consecutive_count': consecutive_count,
            }
        except Exception:
            return None

    def _calculate_sector_score(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        计算个股的板块分

        评分规则：
        - 对每个板块调用 calculate_sector_score() 获取分数
        - 同时调用 get_sector_tier() 获取每个板块的梯队信息
        - 保留包含关系处理逻辑（按分数降序排序、检测包含关系、只保留最具体的板块、取前3个）
        - 计算总分：前3个板块的分数相加
        - 总分上限：60分

        Args:
            stock_code: 股票代码

        Returns:
            包含板块分计算详情的字典，如果无法计算则返回 None
        """
        if not self._sector_manager:
            return None

        try:
            sectors = self._sector_manager.get_stock_sectors(stock_code)
            if not sectors:
                return None

            current_date = date.today()
            sector_scores = []
            sector_details = []

            for sector_name in sectors:
                sector_score = self._sector_manager.calculate_sector_score(sector_name, current_date)
                sector_scores.append(sector_score)

                tier = self._sector_manager.get_sector_tier(sector_name)
                momentum_score = self._sector_manager.calculate_sector_momentum_score(sector_name, current_date)
                sentiment_score = self._sector_manager.calculate_sector_sentiment_score(sector_name)
                is_hot = self._sector_manager.is_hot_sector(sector_name, threshold_pct=2.0)
                current_change = self._sector_manager.get_sector_change_pct(sector_name)

                sector_details.append({
                    'name': sector_name,
                    'score': sector_score,
                    'tier': tier,
                    'momentum_score': momentum_score,
                    'sentiment_score': sentiment_score,
                    'is_hot': is_hot,
                    'current_change': current_change,
                })

            if sector_scores:
                # 新逻辑：有包含关系取平均，独立板块直接相加
                # 1. 按分数降序排序
                sorted_details = sorted(
                    zip(sector_details, sector_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # 2. 分组处理：有包含关系的取平均，独立板块直接保留
                processed_items = []
                processed_indices = set()
                all_names = [detail['name'] for detail, score in sorted_details]
                
                for i in range(len(sorted_details)):
                    if i in processed_indices:
                        continue
                    
                    current_name = all_names[i]
                    current_detail, current_score = sorted_details[i]
                    
                    # 找出与当前板块有包含关系的所有板块
                    contained_group_indices = [i]
                    for j in range(len(sorted_details)):
                        if j == i or j in processed_indices:
                            continue
                        other_name = all_names[j]
                        if current_name in other_name or other_name in current_name:
                            contained_group_indices.append(j)
                    
                    if len(contained_group_indices) > 1:
                        # 有包含关系，取平均分
                        group_details = [sorted_details[j][0] for j in contained_group_indices]
                        group_scores = [sorted_details[j][1] for j in contained_group_indices]
                        avg_score = sum(group_scores) / len(group_scores)
                        
                        # 使用分数最高的那个板块的详情作为代表
                        best_idx = contained_group_indices[group_scores.index(max(group_scores))]
                        best_detail = sorted_details[best_idx][0].copy()
                        best_detail['score'] = avg_score
                        best_detail['is_contained_group'] = True
                        best_detail['group_size'] = len(contained_group_indices)
                        
                        processed_items.append((best_detail, avg_score))
                        for j in contained_group_indices:
                            processed_indices.add(j)
                    else:
                        # 独立板块，直接保留
                        processed_items.append((current_detail, current_score))
                        processed_indices.add(i)
                
                # 3. 按分数降序排序处理后的结果
                processed_items_sorted = sorted(processed_items, key=lambda x: x[1], reverse=True)
                
                # 4. 取前3个
                top_items = processed_items_sorted[:3]
                top_sector_details = [item[0] for item in top_items]
                top_sector_scores = [item[1] for item in top_items]
                
                # 5. 计算总分数（相加）
                total_score = sum(top_sector_scores)
                total_score = min(total_score, 60.0)
                
                # 6. 更新返回的详情
                sector_details = top_sector_details
                sector_scores = top_sector_scores
                sectors = [detail['name'] for detail in sector_details]
            else:
                total_score = 0.0

            return {
                'matched': total_score > 0,
                'score': total_score,
                'sectors': sectors,
                'sector_details': sector_details,
                'sector_scores': sector_scores,
            }
        except Exception as e:
            logger.warning(f"计算板块分时出错 {stock_code}: {e}")
            return None

    def select(self, stock_code: str, stock_name: Optional[str] = None, daily_data: Optional[pd.DataFrame] = None, precomputed_metrics: Optional[Dict[str, Any]] = None) -> StrategyMatch:
        match_details = {}
        conditions_met = []
        matched_strategies = []
        strategy_scores = []
        weighted_scores = []
        max_score = 500.0
        data_source = "database"

        try:
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                
                # 只有在没有传入外部数据时，才自己获取
                if daily_data is None:
                    data_source = "data_provider"
                    daily_data_result = self._data_provider.get_daily_data(stock_code, days=150)
                    if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                        daily_data, data_source = daily_data_result
                    else:
                        daily_data = daily_data_result

                match_details["realtime_quote"] = {}
                match_details["data_source"] = data_source
                match_details["sub_strategies"] = {}

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {"price": price}

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    # logger.debug(f"{stock_code}: 数据行数={len(daily_data)}, 日期范围={daily_data['date'].min() if 'date' in daily_data.columns else 'N/A'} ~ {daily_data['date'].max() if 'date' in daily_data.columns else 'N/A'}")
                    
                    main_trading = self._calculate_main_trading(daily_data)
                    if main_trading:
                        match_details["sub_strategies"]["main_trading_buy_signal"] = main_trading
                        # logger.debug(f"{stock_code}: 主力操盘 matched={main_trading['matched']}, score={main_trading['score']}")
                        if main_trading["matched"]:
                            weighted_score = main_trading["score"] * get_config().six_dimension_main_trading_weight
                            matched_strategies.append("主力操盘买入信号")
                            strategy_scores.append(main_trading["score"])
                            weighted_scores.append(weighted_score)
                            conditions_met.append(f"主力操盘买入信号(+{main_trading['score']}, 权重×{get_config().six_dimension_main_trading_weight})")

                    # 尝试复用预计算的控盘度
                    if precomputed_metrics and "control_degree" in precomputed_metrics and precomputed_metrics["control_degree"] is not None:
                        control_degree = precomputed_metrics["control_degree"]
                        matched = control_degree >= 50 if control_degree is not None else False
                        if control_degree is not None and control_degree >= 50:
                            score = 100 - abs(53 - control_degree) * 5
                        else:
                            score = 0
                        banker_control = {
                            'matched': matched,
                            'score': score,
                            'control_degree': control_degree,
                        }
                        match_details["sub_strategies"]["banker_control_selector"] = banker_control
                        if banker_control["matched"]:
                            weighted_score = banker_control["score"] * get_config().six_dimension_bank_control_weight
                            matched_strategies.append("庄家控盘选股")
                            strategy_scores.append(banker_control["score"])
                            weighted_scores.append(weighted_score)
                            conditions_met.append(f"庄家控盘选股(+{banker_control['score']:.1f}, 权重×{get_config().six_dimension_bank_control_weight})")
                    else:
                        # 没有预计算的控盘度，自己计算
                        banker_control = self._calculate_banker_control(daily_data)
                        if banker_control:
                            match_details["sub_strategies"]["banker_control_selector"] = banker_control
                            if banker_control["matched"]:
                                weighted_score = banker_control["score"] * get_config().six_dimension_bank_control_weight
                                matched_strategies.append("庄家控盘选股")
                                strategy_scores.append(banker_control["score"])
                                weighted_scores.append(weighted_score)
                                conditions_met.append(f"庄家控盘选股(+{banker_control['score']:.1f}, 权重×{get_config().six_dimension_bank_control_weight})")

                    momentum2 = self._calculate_momentum2(daily_data)
                    if momentum2:
                        match_details["sub_strategies"]["momentum_2_red_pillar_python"] = momentum2
                        # logger.debug(f"{stock_code}: 动能二号 matched={momentum2['matched']}, score={momentum2['score']}")
                        if momentum2["matched"]:
                            weighted_score = momentum2["score"] * get_config().six_dimension_momentum_v2_weight
                            matched_strategies.append("动能二号红色柱")
                            strategy_scores.append(momentum2["score"])
                            weighted_scores.append(weighted_score)
                            conditions_met.append(f"动能二号红色柱(+{momentum2['score']}, 权重×{get_config().six_dimension_momentum_v2_weight})")

                    resonance_chase = self._calculate_resonance_chase(daily_data)
                    if resonance_chase:
                        match_details["sub_strategies"]["resonance_chase"] = resonance_chase
                        # logger.debug(f"{stock_code}: 共振追涨 matched={resonance_chase['matched']}, score={resonance_chase['score']}")
                        if resonance_chase["matched"]:
                            weighted_score = resonance_chase["score"] * get_config().six_dimension_resonance_weight
                            matched_strategies.append("共振追涨")
                            strategy_scores.append(resonance_chase["score"])
                            weighted_scores.append(weighted_score)
                            conditions_met.append(f"共振追涨(+{resonance_chase['score']}, 权重×{get_config().six_dimension_resonance_weight})")

                    # 尝试复用预计算的 purple_days，但强势起爆还需要更多计算，所以还是完整计算
                    strong_detonation = self._calculate_strong_detonation(daily_data, stock_code)
                    if strong_detonation:
                        match_details["sub_strategies"]["strong_detonation_selector"] = strong_detonation
                        # logger.debug(f"{stock_code}: 强势起爆 matched={strong_detonation['matched']}, score={strong_detonation['score']}, consecutive_count={strong_detonation.get('consecutive_count')}")
                        if strong_detonation["matched"]:
                            weighted_score = strong_detonation["score"] * get_config().six_dimension_strong_blast_weight
                            consecutive_count = strong_detonation.get("consecutive_count", 0)
                            matched_strategies.append("强势起爆选股")
                            strategy_scores.append(strong_detonation["score"])
                            weighted_scores.append(weighted_score)
                            conditions_met.append(f"强势起爆选股(+{strong_detonation['score']}, 权重×{get_config().six_dimension_strong_blast_weight}, 连续{consecutive_count}次)")

                    sector_score = self._calculate_sector_score(stock_code)
                    if sector_score:
                        match_details["sector_score"] = sector_score
                        # logger.debug(f"{stock_code}: 板块分 score={sector_score['score']}")
                        if sector_score["score"] > 0:
                            weighted_score = sector_score["score"] * get_config().six_dimension_sector_weight
                            matched_strategies.append("板块分")
                            strategy_scores.append(sector_score["score"])
                            weighted_scores.append(weighted_score)
                            
                            sector_details = sector_score.get('sector_details', [])
                            sectors = sector_score.get('sectors', [])
                            sector_scores_list = sector_score.get('sector_scores', [])
                            
                            if sector_details:
                                sector_info = []
                                for i, detail in enumerate(sector_details):
                                    name = detail.get('name', '')
                                    score = detail.get('score', 0)
                                    momentum = detail.get('momentum_score', 0)
                                    sentiment = detail.get('sentiment_score', 0)
                                    current_change = detail.get('current_change', 0)
                                    is_hot = detail.get('is_hot', False)
                                    
                                    hot_mark = '🔥' if is_hot else ''
                                    change_str = f'+{current_change:.1f}%' if current_change and current_change >= 0 else f'{current_change:.1f}%' if current_change else 'N/A'
                                    
                                    sector_info.append(f"{name}{hot_mark}({score:.1f},动量{momentum:.1f},景气{sentiment:.1f},{change_str})")
                                
                                sector_info_str = ' | '.join(sector_info)
                                # 详细说明板块分构成
                                sector_count = len(sector_details)
                                avg_score_desc = f"{sector_score['score']:.1f}"
                                weighted_score_desc = f"{weighted_score:.1f}"
                                conditions_met.append(f"板块分(+{weighted_score_desc},原分{avg_score_desc}×权重{get_config().six_dimension_sector_weight},取{sector_count}个板块平均:[{sector_info_str}])")
                            else:
                                conditions_met.append(f"板块分(+{sector_score['score']:.1f},权重×{get_config().six_dimension_sector_weight})")

        except Exception as e:
            logger.warning(f"执行六维选股策略时出错 {stock_code}: {e}")

        if weighted_scores:
            raw_score = sum(weighted_scores)
        else:
            raw_score = 0.0

        config = get_config()
        raw_score = min(raw_score, max_score)
        matched = len(matched_strategies) >= config.six_dimension_min_matched_dimensions
        
        # 调试：记录最终结果
        # logger.debug(f"{stock_code}: 匹配维度数={len(matched_strategies)}, 需≥{config.six_dimension_min_matched_dimensions}, 总分={raw_score:.1f}, matched={matched}")

        if matched_strategies:
            reason = f"六维评分 {raw_score:.1f}，匹配{len(matched_strategies)}个维度(需≥{config.six_dimension_min_matched_dimensions})：{'; '.join(conditions_met)}"
        else:
            reason = f"六维评分 {raw_score:.1f}：未匹配任何策略"

        match_details["matched_strategies"] = matched_strategies
        match_details["conditions_met"] = conditions_met
        match_details["strategy_scores"] = strategy_scores
        match_details["weighted_scores"] = weighted_scores
        match_details["matched_dimensions_count"] = len(matched_strategies)
        match_details["min_matched_dimensions"] = config.six_dimension_min_matched_dimensions

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
        )
