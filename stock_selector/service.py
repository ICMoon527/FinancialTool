# -*- coding: utf-8 -*-
"""
Stock Selector Service - High-level service for stock screening.
"""

import logging
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import pandas as pd
import numpy as np

from stock_selector.base import StockCandidate, StrategyMetadata
from stock_selector.config import StockSelectorConfig, get_config
from stock_selector.manager import StrategyManager
from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks

logger = logging.getLogger(__name__)


class StockSelectorService:
    """
    High-level service for stock screening.

    Responsibilities:
    - Orchestrate stock selection workflow
    - Manage strategy manager
    - Provide high-level screening API
    """

    def __init__(
        self,
        nl_strategy_dir: Optional[Path] = None,
        python_strategy_dir: Optional[Path] = None,
        data_provider: Optional[any] = None,
        config: Optional[StockSelectorConfig] = None,
    ):
        self.config = config or get_config()
        self._db_manager = None

        # Initialize database manager
        try:
            from src.storage import get_db
            self._db_manager = get_db()
        except Exception as e:
            logger.warning(f"Failed to initialize database manager: {e}")

        if nl_strategy_dir is None and self.config.nl_strategy_dir:
            nl_strategy_dir = Path(self.config.nl_strategy_dir)

        if python_strategy_dir is None and self.config.python_strategy_dir:
            python_strategy_dir = Path(self.config.python_strategy_dir)

        self.strategy_manager = StrategyManager(
            nl_strategy_dir=nl_strategy_dir,
            python_strategy_dir=python_strategy_dir,
            data_provider=data_provider,
            config=self.config,
        )

    def set_data_provider(self, data_provider: any) -> None:
        self.strategy_manager.set_data_provider(data_provider)

    def screen_stocks(
        self,
        stock_codes: Optional[List[str]],
        strategy_ids: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> List[StockCandidate]:
        if top_n is None:
            top_n = self.config.default_top_n

        # Use complete stock pool if none provided
        if stock_codes is None:
            stock_code_name_pairs = get_all_stock_code_name_pairs()
            # 过滤ST股票
            stock_code_name_pairs = filter_st_stocks(stock_code_name_pairs)
            # 过滤特定板块的股票代码（科创板、创业板、北交所等）
            stock_codes = [code for code, name in stock_code_name_pairs]
            stock_codes = filter_special_stock_codes(stock_codes)
            logger.info("No stock codes provided, using complete stock pool: %d stocks", len(stock_codes))
        else:
            # 如果指定了股票代码，也需要过滤ST股票
            try:
                all_pairs = get_all_stock_code_name_pairs()
                code_to_name = {code: name for code, name in all_pairs}
                # 过滤ST股票
                filtered_codes = []
                for code in stock_codes:
                    name = code_to_name.get(code, "")
                    if not any(keyword in name.upper() for keyword in ['ST', '*ST', 'SST', 'S*ST']):
                        filtered_codes.append(code)
                stock_codes = filtered_codes
            except Exception:
                pass

        candidates: List[StockCandidate] = []

        # 根据配置决定使用多线程还是单线程模式
        enable_multithreading = self.config.enable_multithreading
        results = []

        def process_stock(code):
            try:
                return self._screen_single_stock(code, strategy_ids)
            except Exception as e:
                logger.error("Failed to screen stock %s: %s", code, e)
                return None

        if enable_multithreading:
            # 多线程模式：使用 ThreadPoolExecutor 并发处理
            # 线程数从配置读取，并与股票数量取最小值，避免创建过多线程
            max_workers = min(self.config.multithreading_workers, len(stock_codes))
            logger.info("使用多线程模式筛选股票，工作线程数: %d", max_workers)

            # 使用线程池进行并发处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有股票处理任务
                futures = {executor.submit(process_stock, code): code for code in stock_codes}
                
                # 带进度条处理完成的任务
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Screening stocks",
                    unit="stock"
                ):
                    results.append(future.result())
        else:
            # 单线程模式：顺序处理，不使用线程池
            logger.info("使用单线程模式筛选股票")
            # 带进度条顺序处理每只股票
            for code in tqdm(
                stock_codes,
                desc="Screening stocks",
                unit="stock"
            ):
                result = process_stock(code)
                results.append(result)

        # Collect valid candidates
        for result in results:
            if result:
                candidates.append(result)

        ranked_candidates = self.strategy_manager.rank_candidates(candidates, top_n=top_n)
        logger.info("Screened %d stocks, returned top %d", len(stock_codes), len(ranked_candidates))
        return ranked_candidates

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()
    
    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """计算周期内最高值"""
        return data.rolling(window=period).max()
    
    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """计算周期内最低值"""
        return data.rolling(window=period).min()
    
    def _calculate_control_degree(self, df: pd.DataFrame) -> Optional[float]:
        """计算控盘度，复用 banker_control_selector_strategy.py 中的逻辑"""
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
            return float(control_degree.iloc[-1]) if pd.notna(control_degree.iloc[-1]) else None
        except Exception as e:
            logger.debug(f"Failed to calculate control degree: {e}")
            return None
    
    def _get_market_index_code(self, stock_code: str) -> str:
        """根据股票代码获取对应的大盘指数代码"""
        if stock_code.startswith(('60', '688')):
            return 'sh000001'
        elif stock_code.startswith(('00', '30')):
            return 'sz399001'
        else:
            return 'sh000001'
    
    def _calculate_purple_days(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame]) -> Optional[int]:
        """计算连紫数，复用 strong_detonation_python.py 中的逻辑"""
        if df is None or len(df) < 60:
            return None
        
        try:
            df = df.copy()
            close = df['close']
            open_price = df['open']
            high = df['high']
            low = df['low']
            
            aaa = (3 * close + open_price + high + low) / 6
            var1 = self._ema(aaa, 35)
            var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
            var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
            bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
            
            ema120_stock = self._ema(close, 120)
            
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
                        index_close = merged_df["close_index"]
                        ema120_index = self._ema(index_close, 120)
                        a1 = index_close / ema120_index
                        market_midline = self._ema(ema120_stock * a1, 2)
                    else:
                        a1 = close / ema120_stock
                        market_midline = self._ema(ema120_stock * a1, 2)
                else:
                    a1 = close / ema120_stock
                    market_midline = self._ema(ema120_stock * a1, 2)
            else:
                a1 = close / ema120_stock
                market_midline = self._ema(ema120_stock * a1, 2)
            
            purple_box = (aaa > bull_line) & (aaa > market_midline)
            latest_purple = bool(purple_box.iloc[-1]) if pd.notna(purple_box.iloc[-1]) else False
            
            consecutive_count = 0
            if latest_purple:
                for i in range(len(purple_box) - 1, -1, -1):
                    if bool(purple_box.iloc[i]):
                        consecutive_count += 1
                    else:
                        break
            
            return consecutive_count
        except Exception as e:
            logger.debug(f"Failed to calculate purple days: {e}")
            return None

    def _screen_single_stock(
        self,
        stock_code: str,
        strategy_ids: Optional[List[str]] = None,
    ) -> Optional[StockCandidate]:
        current_price = 0.0
        stock_name = stock_code
        change_pct = None
        control_degree = None
        purple_days = None

        # 首先从股票池获取股票名称
        if self._db_manager:
            try:
                from stock_selector.stock_pool import StockPoolItem
                with self._db_manager.get_session() as session:
                    from sqlalchemy import select
                    result = session.execute(
                        select(StockPoolItem.name)
                        .where(StockPoolItem.code == stock_code)
                    ).scalar_one_or_none()
                    if result:
                        stock_name = result
            except Exception as e:
                logger.debug(f"Failed to get stock name from pool for {stock_code}: {e}")

        # 优先从数据库读取日线数据
        daily_data = None
        market_data = None
        if self._db_manager:
            try:
                from src.storage import StockDaily
                from datetime import date, timedelta
                from sqlalchemy import select, desc
                
                # 计算日期范围：获取最近150天的数据
                end_date = date.today()
                start_date = end_date - timedelta(days=200)  # 多取一些确保有足够数据
                
                with self._db_manager.get_session() as session:
                    # 从数据库读取数据
                    records = session.execute(
                        select(StockDaily)
                        .where(
                            StockDaily.code == stock_code,
                            StockDaily.date >= start_date,
                            StockDaily.date <= end_date
                        )
                        .order_by(StockDaily.date)
                    ).scalars().all()
                    
                    if records and len(records) >= 30:  # 确保有足够的数据
                        # 转换为 DataFrame
                        daily_data = pd.DataFrame([
                            {
                                'date': r.date,
                                'open': r.open,
                                'high': r.high,
                                'low': r.low,
                                'close': r.close,
                                'volume': r.volume,
                                'amount': r.amount,
                                'pct_chg': r.pct_chg
                            }
                            for r in records
                        ])
                        logger.debug(f"从数据库读取 {stock_code} 数据成功，共 {len(daily_data)} 条")
                        
                        # 从数据库数据获取涨跌幅
                        if not daily_data.empty and 'pct_chg' in daily_data.columns:
                            change_pct = float(daily_data['pct_chg'].iloc[-1]) if pd.notna(daily_data['pct_chg'].iloc[-1]) else None
                            logger.debug(f"从数据库获取 {stock_code} 涨跌幅: {change_pct}")
                        
                        # 计算控盘度
                        control_degree = self._calculate_control_degree(daily_data)
                        # 注意：从数据库读取数据时不计算连紫数，让后面统一的逻辑处理
                        # 因为 strategy_manager._data_provider 此时可能是 None
            except Exception as e:
                logger.debug(f"Failed to get data from database for {stock_code}: {e}")

        # 如果数据库中没有足够的数据，才从数据提供者获取
        if daily_data is None and self.strategy_manager._data_provider:
            try:
                # 获取日线数据（至少150天，确保计算指标有足够数据）
                daily_data_result = self.strategy_manager._data_provider.get_daily_data(stock_code, days=150)
                if isinstance(daily_data_result, tuple) and len(daily_data_result) == 2:
                    daily_data, _ = daily_data_result
                else:
                    daily_data = daily_data_result
                
                # 尝试从日线数据获取涨跌幅
                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    if 'pct_chg' in daily_data.columns:
                        change_pct = float(daily_data['pct_chg'].iloc[-1]) if pd.notna(daily_data['pct_chg'].iloc[-1]) else None
            except Exception as e:
                logger.debug(f"Failed to get daily data from provider for {stock_code}: {e}")

        # 如果有日线数据但还没有计算控盘度和连紫数，才计算
        if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
            if control_degree is None:
                control_degree = self._calculate_control_degree(daily_data)
            
            # 如果还没有获取大盘数据或计算连紫数，才计算
            if market_data is None or purple_days is None:
                # 获取大盘数据用于计算连紫数
                if self.strategy_manager._data_provider and market_data is None:
                    try:
                        market_index_code = self._get_market_index_code(stock_code)
                        market_data = self.strategy_manager._data_provider.get_index_daily_data(market_index_code)
                    except Exception as e:
                        logger.debug(f"Failed to get market data: {e}")
                
                # 计算连紫数
                if purple_days is None:
                    purple_days = self._calculate_purple_days(daily_data, market_data)

        # 尝试从实时行情获取涨跌幅（作为备选）
        if change_pct is None and self.strategy_manager._data_provider:
            try:
                quote = self.strategy_manager._data_provider.get_realtime_quote(stock_code)
                if quote:
                    change_pct = getattr(quote, "pct_chg", None)
                    if change_pct is not None:
                        change_pct = float(change_pct)
            except Exception as e:
                logger.debug(f"Failed to get realtime quote pct_chg for {stock_code}: {e}")

        # Try to get data from database cache first
        if self._db_manager:
            from datetime import date
            today = date.today()
            if self._db_manager.has_today_data(stock_code, today):
                logger.debug(f"Using cached data for {stock_code} from database in service")
                context = self._db_manager.get_analysis_context(stock_code, today)
                if context:
                    current_price = context.get('today', {}).get('close', 0.0)
                    stock_name = context.get('stock_name', stock_name)

        # 准备预计算的指标
        precomputed_metrics = {
            'control_degree': control_degree,
            'purple_days': purple_days,
        }
        
        # Now execute strategies with potential cached data
        matches = self.strategy_manager.execute_strategies(
            stock_code=stock_code,
            stock_name=stock_name,
            strategy_ids=strategy_ids,
            daily_data=daily_data,
            precomputed_metrics=precomputed_metrics,
        )

        if not matches:
            return None

        # Fallback to realtime data if still no price
        if current_price == 0.0:
            try:
                if self.strategy_manager._data_provider:
                    quote = self.strategy_manager._data_provider.get_realtime_quote(stock_code)
                    if quote:
                        current_price = getattr(quote, "price", 0.0)
                        if getattr(quote, "name", None):
                            stock_name = quote.name
            except Exception as e:
                logger.debug("Failed to get realtime quote for %s: %s", stock_code, e)

        candidate = StockCandidate(
            code=stock_code,
            name=stock_name,
            current_price=current_price,
        )

        # 将计算得到的指标存储到 extra_data 字典中
        candidate.extra_data["change_pct"] = change_pct
        candidate.extra_data["control_degree"] = control_degree
        
        # 优先从六维策略的强势起爆匹配结果中获取连紫数
        final_purple_days = purple_days
        for match in matches:
            if match.strategy_id == "six_dimension_selector" and match.match_details:
                sub_strategies = match.match_details.get("sub_strategies", {})
                strong_detonation = sub_strategies.get("strong_detonation_selector")
                if strong_detonation:
                    consecutive_count = strong_detonation.get("consecutive_count")
                    if consecutive_count is not None:
                        final_purple_days = consecutive_count
                        break
        
        candidate.extra_data["purple_days"] = final_purple_days

        # 从六维策略匹配信息中提取动量二号信息
        for match in matches:
            if match.strategy_id == "six_dimension_selector" and match.match_details:
                sub_strategies = match.match_details.get("sub_strategies", {})
                momentum2 = sub_strategies.get("momentum_2_red_pillar_python")
                if momentum2:
                    candidate.extra_data["momentum2_score_reason"] = momentum2.get("score_reason")
                    candidate.extra_data["momentum2_today_height"] = momentum2.get("today_height")
                    candidate.extra_data["momentum2_prev_height"] = momentum2.get("prev_height")
                    candidate.extra_data["momentum2_red_pillar"] = momentum2.get("red_pillar")
                    candidate.extra_data["momentum2_height_change_pct"] = momentum2.get("height_change_pct")
                    candidate.extra_data["momentum2_prev_color"] = momentum2.get("prev_color")
            candidate.add_strategy_match(match)

        has_matched_strategy = any(match.matched for match in matches)
        if has_matched_strategy:
            return candidate
        else:
            return None

    def get_available_strategies(self) -> List[StrategyMetadata]:
        strategies = self.strategy_manager.get_all_strategies()
        return [s.metadata for s in strategies]

    def activate_strategies(self, strategy_ids: List[str]) -> None:
        self.strategy_manager.deactivate_all_strategies()
        for sid in strategy_ids:
            self.strategy_manager.activate_strategy(sid)

    def get_active_strategy_ids(self) -> List[str]:
        return [s.id for s in self.strategy_manager.get_active_strategies()]

    def deactivate_strategies(self, strategy_ids: List[str]) -> None:
        for sid in strategy_ids:
            self.strategy_manager.deactivate_strategy(sid)
