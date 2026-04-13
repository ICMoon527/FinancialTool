# -*- coding: utf-8 -*-
"""
Batch Data Updater - 批量数据更新器

核心功能：
1. 利用Tushare按交易日批量获取多只股票数据（trade_date参数）
2. 智能分组策略，避免单次请求过多数据
3. 请求频率控制和速率限制
4. 错误重试和失败处理机制
5. 增量更新：昨天已更新的股票今天仅获取单天数据
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple, Dict, Set
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from data_provider import DataFetcherManager, DataFetchError, RateLimitError
from src.storage import DatabaseManager, StockDaily
from sqlalchemy import select, and_, or_

from .data_update_tracker import DataUpdateTracker, get_update_tracker

logger = logging.getLogger(__name__)


class BatchDataUpdater:
    """
    批量数据更新器
    
    优化策略：
    1. 按交易日批量获取（使用Tushare的trade_date参数）
    2. 智能分组：根据数据量动态调整每组股票数量
    3. 增量更新：已更新股票仅获取新数据
    4. 速率控制：遵守Tushare配额限制
    5. 失败重试：指数退避重试机制
    """
    
    # Tushare免费用户配额
    MAX_CALLS_PER_MINUTE = 50
    MAX_STOCKS_PER_BATCH = 100  # 保守估计，根据实际情况调整
    MIN_STOCKS_PER_BATCH = 10
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        data_fetcher_manager: Optional[DataFetcherManager] = None,
        update_tracker: Optional[DataUpdateTracker] = None
    ):
        """
        初始化批量数据更新器
        
        Args:
            db_manager: 数据库管理器
            data_fetcher_manager: 数据获取管理器
            update_tracker: 数据更新追踪器
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self.data_fetcher_manager = data_fetcher_manager or DataFetcherManager()
        self.update_tracker = update_tracker or get_update_tracker()
        
        # 速率限制追踪
        self._call_count = 0
        self._minute_start = None
    
    def _check_rate_limit(self):
        """检查并执行速率限制"""
        current_time = time.time()
        
        if self._minute_start is None:
            self._minute_start = current_time
            self._call_count = 0
        elif current_time - self._minute_start >= 60:
            # 新的一分钟，重置计数器
            self._minute_start = current_time
            self._call_count = 0
        
        if self._call_count >= self.MAX_CALLS_PER_MINUTE:
            # 超过限制，等待到下一分钟
            elapsed = current_time - self._minute_start
            sleep_time = max(0, 60 - elapsed) + 1
            logger.warning(f"Rate limit reached, waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            self._minute_start = time.time()
            self._call_count = 0
        
        self._call_count += 1
        logger.debug(f"API calls this minute: {self._call_count}/{self.MAX_CALLS_PER_MINUTE}")
    
    def calculate_stocks_per_batch(self, num_stocks: int, num_days: int) -> int:
        """
        动态计算每组的股票数量
        
        Args:
            num_stocks: 总股票数量
            num_days: 需要获取的天数
            
        Returns:
            每组股票数量
        """
        # 估算数据量：每只股票每天约1条记录
        # Tushare单次返回限制约5000-10000条记录
        estimated_records = num_stocks * num_days
        
        if estimated_records <= 1000:
            return min(num_stocks, self.MAX_STOCKS_PER_BATCH)
        elif estimated_records <= 5000:
            return min(num_stocks, max(self.MIN_STOCKS_PER_BATCH, self.MAX_STOCKS_PER_BATCH // 2))
        else:
            return max(self.MIN_STOCKS_PER_BATCH, 20)
    
    def group_stocks(self, stock_codes: List[str], stocks_per_batch: int) -> List[List[str]]:
        """
        将股票分组
        
        Args:
            stock_codes: 股票代码列表
            stocks_per_batch: 每组股票数量
            
        Returns:
            分组后的股票列表
        """
        groups = []
        for i in range(0, len(stock_codes), stocks_per_batch):
            groups.append(stock_codes[i:i + stocks_per_batch])
        return groups
    
    def fetch_batch_by_trade_date(
        self,
        trade_date: date,
        stock_codes: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        按交易日批量获取多只股票数据
        
        使用Tushare的trade_date参数，一次获取多只股票的单日数据
        
        Args:
            trade_date: 交易日
            stock_codes: 股票代码列表
            
        Returns:
            Tuple of (dataframe, failed_stocks)
        """
        from data_provider.tushare_fetcher import TushareFetcher
        
        failed_stocks = []
        
        # 获取Tushare fetcher
        tushare_fetcher = None
        for fetcher in self.data_fetcher_manager._fetchers:
            if fetcher.name == "TushareFetcher" and fetcher.is_available():
                tushare_fetcher = fetcher
                break
        
        if tushare_fetcher is None:
            logger.warning("TushareFetcher not available, falling back to individual fetch")
            return self._fetch_individual_fallback(trade_date, stock_codes)
        
        try:
            self._check_rate_limit()
            
            # 转换日期格式
            ts_date = trade_date.strftime('%Y%m%d')
            
            # 转换股票代码为Tushare格式
            ts_codes = [tushare_fetcher._convert_stock_code(code) for code in stock_codes]
            ts_codes_str = ','.join(ts_codes)
            
            logger.info(f"Fetching batch data for {len(stock_codes)} stocks on {trade_date}")
            
            # 使用daily接口，同时指定ts_code和trade_date
            df = tushare_fetcher._api.daily(
                ts_code=ts_codes_str,
                trade_date=ts_date
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned for batch on {trade_date}")
                return pd.DataFrame(), stock_codes
            
            logger.info(f"Successfully fetched {len(df)} records for {trade_date}")
            return df, []
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['quota', '配额', 'limit', '权限']):
                logger.error(f"Rate limit error during batch fetch: {e}")
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            else:
                logger.warning(f"Batch fetch failed, falling back to individual: {e}")
                return self._fetch_individual_fallback(trade_date, stock_codes)
    
    def _fetch_individual_fallback(
        self,
        trade_date: date,
        stock_codes: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        批量获取失败时的降级方案：逐个获取
        
        Args:
            trade_date: 交易日
            stock_codes: 股票代码列表
            
        Returns:
            Tuple of (dataframe, failed_stocks)
        """
        all_data = []
        failed_stocks = []
        
        start_date_str = trade_date.strftime('%Y-%m-%d')
        # 结束日期加一天，确保包含当天的数据
        end_date_for_query = trade_date + timedelta(days=1)
        end_date_str = end_date_for_query.strftime('%Y-%m-%d')
        
        for code in stock_codes:
            try:
                df, source = self.data_fetcher_manager.get_daily_data(
                    code,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                if df is not None and not df.empty:
                    all_data.append(df)
                else:
                    failed_stocks.append(code)
            except Exception as e:
                logger.warning(f"Failed to fetch {code} on {trade_date}: {e}")
                failed_stocks.append(code)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True), failed_stocks
        else:
            return pd.DataFrame(), failed_stocks
    
    def save_batch_data(self, df: pd.DataFrame, stock_code: Optional[str] = None):
        """
        保存批量数据到数据库（优化版：批量查询 + 批量插入）
        
        Args:
            df: 数据DataFrame（可能包含多只股票）
            stock_code: 单一股票代码（如果是单只股票数据）
        """
        if df is None or df.empty:
            return
        
        saved_count = 0
        records_to_process = []
        
        try:
            with self.db_manager.get_session() as session:
                # 第一步：预处理所有数据行
                for _, row in df.iterrows():
                    # 确定股票代码
                    code = stock_code
                    if 'ts_code' in row:
                        code = row['ts_code'].split('.')[0]
                    elif 'code' in row:
                        code = row['code']
                    
                    if not code:
                        continue
                    
                    # 解析日期
                    record_date = None
                    if 'trade_date' in row:
                        td = str(row['trade_date'])
                        record_date = datetime.strptime(td, '%Y%m%d').date()
                    elif 'date' in row:
                        d = row['date']
                        if isinstance(d, str):
                            record_date = datetime.strptime(d, '%Y-%m-%d').date()
                        elif hasattr(d, 'date'):
                            record_date = d.date()
                        else:
                            record_date = d
                    
                    if not record_date:
                        continue
                    
                    records_to_process.append({
                        'code': code,
                        'date': record_date,
                        'row': row
                    })
                
                if not records_to_process:
                    return
                
                # 第二步：批量查询已存在的记录
                code_date_pairs = [(rec['code'], rec['date']) for rec in records_to_process]
                existing_records = {}
                
                if code_date_pairs:
                    # 构建批量查询条件
                    or_conditions = []
                    for code, dt in code_date_pairs:
                        or_conditions.append(and_(StockDaily.code == code, StockDaily.date == dt))
                    
                    if or_conditions:
                        stmt = select(StockDaily).where(or_(*or_conditions))
                        results = session.execute(stmt).scalars().all()
                        
                        # 构建查找字典
                        for rec in results:
                            key = (rec.code, rec.date)
                            existing_records[key] = rec
                
                # 第三步：分别处理更新和插入
                new_records = []
                
                for rec_info in records_to_process:
                    code = rec_info['code']
                    record_date = rec_info['date']
                    row = rec_info['row']
                    key = (code, record_date)
                    
                    if key in existing_records:
                        # 更新现有记录
                        existing = existing_records[key]
                        if 'open' in row:
                            existing.open = row.get('open')
                        if 'high' in row:
                            existing.high = row.get('high')
                        if 'low' in row:
                            existing.low = row.get('low')
                        if 'close' in row:
                            existing.close = row.get('close')
                        if 'vol' in row:
                            existing.volume = row.get('vol') * 100  # 手->股
                        elif 'volume' in row:
                            existing.volume = row.get('volume')
                        if 'amount' in row:
                            existing.amount = row.get('amount') * 1000  # 千元->元
                        if 'pct_chg' in row:
                            existing.pct_chg = row.get('pct_chg')
                    else:
                        # 创建新记录（批量插入）
                        new_record = StockDaily(
                            code=code,
                            date=record_date,
                            open=row.get('open'),
                            high=row.get('high'),
                            low=row.get('low'),
                            close=row.get('close'),
                            volume=row.get('vol', row.get('volume', 0)) * 100 if 'vol' in row else row.get('volume', 0),
                            amount=row.get('amount', 0) * 1000,
                            pct_chg=row.get('pct_chg')
                        )
                        new_records.append(new_record)
                    
                    saved_count += 1
                
                # 批量插入新记录
                if new_records:
                    session.add_all(new_records)
                
                session.commit()
                logger.debug(f"Saved {saved_count} records from batch (optimized)")
        except Exception as e:
            logger.error(f"Error saving batch data: {e}", exc_info=True)
            if 'session' in locals():
                session.rollback()
    
    def update_stocks_for_date_range(
        self,
        stock_codes: List[str],
        start_date: date,
        end_date: date,
        use_batch: bool = True
    ) -> Dict[str, any]:
        """
        更新指定日期范围的股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_batch: 是否使用批量获取
            
        Returns:
            更新统计信息
        """
        stats = {
            'total_stocks': len(stock_codes),
            'stocks_updated': 0,
            'stocks_failed': 0,
            'failed_stocks': [],
            'dates_processed': 0,
            'api_calls': 0
        }
        
        # 获取需要更新的交易日
        trading_days = self.update_tracker.get_trading_days_between(start_date, end_date)
        
        if not trading_days:
            # 没有已知交易日，按日历日处理
            current = start_date
            trading_days = []
            while current <= end_date:
                trading_days.append(current)
                current += timedelta(days=1)
        
        logger.info(f"Updating data for {len(stock_codes)} stocks across {len(trading_days)} days")
        
        # 按交易日处理，带进度条
        for trade_date in tqdm(trading_days, desc="Processing dates", unit="day"):
            logger.info(f"Processing date: {trade_date}")
            
            # 筛选出这一天需要更新的股票
            needs_update = []
            for code in stock_codes:
                record = self.update_tracker.get_update_record(code)
                if record is None or record.last_updated_date is None or record.last_updated_date < trade_date:
                    needs_update.append(code)
            
            if not needs_update:
                logger.debug(f"All stocks already up to date for {trade_date}")
                continue
            
            logger.info(f"Stocks needing update for {trade_date}: {len(needs_update)}")
            
            # 动态计算每组大小
            stocks_per_batch = self.calculate_stocks_per_batch(len(needs_update), 1)
            groups = self.group_stocks(needs_update, stocks_per_batch)
            
            logger.info(f"Split into {len(groups)} batches ({stocks_per_batch} stocks/batch)")
            
            # 按组处理，带进度条
            for group in tqdm(groups, desc=f"  Processing batches for {trade_date}", unit="batch", leave=False):
                try:
                    if use_batch:
                        df, failed = self.fetch_batch_by_trade_date(trade_date, group)
                    else:
                        df, failed = self._fetch_individual_fallback(trade_date, group)
                    
                    stats['api_calls'] += 1
                    
                    if not df.empty:
                        self.save_batch_data(df)
                        stats['stocks_updated'] += len(group) - len(failed)
                        
                        # 更新成功股票的记录
                        for code in group:
                            if code not in failed:
                                self.update_tracker.update_record(code, data_end_date=trade_date)
                    
                    if failed:
                        stats['stocks_failed'] += len(failed)
                        stats['failed_stocks'].extend(failed)
                        logger.warning(f"Failed to fetch {len(failed)} stocks for {trade_date}")
                
                except RateLimitError:
                    logger.error("Rate limit hit, stopping update")
                    stats['dates_processed'] += 1
                    return stats
                except Exception as e:
                    logger.error(f"Error processing batch for {trade_date}: {e}")
                    stats['stocks_failed'] += len(group)
                    stats['failed_stocks'].extend(group)
            
            stats['dates_processed'] += 1
        
        # 去重失败股票
        stats['failed_stocks'] = list(set(stats['failed_stocks']))
        
        logger.info(f"Update complete: {stats['stocks_updated']} updated, {stats['stocks_failed']} failed")
        return stats


def get_batch_updater() -> BatchDataUpdater:
    """获取全局批量数据更新器实例"""
    if not hasattr(get_batch_updater, '_instance'):
        get_batch_updater._instance = BatchDataUpdater()
    return get_batch_updater._instance
