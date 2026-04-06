# -*- coding: utf-8 -*-
"""
Tushare Data Downloader - 专门为 stock_selector 设计的 Tushare 数据下载器

核心功能：
1. 使用 Tushare 接口获取股票数据
2. 每次获取 10 只股票的 365 天数据
3. 显示下载进度条
4. 自动处理速率限制
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

import pandas as pd

from data_provider.tushare_fetcher import TushareFetcher
from data_provider.efinance_fetcher import EfinanceFetcher
from src.storage import DatabaseManager, StockDaily
from sqlalchemy import select, and_

from .data_update_tracker import DataUpdateTracker, get_update_tracker
from .stock_pool import get_all_stock_codes, filter_special_stock_codes
from .config import get_config

logger = logging.getLogger(__name__)


class TushareDataDownloader:
    """
    专门为 stock_selector 设计的 Tushare 数据下载器
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        update_tracker: Optional[DataUpdateTracker] = None,
        rate_limit_per_minute: int = 50
    ):
        """
        初始化 Tushare 数据下载器
        
        Args:
            db_manager: 数据库管理器
            update_tracker: 数据更新追踪器
            rate_limit_per_minute: 每分钟最大请求数（默认50，Tushare免费配额）
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self.update_tracker = update_tracker or get_update_tracker()
        
        self.tushare_fetcher = TushareFetcher(rate_limit_per_minute=rate_limit_per_minute)
        self.efinance_fetcher = EfinanceFetcher(sleep_min=1.5, sleep_max=3.0)
        
        if not self.tushare_fetcher.is_available():
            logger.error("Tushare API 不可用，请检查 Token 配置")
            raise RuntimeError("Tushare API 不可用，请检查 Token 配置")
        
        self._should_stop = False
        
        logger.info(f"TushareDataDownloader 初始化成功，速率限制：{rate_limit_per_minute} 次/分钟")
    
    def stop(self):
        """
        停止下载
        """
        logger.info("TushareDataDownloader 收到停止信号")
        self._should_stop = True
    
    def _estimate_trading_days(self, start_date, end_date):
        """
        估算日期范围内的交易日数量
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            估算的交易日数量
        """
        total_days = (end_date - start_date).days + 1
        trading_days = int(total_days * 250 / 365)
        return max(1, trading_days)
    
    def _calculate_tushare_batch_size(self, start_date, end_date):
        """
        根据日期范围动态计算 Tushare 批量大小
        
        公式：batch_size = floor(5000 / 交易日数量)
        边界：最小 1 只，最大 100 只
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            计算得到的批量大小
        """
        trading_days = self._estimate_trading_days(start_date, end_date)
        
        batch_size = 5000 // trading_days
        
        min_batch = 1
        max_batch = 100
        
        batch_size = max(min_batch, min(batch_size, max_batch))
        
        logger.info(f"动态计算 Tushare 批量大小: {batch_size} 只/批 "
                   f"(交易日估算: {trading_days} 天, "
                   f"5000/{trading_days} = {5000/trading_days:.2f})")
        
        return batch_size
    
    def _save_stock_data(self, df: pd.DataFrame, stock_code: str):
        """
        保存单只股票的数据到数据库
        
        Args:
            df: 股票数据 DataFrame
            stock_code: 股票代码
        """
        if df is None or df.empty:
            return
        
        saved_count = 0
        
        try:
            with self.db_manager.session_scope() as session:
                record_dates = []
                valid_rows = []
                
                for _, row in df.iterrows():
                    code = stock_code
                    
                    record_date = None
                    if 'date' in row:
                        d = row['date']
                        if isinstance(d, str):
                            record_date = datetime.strptime(d, '%Y-%m-%d').date()
                        elif hasattr(d, 'date'):
                            record_date = d.date()
                        else:
                            record_date = d
                    
                    if record_date:
                        record_dates.append(record_date)
                        valid_rows.append((record_date, row))
                
                if not record_dates:
                    return
                
                existing_records = {
                    r.date: r for r in session.execute(
                        select(StockDaily).where(
                            and_(
                                StockDaily.code == stock_code,
                                StockDaily.date.in_(record_dates)
                            )
                        )
                    ).scalars().all()
                }
                
                for record_date, row in valid_rows:
                    try:
                        code = stock_code
                        
                        existing = existing_records.get(record_date)
                        
                        if existing:
                            if 'open' in row:
                                existing.open = row.get('open')
                            if 'high' in row:
                                existing.high = row.get('high')
                            if 'low' in row:
                                existing.low = row.get('low')
                            if 'close' in row:
                                existing.close = row.get('close')
                            if 'volume' in row:
                                existing.volume = row.get('volume')
                            if 'amount' in row:
                                existing.amount = row.get('amount')
                            if 'pct_chg' in row:
                                existing.pct_chg = row.get('pct_chg')
                            saved_count += 1
                        else:
                            new_record = StockDaily(
                                code=code,
                                date=record_date,
                                open=row.get('open'),
                                high=row.get('high'),
                                low=row.get('low'),
                                close=row.get('close'),
                                volume=row.get('volume', 0),
                                amount=row.get('amount', 0),
                                pct_chg=row.get('pct_chg')
                            )
                            session.add(new_record)
                            saved_count += 1
                    except Exception as row_error:
                        logger.warning(f"Failed to process row for {stock_code} on {record_date}: {row_error}, skipping this record")
                        continue
                
                logger.debug(f"Saved {saved_count} records for {stock_code}")
        
        except Exception as e:
            logger.error(f"Error saving data for {stock_code}: {e}")
    
    def _download_single_stock_from_tushare(
        self,
        stock_code: str,
        start_date: date,
        end_date: date
    ) -> Tuple[bool, int, str, bool]:
        """
        尝试从 Tushare 下载单只股票的数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Tuple of (success, records_count, error_message, need_to_wait)
            - need_to_wait: True 表示 Tushare 需要等待配额
        """
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            df = self.tushare_fetcher.get_daily_data(
                stock_code,
                start_date=start_str,
                end_date=end_str
            )
            
            if df is not None and not df.empty:
                self._save_stock_data(df, stock_code)
                self.update_tracker.update_record(
                    stock_code,
                    data_start_date=start_date,
                    data_end_date=end_date
                )
                logger.debug(f"Successfully downloaded {stock_code} from Tushare")
                return True, len(df), "", False
            else:
                return False, 0, "No data returned from Tushare", False
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检查是否是配额超限错误
            need_to_wait = any(
                keyword in error_msg 
                for keyword in ['quota', '配额', 'limit', 'rate limit', 'rate_limit']
            )
            
            if need_to_wait:
                logger.debug(f"Tushare quota reached for {stock_code}, will use other sources")
            else:
                logger.debug(f"Tushare failed for {stock_code}: {e}")
            
            return False, 0, str(e), need_to_wait
    
    def _download_single_stock_from_other_sources(
        self,
        stock_code: str,
        start_date: date,
        end_date: date
    ) -> Tuple[bool, int, str]:
        """
        从其他数据源下载单只股票的数据（不使用 Tushare 和 efinance）
        
        注意：根据要求，efinance 不用作单一股票获取 API 途径
        
        尝试顺序：AKshare → Baostock → Yahoo Finance
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Tuple of (success, records_count, error_message)
        """
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 1. 尝试 AKshare
        try:
            from data_provider.akshare_fetcher import AkshareFetcher
            akshare_fetcher = AkshareFetcher(sleep_min=1.0, sleep_max=2.0)
            df = akshare_fetcher.get_daily_data(
                stock_code,
                start_date=start_str,
                end_date=end_str
            )
            
            if df is not None and not df.empty:
                self._save_stock_data(df, stock_code)
                self.update_tracker.update_record(
                    stock_code,
                    data_start_date=start_date,
                    data_end_date=end_date
                )
                logger.debug(f"Successfully downloaded {stock_code} from akshare")
                return True, len(df), ""
        except Exception as e:
            logger.debug(f"akshare failed for {stock_code}: {e}")
        
        # 2. Pytdx 已禁用，不用
        # try:
        #     from data_provider.pytdx_fetcher import PytdxFetcher
        #     pytdx_fetcher = PytdxFetcher()
        #     df = pytdx_fetcher.get_daily_data(
        #         stock_code,
        #         start_date=start_str,
        #         end_date=end_str
        #     )
        #     
        #     if df is not None and not df.empty:
        #         self._save_stock_data(df, stock_code)
        #         self.update_tracker.update_record(
        #             stock_code,
        #             data_start_date=start_date,
        #             data_end_date=end_date
        #         )
        #         logger.debug(f"Successfully downloaded {stock_code} from pytdx")
        #         return True, len(df), ""
        # except Exception as e:
        #     logger.debug(f"pytdx failed for {stock_code}: {e}")
        
        # 2. 尝试 Baostock
        try:
            from data_provider.baostock_fetcher import BaostockFetcher
            baostock_fetcher = BaostockFetcher()
            df = baostock_fetcher.get_daily_data(
                stock_code,
                start_date=start_str,
                end_date=end_str
            )
            
            if df is not None and not df.empty:
                self._save_stock_data(df, stock_code)
                self.update_tracker.update_record(
                    stock_code,
                    data_start_date=start_date,
                    data_end_date=end_date
                )
                logger.debug(f"Successfully downloaded {stock_code} from baostock")
                return True, len(df), ""
        except Exception as e:
            logger.debug(f"baostock failed for {stock_code}: {e}")
        
        # 3. 尝试 Yahoo Finance
        try:
            from data_provider.yfinance_fetcher import YfinanceFetcher
            yfinance_fetcher = YfinanceFetcher()
            df = yfinance_fetcher.get_daily_data(
                stock_code,
                start_date=start_str,
                end_date=end_str
            )
            
            if df is not None and not df.empty:
                self._save_stock_data(df, stock_code)
                self.update_tracker.update_record(
                    stock_code,
                    data_start_date=start_date,
                    data_end_date=end_date
                )
                logger.debug(f"Successfully downloaded {stock_code} from yfinance")
                return True, len(df), ""
        except Exception as e:
            logger.debug(f"yahoo failed for {stock_code}: {e}")
        
        # 所有数据源都失败了
        return False, 0, "All sources failed"

    
    def _download_single_stock(
        self,
        stock_code: str,
        start_date: date,
        end_date: date
    ) -> Tuple[bool, int, str]:
        """
        下载单只股票的数据
        
        策略：
        1. 检查 Tushare 是否需要等待配额
        2. 如果不需要等待，使用 Tushare
        3. 如果需要等待，使用其他数据源，不等待
        4. 下一只股票继续检查并尝试 Tushare
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Tuple of (success, records_count, error_message)
        """
        # 检查 Tushare 是否需要等待配额
        need_to_wait = self.tushare_fetcher.will_need_to_wait()
        
        if not need_to_wait:
            # 不需要等待，尝试使用 Tushare
            logger.debug(f"Tushare quota available, trying Tushare for {stock_code}")
            success, records, error, _ = self._download_single_stock_from_tushare(
                stock_code, start_date, end_date
            )
            
            if success:
                return True, records, error
            
            # Tushare 失败了，使用其他数据源
            logger.debug(f"Tushare failed for {stock_code}, switching to other sources")
        
        # 需要等待配额或者 Tushare 失败了，使用其他数据源
        logger.debug(f"Using other sources for {stock_code} (need_to_wait={need_to_wait})")
        return self._download_single_stock_from_other_sources(
            stock_code, start_date, end_date
        )
    
    def _calculate_date_range(self, trading_days: int) -> Tuple[date, date]:
        """
        根据交易日数量计算准确的日期范围
        
        Args:
            trading_days: 需要的交易日数量
            
        Returns:
            (start_date, end_date) 元组
        """
        end_date = date.today()
        
        try:
            # 获取足够长的交易日历（获取过去 2 年的数据）
            start_calendar_date = end_date - timedelta(days=730)
            trade_dates = self.tushare_fetcher.get_trade_calendar(
                start_date=start_calendar_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                is_open='1'
            )
            
            if trade_dates and len(trade_dates) >= trading_days:
                # 使用交易日历计算准确的开始日期
                start_date = trade_dates[-trading_days]
                actual_trading_days = len(trade_dates[trade_dates.index(start_date):])
                logger.info(f"使用交易日历：需要 {trading_days} 个交易日，实际获取 {actual_trading_days} 个交易日")
                logger.info(f"日期范围：{start_date} 至 {end_date}")
                return start_date, end_date
        except Exception as e:
            logger.warning(f"获取交易日历失败: {e}，回退到日历日计算")
        
        # 回退到日历日计算
        start_date = end_date - timedelta(days=trading_days * 2 - 1)  # 多获取一些确保覆盖
        logger.info(f"使用日历日计算：日期范围 {start_date} 至 {end_date}")
        return start_date, end_date
    
    def _process_batch_data(self, df: pd.DataFrame, batch_stocks: List[str], start_date: date, end_date: date, stats: Dict[str, any]):
        """
        处理批量获取的数据，拆分并保存到每只股票
        
        Args:
            df: 批量获取的 DataFrame
            batch_stocks: 本批股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            stats: 统计信息字典
        """
        if df is None or df.empty:
            return
        
        # 标准化数据
        df = df.copy()
        
        # 列名映射
        column_mapping = {
            'trade_date': 'date',
            'vol': 'volume',
        }
        df = df.rename(columns=column_mapping)
        
        # 转换日期格式（YYYYMMDD -> YYYY-MM-DD）
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date
        
        # 成交量单位转换（Tushare 的 vol 单位是手，需要转换为股）
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * 100
        
        # 成交额单位转换（Tushare 的 amount 单位是千元，转换为元）
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 1000
        
        # 从 ts_code 中提取股票代码（去掉 .SH/.SZ）
        if 'ts_code' in df.columns:
            df['code'] = df['ts_code'].apply(lambda x: x.split('.')[0])
        
        # 按股票代码分组处理
        for stock_code in batch_stocks:
            try:
                # 筛选当前股票的数据
                stock_df = df[df['code'] == stock_code].copy()
                
                if stock_df is not None and not stock_df.empty:
                    # 保存数据
                    self._save_stock_data(stock_df, stock_code)
                    self.update_tracker.update_record(
                        stock_code,
                        data_start_date=start_date,
                        data_end_date=end_date
                    )
                    stats['stocks_success'] += 1
                    stats['total_records'] += len(stock_df)
                    logger.debug(f"Successfully processed {stock_code} from batch")
                else:
                    # 批量获取没有这只股票的数据，尝试单独获取
                    logger.debug(f"No data for {stock_code} in batch, trying individual download")
                    success, records, error = self._download_single_stock_from_other_sources(
                        stock_code, start_date, end_date
                    )
                    if success:
                        stats['stocks_success'] += 1
                        stats['total_records'] += records
                    else:
                        stats['stocks_failed'] += 1
                        stats['failed_stocks'].append({
                            'code': stock_code,
                            'error': error
                        })
            except Exception as e:
                logger.error(f"Error processing {stock_code} from batch: {e}")
                # 失败时尝试单独获取
                success, records, error = self._download_single_stock_from_other_sources(
                    stock_code, start_date, end_date
                )
                if success:
                    stats['stocks_success'] += 1
                    stats['total_records'] += records
                else:
                    stats['stocks_failed'] += 1
                    stats['failed_stocks'].append({
                        'code': stock_code,
                        'error': error
                    })
    
    def _process_efinance_batch_data(self, batch_result: Dict[str, pd.DataFrame], batch_stocks: List[str], start_date: date, end_date: date, stats: Dict[str, any]):
        """
        处理 efinance 批量获取的数据
        
        Args:
            batch_result: efinance 批量获取返回的字典 {code: DataFrame}
            batch_stocks: 本批股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            stats: 统计信息字典
        """
        if batch_result is None:
            return
        
        success_stocks = set()
        
        for stock_code, df in batch_result.items():
            if df is not None and not df.empty:
                try:
                    self._save_stock_data(df, stock_code)
                    self.update_tracker.update_record(
                        stock_code,
                        data_start_date=start_date,
                        data_end_date=end_date
                    )
                    stats['stocks_success'] += 1
                    stats['total_records'] += len(df)
                    success_stocks.add(stock_code)
                    logger.debug(f"Successfully processed {stock_code} from efinance batch")
                except Exception as e:
                    logger.warning(f"Error processing {stock_code} from efinance batch: {e}, will retry with other sources")
        
        # 补救措施：所有未成功获取的股票，都尝试用其他数据源单独获取
        for stock_code in batch_stocks:
            if stock_code not in success_stocks:
                logger.debug(f"Trying to recover {stock_code} with other sources")
                success, records, error = self._download_single_stock_from_other_sources(
                    stock_code, start_date, end_date
                )
                if success:
                    stats['stocks_success'] += 1
                    stats['total_records'] += records
                else:
                    stats['stocks_failed'] += 1
                    stats['failed_stocks'].append({
                        'code': stock_code,
                        'error': error
                    })
    
    def download_data(
        self,
        stock_codes: Optional[List[str]] = None,
        days: int = 365,
        efinance_batch_size: Optional[int] = None,
        tushare_batch_size: Optional[int] = None
    ) -> Dict[str, any]:
        """
        下载股票数据
        
        策略（优先级从高到低）：
        1. 使用 Tushare 批量获取每批 tushare_batch_size 只股票的数据（稳定可靠）
        2. Tushare 需要等待配额或失败时，使用 efinance 批量获取每批 efinance_batch_size 只股票的数据（免费，无配额限制）
        3. 如果都失败，使用其他数据源单独获取
        
        注意：efinance 不用作单一股票获取 API 途径，只用于批量获取
        
        Args:
            stock_codes: 股票代码列表（默认所有股票）
            days: 获取多少个交易日的数据（默认 365）
            efinance_batch_size: efinance 每批处理多少只股票（默认 50）
            tushare_batch_size: Tushare 每批处理多少只股票（默认 13）
            
        Returns:
            下载统计信息
        """
        # 从配置读取默认值
        config = get_config()
        if efinance_batch_size is None:
            efinance_batch_size = config.efinance_batch_size
        
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
        
        # 1. 过滤特定板块的股票代码（科创板、创业板、北交所等）
        original_count = len(stock_codes)
        stock_codes = filter_special_stock_codes(stock_codes)
        filtered_count = original_count - len(stock_codes)
        
        # 使用交易日历计算准确的日期范围
        start_date, end_date = self._calculate_date_range(days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 动态计算 Tushare 批量大小（仅在用户未指定时）
        if tushare_batch_size is None:
            tushare_batch_size = self._calculate_tushare_batch_size(start_date, end_date)
        
        logger.info(f"开始下载数据：{len(stock_codes)} 只股票，{days} 个交易日数据（{start_date} 至 {end_date}）")
        
        stats = {
            'total_stocks': len(stock_codes),
            'stocks_success': 0,
            'stocks_failed': 0,
            'total_records': 0,
            'failed_stocks': [],
            'start_time': datetime.now(),
            'end_time': None
        }
        
        total_batches = (len(stock_codes) + tushare_batch_size - 1) // tushare_batch_size
        
        print("\n" + "=" * 80)
        print(f"开始下载：{len(stock_codes)} 只股票，{days} 个交易日数据")
        if filtered_count > 0:
            print(f"（已过滤 {filtered_count} 只北交所股票）")
        print(f"Tushare批量：{tushare_batch_size} 只/批，efinance批量：{efinance_batch_size} 只/批")
        print(f"共 {total_batches} 批")
        print(f"日期范围：{start_date} 至 {end_date}")
        print(f"数据源优先级：Tushare (批量) > efinance (批量) > 其他数据源 (单独)")
        print("=" * 80 + "\n")
        
        try:
            pbar = tqdm(range(total_batches), desc="总体进度", unit="batch")
            for batch_idx in pbar:
                # 检查停止标志
                if self._should_stop:
                    logger.info("下载被用户终止")
                    break
                
                start_idx = batch_idx * tushare_batch_size
                end_idx = min((batch_idx + 1) * tushare_batch_size, len(stock_codes))
                batch_stocks = stock_codes[start_idx:end_idx]
                
                logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批：{len(batch_stocks)} 只股票")
                
                # 策略 1: 优先使用 Tushare 批量获取（稳定可靠）
                pbar.set_description(f"总体进度 | 数据源: Tushare")
                # 检查 Tushare 是否需要等待配额
                need_to_wait = self.tushare_fetcher.will_need_to_wait()
                
                if not need_to_wait:
                    # 不需要等待，尝试批量获取
                    try:
                        logger.debug(f"Trying Tushare batch download for {len(batch_stocks)} stocks")
                        batch_df = self.tushare_fetcher.get_daily_data_batch(
                            batch_stocks,
                            start_date=start_str,
                            end_date=end_str
                        )
                        
                        if batch_df is not None and not batch_df.empty:
                            # 成功批量获取，处理数据
                            logger.debug(f"Tushare batch download successful, processing {len(batch_df)} records")
                            self._process_batch_data(batch_df, batch_stocks, start_date, end_date, stats)
                            continue
                    except Exception as e:
                        logger.debug(f"Tushare batch download failed: {e}, trying efinance")
                
                # 策略 2: Tushare 需要等待或失败，尝试 efinance 批量获取
                pbar.set_description(f"总体进度 | 数据源: efinance")
                try:
                    logger.debug(f"Trying efinance batch download for {len(batch_stocks)} stocks")
                    batch_result = self.efinance_fetcher.get_daily_data_batch(
                        batch_stocks,
                        start_date=start_str,
                        end_date=end_str
                    )
                    
                    if batch_result and len(batch_result) > 0:
                        # 成功批量获取，处理数据
                        logger.debug(f"efinance batch download successful, got {len(batch_result)} stocks")
                        self._process_efinance_batch_data(batch_result, batch_stocks, start_date, end_date, stats)
                        continue
                except Exception as e:
                    logger.debug(f"efinance batch download failed: {e}, falling back to individual download")
                
                # 策略 3: 都失败或需要等待，使用其他数据源单独获取
                pbar.set_description(f"总体进度 | 数据源: 其他")
                logger.debug(f"Using other sources for batch {batch_idx + 1}")
                for stock_code in tqdm(
                    batch_stocks,
                    desc=f"  第 {batch_idx + 1} 批",
                    unit="stock",
                    leave=False
                ):
                    success, records, error = self._download_single_stock_from_other_sources(
                        stock_code, start_date, end_date
                    )
                    
                    if success:
                        stats['stocks_success'] += 1
                        stats['total_records'] += records
                    else:
                        stats['stocks_failed'] += 1
                        stats['failed_stocks'].append({
                            'code': stock_code,
                            'error': error
                        })
        
        except KeyboardInterrupt:
            print("\n\n下载被用户中断")
        except Exception as e:
            print(f"\n\n下载过程中发生错误：{e}")
            logger.error(f"Download error: {e}")
        
        stats['end_time'] = datetime.now()
        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        
        print("\n" + "=" * 80)
        print("下载完成！")
        print("=" * 80)
        print(f"总股票数：{stats['total_stocks']}")
        print(f"成功：{stats['stocks_success']}")
        print(f"失败：{stats['stocks_failed']}")
        print(f"总记录数：{stats['total_records']}")
        print(f"总耗时：{duration:.2f} 秒")
        
        if stats['failed_stocks']:
            print(f"\n失败股票 ({len(stats['failed_stocks'])}):")
            for i, failed in enumerate(stats['failed_stocks'][:20], 1):
                print(f"  {i}. {failed['code']}: {failed['error']}")
            if len(stats['failed_stocks']) > 20:
                print(f"  ... 还有 {len(stats['failed_stocks']) - 20} 只")
        
        print("=" * 80 + "\n")
        
        return stats
    
    def download_data_for_date_range(
        self,
        stock_codes: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        efinance_batch_size: Optional[int] = None,
        tushare_batch_size: Optional[int] = None
    ) -> Dict[str, any]:
        """
        下载指定日期范围的股票数据（与download_data逻辑完全相同，但直接接受日期范围
        
        策略（优先级从高到低）：
        1. 使用 Tushare 批量获取每批 tushare_batch_size 只股票的数据（稳定可靠）
        2. Tushare 需要等待配额或失败时，使用 efinance 批量获取每批 efinance_batch_size 只股票的数据（免费，无配额限制）
        3. 如果都失败，使用其他数据源单独获取
        
        注意：efinance 不用作单一股票获取 API 途径，只用于批量获取
        
        Args:
            stock_codes: 股票代码列表（默认所有股票）
            start_date: 开始日期
            end_date: 结束日期
            efinance_batch_size: efinance 每批处理多少只股票（默认 50）
            tushare_batch_size: Tushare 每批处理多少只股票（默认 20）
            
        Returns:
            下载统计信息
        """
        # 从配置读取默认值
        config = get_config()
        if efinance_batch_size is None:
            efinance_batch_size = config.efinance_batch_size
        
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
        
        # 1. 过滤特定板块的股票代码（科创板、创业板、北交所等）
        original_count = len(stock_codes)
        stock_codes = filter_special_stock_codes(stock_codes)
        filtered_count = original_count - len(stock_codes)
        
        # 如果没有指定日期，默认使用默认值
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            # 默认获取最近365个交易日（约1.5个日历年）
            start_date = end_date - timedelta(days=540)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 动态计算 Tushare 批量大小（仅在用户未指定时）
        if tushare_batch_size is None:
            tushare_batch_size = self._calculate_tushare_batch_size(start_date, end_date)
        
        # 估算交易日数量（每周约5个交易日）
        total_calendar_days = (end_date - start_date).days + 1
        days = int(total_calendar_days * 5 / 7)
        
        logger.info(f"开始下载数据：{len(stock_codes)} 只股票，{days} 个交易日数据（{start_date} 至 {end_date}）")
        
        stats = {
            'total_stocks': len(stock_codes),
            'stocks_success': 0,
            'stocks_failed': 0,
            'total_records': 0,
            'failed_stocks': [],
            'start_time': datetime.now(),
            'end_time': None
        }
        
        total_batches = (len(stock_codes) + tushare_batch_size - 1) // tushare_batch_size
        
        print("\n" + "=" * 80)
        print(f"开始下载：{len(stock_codes)} 只股票，{days} 个交易日数据")
        if filtered_count > 0:
            print(f"（已过滤 {filtered_count} 只北交所股票）")
        print(f"Tushare批量：{tushare_batch_size} 只/批，efinance批量：{efinance_batch_size} 只/批")
        print(f"共 {total_batches} 批")
        print(f"日期范围：{start_date} 至 {end_date}")
        print(f"数据源优先级：Tushare (批量) > efinance (批量) > 其他数据源 (单独)")
        print("=" * 80 + "\n")
        
        try:
            pbar = tqdm(range(total_batches), desc="总体进度", unit="batch")
            for batch_idx in pbar:
                # 检查停止标志
                if self._should_stop:
                    logger.info("下载被用户终止")
                    break
                
                start_idx = batch_idx * tushare_batch_size
                end_idx = min((batch_idx + 1) * tushare_batch_size, len(stock_codes))
                batch_stocks = stock_codes[start_idx:end_idx]
                
                logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批：{len(batch_stocks)} 只股票")
                
                # 策略 1: 优先使用 Tushare 批量获取（稳定可靠）
                pbar.set_description(f"总体进度 | 数据源: Tushare")
                # 检查 Tushare 是否需要等待配额
                need_to_wait = self.tushare_fetcher.will_need_to_wait()
                
                if not need_to_wait:
                    # 不需要等待，尝试批量获取
                    try:
                        logger.debug(f"Trying Tushare batch download for {len(batch_stocks)} stocks")
                        batch_df = self.tushare_fetcher.get_daily_data_batch(
                            batch_stocks,
                            start_date=start_str,
                            end_date=end_str
                        )
                        
                        if batch_df is not None and not batch_df.empty:
                            # 成功批量获取，处理数据
                            logger.debug(f"Tushare batch download successful, processing {len(batch_df)} records")
                            self._process_batch_data(batch_df, batch_stocks, start_date, end_date, stats)
                            continue
                    except Exception as e:
                        logger.debug(f"Tushare batch download failed: {e}, trying efinance")
                
                # 策略 2: Tushare 需要等待或失败，尝试 efinance 批量获取
                pbar.set_description(f"总体进度 | 数据源: efinance")
                try:
                    logger.debug(f"Trying efinance batch download for {len(batch_stocks)} stocks")
                    batch_result = self.efinance_fetcher.get_daily_data_batch(
                        batch_stocks,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if batch_result['success'] and not batch_result['data'].empty:
                        logger.debug(f"efinance batch download successful, processing {len(batch_result['data'])} records")
                        self._process_batch_data(batch_result['data'], batch_stocks, start_date, end_date, stats)
                        continue
                    else:
                        logger.debug(f"efinance batch failed or empty, trying individual sources")
                except Exception as e:
                    logger.debug(f"efinance batch download failed: {e}, trying individual sources")
                
                # 策略 3: 两种批量都失败，使用单独的数据源
                pbar.set_description(f"总体进度 | 数据源: 单只获取")
                for stock_code in batch_stocks:
                    try:
                        logger.debug(f"Trying individual download for {stock_code}")
                        # 尝试多种数据源
                        df = self._download_single_stock(stock_code, start_date, end_date)
                        if df is not None and not df.empty:
                            logger.debug(f"Individual download successful for {stock_code}, processing {len(df)} records")
                            self._process_single_stock_data(df, stock_code, stats)
                        else:
                            logger.debug(f"No data for {stock_code} from any source")
                            stats['stocks_failed'] += 1
                            stats['failed_stocks'].append({
                                'code': stock_code,
                                'error': 'No data available from any source'
                            })
                    except Exception as e:
                        logger.error(f"Failed to download {stock_code}: {e}")
                        stats['stocks_failed'] += 1
                        stats['failed_stocks'].append({
                            'code': stock_code,
                            'error': str(e)
                        })
        
        except Exception as e:
            logger.error(f"下载过程出错: {e}", exc_info=True)
            stats['error'] = str(e)
        finally:
            stats['end_time'] = datetime.now()
            duration = (stats['end_time'] - stats['start_time']).total_seconds()
            print("\n" + "=" * 80)
            print("下载完成！")
            print(f"总股票数：{stats['total_stocks']}")
            print(f"成功：{stats['stocks_success']}")
            print(f"失败：{stats['stocks_failed']}")
            print(f"总记录数：{stats['total_records']}")
            print(f"总耗时：{duration:.2f} 秒")
            
            if stats['failed_stocks']:
                print(f"\n失败股票 ({len(stats['failed_stocks'])}):")
                for i, failed in enumerate(stats['failed_stocks'][:20], 1):
                    print(f"  {i}. {failed['code']}: {failed['error']}")
                if len(stats['failed_stocks']) > 20:
                    print(f"  ... 还有 {len(stats['failed_stocks']) - 20} 只")
            
            print("=" * 80 + "\n")
            
            return stats


_tushare_downloader_instance: Optional[TushareDataDownloader] = None


def get_tushare_downloader(rate_limit_per_minute: int = 50) -> TushareDataDownloader:
    """
    获取全局 Tushare 数据下载器实例
    
    Args:
        rate_limit_per_minute: 每分钟最大请求数（默认50，Tushare免费配额）
        
    Returns:
        TushareDataDownloader 实例
    """
    global _tushare_downloader_instance
    if _tushare_downloader_instance is None:
        _tushare_downloader_instance = TushareDataDownloader(rate_limit_per_minute=rate_limit_per_minute)
    return _tushare_downloader_instance
