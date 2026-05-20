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
from typing import Any, Callable, Optional, List, Tuple, Dict
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

    def reset_rate_limit(self) -> None:
        """
        重置 Tushare 速率限制计数器
        
        在每次新的 screen 任务开始前调用，确保计数器不会跨任务累积。
        """
        self.tushare_fetcher.reset_rate_limit()

    def _estimate_trading_days(self, start_date, end_date):
        """
        计算日期范围内的交易日数量（使用精确的交易日历）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日数量
        """
        try:
            from .trading_calendar import get_trading_calendar
            trading_calendar = get_trading_calendar()
            trading_days_list = trading_calendar.get_trading_days(start_date, end_date)
            trading_days = len(trading_days_list)
            logger.debug(f"精确计算交易日数量: {trading_days} 天 ({start_date} 至 {end_date})")
            return max(1, trading_days)
        except Exception as e:
            logger.warning(f"使用精确交易日历失败: {e}，回退到估算方法")
            total_days = (end_date - start_date).days + 1
            trading_days = int(total_days * 250 / 365)
            return max(1, trading_days)
    
    def _calculate_tushare_batch_size(self, start_date, end_date, actual_trading_days=None):
        """
        根据日期范围动态计算 Tushare 批量大小
        
        公式：batch_size = floor(5000 / 交易日数量)
        边界：最小 1 只
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            actual_trading_days: 实际需要的交易日数量（如果提供则优先使用）
            
        Returns:
            计算得到的批量大小
        """
        if actual_trading_days is not None:
            trading_days = actual_trading_days
        else:
            trading_days = self._estimate_trading_days(start_date, end_date)
        
        batch_size = 5000 // trading_days
        
        min_batch = 1
        max_batch = 100
        
        batch_size = max(min_batch, min(batch_size, max_batch))
        
        if actual_trading_days is not None:
            logger.info(f"动态计算 Tushare 批量大小: {batch_size} 只/批 "
                       f"(实际交易日: {trading_days} 天, "
                       f"5000/{trading_days} = {5000/trading_days:.2f})")
        else:
            logger.info(f"动态计算 Tushare 批量大小: {batch_size} 只/批 "
                       f"(交易日估算: {trading_days} 天, "
                       f"5000/{trading_days} = {5000/trading_days:.2f})")
        
        return batch_size
    
    def _save_stock_data(self, df: pd.DataFrame, stock_code: str) -> Optional[date]:
        """
        保存单只股票的数据到数据库，返回实际保存的最大数据日期。

        Args:
            df: 股票数据 DataFrame
            stock_code: 股票代码

        Returns:
            实际保存的最大数据日期，若未保存任何数据则返回 None
        """
        if df is None or df.empty:
            return None

        saved_count = 0
        actual_max_date = None

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
                    return None

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
                            # 策略4：如果有 volume_ratio 字段也更新它
                            if 'volume_ratio' in row and row.get('volume_ratio') is not None:
                                existing.volume_ratio = row.get('volume_ratio')
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
                                pct_chg=row.get('pct_chg'),
                                # 策略4：如果有 volume_ratio 字段也设置它
                                volume_ratio=row.get('volume_ratio') if 'volume_ratio' in row else None
                            )
                            session.add(new_record)
                            saved_count += 1
                    except Exception as row_error:
                        logger.warning(f"Failed to process row for {stock_code} on {record_date}: {row_error}, skipping this record")
                        continue

                actual_max_date = max(record_dates)
                logger.debug(f"Saved {saved_count} records for {stock_code}, 实际最新日期: {actual_max_date}")

        except Exception as e:
            logger.error(f"Error saving data for {stock_code}: {e}")
            return None

        return actual_max_date
    
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
                actual_max_date = self._save_stock_data(df, stock_code)
                if actual_max_date is not None:
                    self.update_tracker.update_record(
                        stock_code,
                        data_start_date=start_date,
                        data_end_date=actual_max_date
                    )
                logger.debug(f"Successfully downloaded {stock_code} from Tushare, 实际最新日期: {actual_max_date}")
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
    
    def _calculate_date_range(self, trading_days: int) -> Tuple[date, date, int]:
        """
        根据交易日数量计算准确的日期范围
        
        Args:
            trading_days: 需要的交易日数量
            
        Returns:
            (start_date, end_date, actual_trading_days) 元组
        """
        end_date = date.today()
        
        try:
            # 使用本地交易日历
            from .trading_calendar import get_trading_calendar
            trading_calendar = get_trading_calendar()
            
            # 获取足够长的交易日历（获取过去 2 年的数据）
            start_calendar_date = end_date - timedelta(days=730)
            trade_dates = trading_calendar.get_trading_days(start_calendar_date, end_date)
            
            if trade_dates and len(trade_dates) >= trading_days:
                # 使用交易日历计算准确的开始日期
                start_date = trade_dates[-trading_days]
                actual_trading_days = len(trade_dates[trade_dates.index(start_date):])
                logger.info(f"使用本地交易日历：需要 {trading_days} 个交易日，实际获取 {actual_trading_days} 个交易日")
                logger.info(f"日期范围：{start_date} 至 {end_date}")
                return start_date, end_date, actual_trading_days
            else:
                logger.warning(f"本地交易日历数据不足，需要 {trading_days} 个，实际只有 {len(trade_dates) if trade_dates else 0} 个")
                logger.info("尝试从 AKShare 更新交易日历...")
                try:
                    trading_calendar.refresh()
                    # 重新获取交易日历
                    trade_dates = trading_calendar.get_trading_days(start_calendar_date, end_date)
                    if trade_dates and len(trade_dates) >= trading_days:
                        start_date = trade_dates[-trading_days]
                        actual_trading_days = len(trade_dates[trade_dates.index(start_date):])
                        logger.info(f"使用更新后的本地交易日历：需要 {trading_days} 个交易日，实际获取 {actual_trading_days} 个交易日")
                        logger.info(f"日期范围：{start_date} 至 {end_date}")
                        return start_date, end_date, actual_trading_days
                    else:
                        logger.warning(f"更新后交易日历数据仍然不足，需要 {trading_days} 个，实际只有 {len(trade_dates) if trade_dates else 0} 个")
                except Exception as update_error:
                    logger.warning(f"更新交易日历失败: {update_error}")
        except Exception as e:
            logger.warning(f"获取本地交易日历失败: {e}，回退到日历日计算")
        
        # 回退到日历日计算（1.5倍自然日）
        start_date = end_date - timedelta(days=int(trading_days * 1.5))
        logger.info(f"使用日历日计算（1.5倍）：日期范围 {start_date} 至 {end_date}")
        return start_date, end_date, trading_days
    
    def _process_batch_data(self, batch_df: pd.DataFrame, batch_stocks: List[str], start_date: date, end_date: date, stats: Dict[str, Any]):
        """
        批量处理数据并保存到数据库（性能优化版）。

        使用单次数据库会话完成所有股票的数据保存：
        - 一次查询获取所有已存在记录
        - 批量插入新记录（session.add_all）
        - SQLAlchemy 自动追踪对已有记录的修改
        - 单次事务提交

        Args:
            batch_df: 批量获取的 DataFrame
            batch_stocks: 本批股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            stats: 统计信息字典
        """
        if batch_df is None or batch_df.empty:
            return

        batch_df = batch_df.copy()

        # 列名映射
        column_mapping = {
            'trade_date': 'date',
            'vol': 'volume',
        }
        batch_df = batch_df.rename(columns=column_mapping)

        # 转换日期格式（YYYYMMDD -> YYYY-MM-DD）
        if 'date' in batch_df.columns:
            batch_df['date'] = pd.to_datetime(batch_df['date'], format='%Y%m%d').dt.date

        # 成交量单位转换（Tushare 的 vol 单位是手，需要转换为股）
        if 'volume' in batch_df.columns:
            batch_df['volume'] = batch_df['volume'] * 100

        # 成交额单位转换（Tushare 的 amount 单位是千元，转换为元）
        if 'amount' in batch_df.columns:
            batch_df['amount'] = batch_df['amount'] * 1000

        # 从 ts_code 中提取股票代码（去掉 .SH/.SZ）
        if 'ts_code' in batch_df.columns:
            batch_df['code'] = batch_df['ts_code'].apply(lambda x: x.split('.')[0])

        # 过滤出实际存在于 DataFrame 中的股票代码
        codes_in_df = [c for c in batch_stocks if c in batch_df['code'].values]

        if not codes_in_df:
            return

        actual_max_dates = {}  # {code: max_date}

        try:
            with self.db_manager.session_scope() as session:
                # 批量查询这批股票在日期范围内的已有记录（一次查询）
                stmt = select(StockDaily).where(
                    and_(
                        StockDaily.code.in_(codes_in_df),
                        StockDaily.date.between(start_date, end_date)
                    )
                )
                existing_records = session.execute(stmt).scalars().all()
                # 构建 (code, date) -> record 的快速查找字典
                existing_dict = {(r.code, r.date): r for r in existing_records}

                new_records = []  # 批量收集待插入的新记录

                for stock_code in codes_in_df:
                    stock_df = batch_df[batch_df['code'] == stock_code]
                    if stock_df.empty:
                        continue

                    stock_max_date = None
                    stock_saved = 0

                    for _, row in stock_df.iterrows():
                        record_date = row['date']
                        if not isinstance(record_date, date):
                            continue

                        if stock_max_date is None or record_date > stock_max_date:
                            stock_max_date = record_date

                        key = (stock_code, record_date)

                        if key in existing_dict:
                            # 更新已有记录（SQLAlchemy 自动追踪变更）
                            record = existing_dict[key]
                            if 'open' in row and row['open'] is not None:
                                record.open = row['open']
                            if 'high' in row and row['high'] is not None:
                                record.high = row['high']
                            if 'low' in row and row['low'] is not None:
                                record.low = row['low']
                            if 'close' in row and row['close'] is not None:
                                record.close = row['close']
                            if 'volume' in row and row['volume'] is not None:
                                record.volume = row['volume']
                            if 'amount' in row and row['amount'] is not None:
                                record.amount = row['amount']
                            if 'pct_chg' in row and row['pct_chg'] is not None:
                                record.pct_chg = row['pct_chg']
                            if 'volume_ratio' in row and row['volume_ratio'] is not None:
                                record.volume_ratio = row['volume_ratio']
                            stock_saved += 1
                        else:
                            # 收集新记录，最后批量插入
                            new_records.append(StockDaily(
                                code=stock_code,
                                date=record_date,
                                open=row.get('open'),
                                high=row.get('high'),
                                low=row.get('low'),
                                close=row.get('close'),
                                volume=row.get('volume', 0),
                                amount=row.get('amount', 0),
                                pct_chg=row.get('pct_chg'),
                                volume_ratio=row.get('volume_ratio') if 'volume_ratio' in row else None
                            ))
                            stock_saved += 1

                    if stock_saved > 0 and stock_max_date is not None:
                        actual_max_dates[stock_code] = stock_max_date
                        stats['stocks_success'] += 1
                        stats['total_records'] += stock_saved
                    else:
                        stats['stocks_skipped'] = stats.get('stocks_skipped', 0) + 1

                # 批量插入所有新记录
                if new_records:
                    session.add_all(new_records)

            # session_scope 在此自动提交

            # 批量更新 tracker（使用独立的 session）
            if actual_max_dates:
                self.update_tracker.update_records_batch(
                    list(actual_max_dates.keys()),
                    data_start_date=start_date,
                    data_end_date=end_date
                )

        except Exception as e:
            logger.error(f"批量处理数据失败: {e}")
            stats['stocks_failed'] += len(codes_in_df)
            stats['failed_stocks'].extend([{'code': code, 'error': str(e)} for code in codes_in_df])
    
    def _process_efinance_batch_data(self, batch_result: Dict[str, pd.DataFrame], batch_stocks: List[str], start_date: date, end_date: date, stats: Dict[str, Any]):
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
                    actual_max_date = self._save_stock_data(df, stock_code)
                    if actual_max_date is not None:
                        self.update_tracker.update_record(
                            stock_code,
                            data_start_date=start_date,
                            data_end_date=actual_max_date
                        )
                    stats['stocks_success'] += 1
                    stats['total_records'] += len(df)
                    success_stocks.add(stock_code)
                    logger.debug(f"Successfully processed {stock_code} from efinance batch, 实际最新日期: {actual_max_date}")
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
    
    def _group_stocks_by_update_need(
        self, stock_codes: List[str], target_start: date, target_end: date, full_days: int
    ) -> Dict[str, dict]:
        """
        使用交易日历精确计算每只股票需要的交易日数量，按需分组下载。

        分组阈值（交易日数）：
          - skip: 0 天（最后更新 ≥ 今日）
          - micro: 1~5 天，batch_size = 5000 // 5 = 1000
          - small: 6~20 天，batch_size = 5000 // 20 = 250
          - medium: 21~60 天，batch_size = 5000 // 60 = 83
          - full: 61+ 天 / 新股票，batch_size = 5000 // full_days

        Args:
            stock_codes: 待处理的股票代码列表
            target_start: 全量更新的目标起始日期
            target_end: 目标结束日期（通常为今日）
            full_days: 全量更新的交易日数量（用于计算 full 组的 batch_size）

        Returns:
            分组信息字典，key 为组名，value 包含 stocks、start_date、end_date、batch_size
        """
        groups = {
            "micro":  {"stocks": [], "min_start": None, "max_trading_days": 5},
            "small":  {"stocks": [], "min_start": None, "max_trading_days": 20},
            "medium": {"stocks": [], "min_start": None, "max_trading_days": 60},
            "full":   {"stocks": [], "min_start": target_start, "max_trading_days": max(full_days, 150)},
        }
        skipped_count = 0
        incremental_count = 0
        full_count = 0

        for code in stock_codes:
            try:
                actual_start, actual_end, is_incremental = self.update_tracker.determine_update_range(
                    code, target_start, target_end
                )
            except Exception as e:
                logger.warning(f"获取 {code} 更新范围失败: {e}，回退到全量")
                actual_start, actual_end, is_incremental = target_start, target_end, False

            if actual_start is None:
                # 已是最新，无需更新
                skipped_count += 1
                continue

            if not is_incremental:
                # 新股票或无记录，全量下载
                groups["full"]["stocks"].append(code)
                full_count += 1
                continue

            # 增量更新：用交易日历精确计算需要的交易日数量
            try:
                trading_days = self._estimate_trading_days(actual_start, target_end)
            except Exception:
                trading_days = (target_end - actual_start).days + 1

            if trading_days <= 5:
                group_key = "micro"
            elif trading_days <= 20:
                group_key = "small"
            elif trading_days <= 60:
                group_key = "medium"
            else:
                group_key = "full"

            groups[group_key]["stocks"].append(code)
            if groups[group_key]["min_start"] is None or actual_start < groups[group_key]["min_start"]:
                groups[group_key]["min_start"] = actual_start
            incremental_count += 1

        # 构建返回结果
        result = {}
        for group_key, group_data in groups.items():
            if not group_data["stocks"]:
                continue

            start = group_data["min_start"] if group_data["min_start"] is not None else target_start
            max_td = group_data["max_trading_days"]
            batch_size = max(1, 5000 // max_td)

            result[group_key] = {
                "stocks": group_data["stocks"],
                "start_date": start,
                "end_date": target_end,
                "batch_size": batch_size,
                "max_trading_days": max_td,
            }

        logger.info(
            f"股票分组完成: skip={skipped_count}, micro={len(groups['micro']['stocks'])}, "
            f"small={len(groups['small']['stocks'])}, medium={len(groups['medium']['stocks'])}, "
            f"full={len(groups['full']['stocks'])}"
        )
        return result

    def download_data(
        self,
        stock_codes: Optional[List[str]] = None,
        days: Optional[int] = None,
        tushare_batch_size: Optional[int] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        下载股票数据
        
        策略：
        1. 只使用 Tushare 批量获取每批 tushare_batch_size 只股票的数据（稳定可靠）
        2. Tushare 失败或需要等待配额时，此批直接失败，不尝试其他数据源（节省时间）
        
        Args:
            stock_codes: 股票代码列表（默认所有股票）
            days: 获取多少个交易日的数据（默认 365）
            tushare_batch_size: Tushare 每批处理多少只股票（默认 13）
            verbose: 是否输出详细进度日志（含 tqdm 进度条），后台调用时建议设为 False
            progress_callback: 进度回调 (completed, total, current_code, current_name)
            
        Returns:
            下载统计信息
        """
        # 从配置读取默认值
        config = get_config()
        if days is None:
            days = config.update_data_default_days
        
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
        
        # 1. 过滤特定板块的股票代码（科创板、创业板、北交所等）
        original_count = len(stock_codes)
        stock_codes = filter_special_stock_codes(stock_codes)
        filtered_count = original_count - len(stock_codes)
        
        # 使用交易日历计算准确的日期范围
        start_date, end_date, actual_trading_days = self._calculate_date_range(days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 动态计算 Tushare 批量大小（仅在用户未指定时）
        if tushare_batch_size is None:
            tushare_batch_size = self._calculate_tushare_batch_size(start_date, end_date, actual_trading_days=days)
        
        logger.info(f"开始下载数据：{len(stock_codes)} 只股票，{days} 个交易日数据（{start_date} 至 {end_date}）")
        
        stats = {
            'total_stocks': len(stock_codes),
            'stocks_success': 0,
            'stocks_failed': 0,
            'stocks_skipped': 0,
            'total_records': 0,
            'failed_stocks': [],
            'start_time': datetime.now(),
            'end_time': None,
            'group_details': {}
        }

        # 修复 tracker 中可能虚高的记录日期（对比数据库实际数据）
        repaired = self.update_tracker.repair_invalid_dates(stock_codes)
        if repaired > 0:
            logger.info(f"修复了 {repaired} 条虚高的 tracker 记录")

        # 按增量需求分组
        groups = self._group_stocks_by_update_need(stock_codes, start_date, end_date, days)

        if verbose:
            logger.info("=" * 80)
            logger.info(f"开始下载：{len(stock_codes)} 只股票，{days} 个交易日数据")
            if filtered_count > 0:
                logger.info(f"（已过滤 {filtered_count} 只北交所股票）")
            logger.info(f"日期范围：{start_date} 至 {end_date}")
            logger.info(f"数据源：仅使用 Tushare 批量获取，失败则直接停止")
            logger.info("=" * 80)
            for gk, gi in groups.items():
                logger.info(f"  [{gk}] {len(gi['stocks'])} 只，{gi['max_trading_days']} 天，{gi['batch_size']} 只/批，"
                            f"日期: {gi['start_date']} ~ {gi['end_date']}")
            logger.info("=" * 80 + "\n")

        try:
            total_batches = sum(
                (len(gi['stocks']) + gi['batch_size'] - 1) // gi['batch_size']
                for gi in groups.values()
            )
            pbar = tqdm(range(total_batches), desc="总体进度", unit="batch", disable=not verbose)
            batch_count = 0

            for group_name, group_info in groups.items():
                group_stocks = group_info["stocks"]
                group_start = group_info["start_date"]
                group_end = group_info["end_date"]
                group_batch_size = group_info["batch_size"]
                group_start_str = group_start.strftime('%Y-%m-%d')
                group_end_str = group_end.strftime('%Y-%m-%d')

                if not group_stocks:
                    continue

                group_success = 0
                group_failed = 0
                group_batches = (len(group_stocks) + group_batch_size - 1) // group_batch_size

                for batch_idx_in_group in range(group_batches):
                    if self._should_stop:
                        logger.info("下载被用户终止")
                        break

                    start_idx = batch_idx_in_group * group_batch_size
                    end_idx = min((batch_idx_in_group + 1) * group_batch_size, len(group_stocks))
                    batch_stocks = group_stocks[start_idx:end_idx]

                    pbar.set_description(f"[{group_name}] {len(batch_stocks)} 只")

                    try:
                        batch_df = self.tushare_fetcher.get_daily_data_batch(
                            batch_stocks,
                            start_date=group_start_str,
                            end_date=group_end_str
                        )

                        if batch_df is not None and not batch_df.empty:
                            before_success = stats['stocks_success']
                            self._process_batch_data(
                                batch_df, batch_stocks, group_start, group_end, stats
                            )
                            batch_added = stats['stocks_success'] - before_success
                            group_success += batch_added
                        else:
                            logger.warning(f"Tushare 返回空数据，此批失败")
                            stats['stocks_failed'] += len(batch_stocks)
                            group_failed += len(batch_stocks)
                            stats['failed_stocks'].extend([
                                {'code': code, 'error': 'Tushare 返回空数据'}
                                for code in batch_stocks
                            ])
                    except Exception as e:
                        logger.warning(f"Tushare 批量获取失败: {e}，此批失败")
                        stats['stocks_failed'] += len(batch_stocks)
                        group_failed += len(batch_stocks)
                        stats['failed_stocks'].extend([
                            {'code': code, 'error': f'Tushare 批量获取失败: {e}'}
                            for code in batch_stocks
                        ])

                    pbar.update(1)
                    batch_count += 1
                    if progress_callback:
                        progress_callback(
                            stats['stocks_success'] + stats['stocks_failed'],
                            len(stock_codes),
                            batch_stocks[-1] if batch_stocks else "",
                            ""
                        )

                stats['group_details'][group_name] = {
                    'stocks': len(group_stocks),
                    'success': group_success,
                    'failed': group_failed,
                    'trading_days': group_info['max_trading_days'],
                    'date_range': f"{group_start_str} ~ {group_end_str}",
                }
        
        except KeyboardInterrupt:
            logger.info("\n下载被用户中断")
        except Exception as e:
            logger.error(f"\n下载过程中发生错误：{e}")
            logger.error(f"Download error: {e}")
        
        stats['end_time'] = datetime.now()
        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        
        if verbose:
            logger.info("=" * 80)
            logger.info("下载完成！")
            logger.info("=" * 80)
            logger.info(f"总股票数：{stats['total_stocks']}")
            logger.info(f"成功：{stats['stocks_success']}")
            logger.info(f"跳过（已最新）：{stats.get('stocks_skipped', 0)}")
            logger.info(f"失败：{stats['stocks_failed']}")
            logger.info(f"总记录数：{stats['total_records']}")
            for gk, gi in stats.get('group_details', {}).items():
                logger.info(f"  [{gk}] {gi['stocks']} 只 ({gi['trading_days']}天) | "
                            f"成功={gi['success']} 失败={gi['failed']} | {gi['date_range']}")
            logger.info(f"总耗时：{duration:.2f} 秒")

            if stats['failed_stocks']:
                logger.info(f"\n失败股票 ({len(stats['failed_stocks'])}):")
                for i, failed in enumerate(stats['failed_stocks'][:20], 1):
                    logger.info(f"  {i}. {failed['code']}: {failed['error']}")
                if len(stats['failed_stocks']) > 20:
                    logger.info(f"  ... 还有 {len(stats['failed_stocks']) - 20} 只")

            logger.info("=" * 80 + "\n")
        
        return stats
    
    def download_data_for_date_range(
        self,
        stock_codes: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        tushare_batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        下载指定日期范围的股票数据
        
        策略：
        1. 只使用 Tushare 批量获取每批 tushare_batch_size 只股票的数据（稳定可靠）
        2. Tushare 失败或需要等待配额时，此批直接失败，不尝试其他数据源（节省时间）
        
        Args:
            stock_codes: 股票代码列表（默认所有股票）
            start_date: 开始日期
            end_date: 结束日期
            tushare_batch_size: Tushare 每批处理多少只股票（默认 20）
            
        Returns:
            下载统计信息
        """
        # 从配置读取默认值
        config = get_config()
        
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
        print(f"Tushare批量：{tushare_batch_size} 只/批")
        print(f"共 {total_batches} 批")
        print(f"日期范围：{start_date} 至 {end_date}")
        print(f"数据源：仅使用 Tushare 批量获取，失败则直接停止")
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
                
                # 策略 1: 只使用 Tushare 批量获取，失败则直接停止
                pbar.set_description(f"总体进度 | 数据源: Tushare")
                # 检查 Tushare 是否需要等待配额
                need_to_wait = self.tushare_fetcher.will_need_to_wait()
                
                if need_to_wait:
                    logger.warning(f"Tushare 需要等待配额，跳过此批")
                    stats['stocks_failed'] += len(batch_stocks)
                    stats['failed_stocks'].extend([
                        {'code': code, 'error': 'Tushare 需要等待配额'} 
                        for code in batch_stocks
                    ])
                    continue
                
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
                    else:
                        logger.warning(f"Tushare 返回空数据，此批失败")
                        stats['stocks_failed'] += len(batch_stocks)
                        stats['failed_stocks'].extend([
                            {'code': code, 'error': 'Tushare 返回空数据'} 
                            for code in batch_stocks
                        ])
                except Exception as e:
                    logger.warning(f"Tushare 批量获取失败: {e}，此批失败")
                    stats['stocks_failed'] += len(batch_stocks)
                    stats['failed_stocks'].extend([
                        {'code': code, 'error': f'Tushare 批量获取失败: {e}'} 
                        for code in batch_stocks
                    ])
        
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
            print(f"跳过（无数据）：{stats.get('stocks_skipped', 0)}")
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
