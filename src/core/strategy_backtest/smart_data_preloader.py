# -*- coding: utf-8 -*-
"""
智能数据预加载器

在回测开始前检查数据库中已有数据，只下载缺失的部分
"""

import logging
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class SmartDataPreloader:
    """
    智能数据预加载器
    
    流程：
    1. 检查数据库已有数据
    2. 计算缺失的数据
    3. 使用 TushareDataDownloader 批量下载缺失部分
    4. 确保回测前所有数据都已就绪
    """
    
    def __init__(
        self,
        db_manager: Any,
        tushare_downloader: Any
    ):
        """
        初始化智能数据预加载器
        
        Args:
            db_manager: 数据库管理器
            tushare_downloader: Tushare数据下载器
        """
        self.db_manager = db_manager
        self.tushare_downloader = tushare_downloader
        self._should_stop = False
    
    def stop(self) -> None:
        """
        停止预加载
        """
        logger.info("智能数据预加载器收到停止信号")
        self._should_stop = True
        # 同时停止数据下载器
        if hasattr(self.tushare_downloader, 'stop'):
            self.tushare_downloader.stop()
    
    def check_data_coverage(
        self,
        stock_codes: List[str],
        target_start_date: date,
        target_end_date: date
    ) -> Dict[str, Dict]:
        """
        检查数据覆盖情况
        
        Args:
            stock_codes: 股票代码列表
            target_start_date: 目标开始日期
            target_end_date: 目标结束日期
            
        Returns:
            {
                stock_code: {
                    'status': 'complete' | 'partial' | 'missing',
                    'db_min_date': date | None,
                    'db_max_date': date | None,
                    'need_start_date': date | None,
                    'need_end_date': date | None
                }
            }
        """
        logger.info("=== 检查数据覆盖情况 ===")
        logger.info(f"目标日期范围: {target_start_date} 至 {target_end_date}")
        logger.info(f"股票数量: {len(stock_codes)}")
        
        # 批量检查数据覆盖
        coverage = self.db_manager.check_data_coverage(
            stock_codes,
            target_start_date,
            target_end_date
        )
        
        result = {}
        complete_count = 0
        partial_count = 0
        missing_count = 0
        
        for stock_code, (db_min_date, db_max_date) in coverage.items():
            stock_info = {
                'db_min_date': db_min_date,
                'db_max_date': db_max_date,
                'need_start_date': None,
                'need_end_date': None
            }
            
            if db_min_date is None or db_max_date is None:
                # 完全没有数据
                stock_info['status'] = 'missing'
                stock_info['need_start_date'] = target_start_date
                stock_info['need_end_date'] = target_end_date
                missing_count += 1
            elif db_min_date <= target_start_date and db_max_date >= target_end_date:
                # 完全覆盖
                stock_info['status'] = 'complete'
                complete_count += 1
            else:
                # 部分覆盖，计算需要补充的区间
                stock_info['status'] = 'partial'
                
                # 计算需要的起始日期
                if db_min_date > target_start_date:
                    stock_info['need_start_date'] = target_start_date
                else:
                    stock_info['need_start_date'] = db_max_date + timedelta(days=1)
                
                # 计算需要的结束日期
                stock_info['need_end_date'] = target_end_date
                
                partial_count += 1
            
            result[stock_code] = stock_info
        
        logger.info(f"数据覆盖检查完成:")
        logger.info(f"  完全覆盖: {complete_count} 只")
        logger.info(f"  部分覆盖: {partial_count} 只")
        logger.info(f"  完全缺失: {missing_count} 只")
        
        return result
    
    def _calculate_unified_date_range(
        self,
        coverage: Dict[str, Dict]
    ) -> Tuple[Optional[date], Optional[date]]:
        """
        计算所有股票数据覆盖范围的并集
        
        Args:
            coverage: 数据覆盖情况字典
            
        Returns:
            (min_date, max_date) - 所有股票的最早和最晚数据日期
        """
        all_min_dates = []
        all_max_dates = []
        
        for info in coverage.values():
            if info['db_min_date']:
                all_min_dates.append(info['db_min_date'])
            if info['db_max_date']:
                all_max_dates.append(info['db_max_date'])
        
        if not all_min_dates or not all_max_dates:
            return None, None
        
        overall_min_date = min(all_min_dates)
        overall_max_date = max(all_max_dates)
        
        return overall_min_date, overall_max_date
    
    def ensure_data_available(
        self,
        stock_codes: List[str],
        target_start_date: date,
        target_end_date: date
    ) -> None:
        """
        确保所有数据都可用，使用与stock_selector相同的逻辑
        
        Args:
            stock_codes: 股票代码列表
            target_start_date: 目标开始日期
            target_end_date: 目标结束日期
        """
        logger.info("=== 智能数据预加载开始 ===")
        logger.info("使用与stock_selector相同的逻辑")
        
        # 检查终止标志
        if self._should_stop:
            logger.info("智能数据预加载已被终止")
            return
        
        # 检查数据覆盖情况
        coverage = self.check_data_coverage(
            stock_codes,
            target_start_date,
            target_end_date
        )
        
        # 再次检查终止标志
        if self._should_stop:
            logger.info("智能数据预加载已被终止")
            return
        
        # 计算所有股票数据覆盖范围的并集
        overall_min_date, overall_max_date = self._calculate_unified_date_range(coverage)
        
        if overall_min_date and overall_max_date:
            logger.info(f"数据库中已有数据范围: {overall_min_date} 至 {overall_max_date}")
        else:
            logger.info("数据库中没有已有数据")
        
        logger.info(f"目标日期范围: {target_start_date} 至 {target_end_date}")
        
        # 再次检查终止标志
        if self._should_stop:
            logger.info("智能数据预加载已被终止")
            return
        
        # 检查是否真的需要下载数据，并计算精确的下载范围
        need_download = False
        stocks_need_data = []
        download_start_date: Optional[date] = None
        download_end_date: Optional[date] = None
        
        for stock_code, info in coverage.items():
            if info['status'] in ['partial', 'missing']:
                need_download = True
                if info['need_start_date'] and info['need_end_date']:
                    stocks_need_data.append(stock_code)
                    # 更新下载范围
                    if download_start_date is None or info['need_start_date'] < download_start_date:
                        download_start_date = info['need_start_date']
                    if download_end_date is None or info['need_end_date'] > download_end_date:
                        download_end_date = info['need_end_date']
        
        if not need_download:
            logger.info("所有数据已完全覆盖，无需下载")
            logger.info("=== 智能数据预加载完成 ===")
            return
        
        logger.info(f"需要补充数据的股票数量: {len(stocks_need_data)} 只")
        logger.info(f"下载日期范围: {download_start_date} 至 {download_end_date}")
        
        # 再次检查终止标志
        if self._should_stop:
            logger.info("智能数据预加载已被终止")
            return
        
        # 使用与stock_selector --update-data完全相同的逻辑（但使用指定日期范围）
        logger.info("使用与stock_selector相同的Tushare下载器（指定日期范围）...")
        
        try:
            stats = self.tushare_downloader.download_data_for_date_range(
                stock_codes=stocks_need_data,
                start_date=download_start_date,
                end_date=download_end_date,
                efinance_batch_size=50,
                tushare_batch_size=13
            )
            
            logger.info(f"批量下载完成:")
            logger.info(f"  成功: {stats.get('stocks_success', 0)} 只")
            logger.info(f"  失败: {stats.get('stocks_failed', 0)} 只")
            logger.info(f"  总记录: {stats.get('total_records', 0)} 条")
            
        except KeyboardInterrupt:
            logger.info("智能数据预加载被用户中断 (Ctrl+C)")
            self._should_stop = True
        except Exception as e:
            logger.error(f"批量下载失败: {e}", exc_info=True)
            # 即使失败也继续，因为可能已经下载了部分数据
        
        logger.info("=== 智能数据预加载完成 ===")
