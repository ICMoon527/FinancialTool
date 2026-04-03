#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断工具 - 检查数据库数据完整性
"""

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_database_data():
    """诊断数据库数据完整性"""
    logger.info("=" * 80)
    logger.info("数据库数据完整性诊断")
    logger.info("=" * 80)
    
    from src.storage import DatabaseManager
    from src.storage import StockDaily
    from sqlalchemy import select, and_, func, distinct
    
    db_manager = DatabaseManager.get_instance()
    
    # 1. 检查数据库中的所有股票
    logger.info("\n" + "=" * 80)
    logger.info("步骤1: 检查数据库中的股票代码")
    logger.info("=" * 80)
    
    with db_manager.get_session() as session:
        # 获取数据库中所有不同的股票代码
        result = session.execute(
            select(distinct(StockDaily.code))
        ).scalars().all()
        
        db_stock_codes = set(result)
        logger.info(f"数据库中共有 {len(db_stock_codes)} 只不同的股票")
        if len(db_stock_codes) <= 20:
            logger.info(f"数据库中的股票代码: {sorted(db_stock_codes)}")
        else:
            logger.info(f"数据库中的股票代码（前20只）: {sorted(db_stock_codes)[:20]}")
    
    # 2. 检查股票池
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 获取并检查股票池")
    logger.info("=" * 80)
    
    from stock_selector.stock_pool import get_all_stock_codes, filter_special_stock_codes
    
    all_stock_codes = get_all_stock_codes()
    logger.info(f"从stock_pool获取的所有股票: {len(all_stock_codes)} 只")
    
    filtered_stock_codes = filter_special_stock_codes(all_stock_codes)
    logger.info(f"过滤特殊股票后的股票池: {len(filtered_stock_codes)} 只")
    
    stock_pool_set = set(filtered_stock_codes)
    
    # 3. 对比股票池和数据库
    logger.info("\n" + "=" * 80)
    logger.info("步骤3: 对比股票池和数据库")
    logger.info("=" * 80)
    
    missing_in_db = stock_pool_set - db_stock_codes
    extra_in_db = db_stock_codes - stock_pool_set
    common = stock_pool_set & db_stock_codes
    
    logger.info(f"股票池中但数据库中没有的股票: {len(missing_in_db)} 只")
    if len(missing_in_db) <= 30:
        logger.info(f"缺失的股票: {sorted(missing_in_db)}")
    else:
        logger.info(f"缺失的股票（前30只）: {sorted(missing_in_db)[:30]}")
    
    logger.info(f"数据库中有但股票池中没有的股票: {len(extra_in_db)} 只")
    if len(extra_in_db) <= 20:
        logger.info(f"额外的股票: {sorted(extra_in_db)}")
    
    logger.info(f"两者都有的股票: {len(common)} 只")
    
    # 4. 检查具体缺失的股票
    logger.info("\n" + "=" * 80)
    logger.info("步骤4: 检查具体缺失的股票（如001390）")
    logger.info("=" * 80)
    
    test_stocks = ['001390', '000999', '001201', '001202', '001203', '001205', '001206', '001207', '001208', '001209']
    
    for test_code in test_stocks:
        with db_manager.get_session() as session:
            result = session.execute(
                select(StockDaily).where(StockDaily.code == test_code)
            ).scalars().all()
            
            if result:
                min_date = min(r.date for r in result)
                max_date = max(r.date for r in result)
                logger.info(f"股票 {test_code}: 找到 {len(result)} 条数据 ({min_date} ~ {max_date})")
            else:
                logger.warning(f"股票 {test_code}: 数据库中没有找到任何数据!")
    
    # 5. 检查数据保存时的代码格式
    logger.info("\n" + "=" * 80)
    logger.info("步骤5: 检查数据保存时的代码格式")
    logger.info("=" * 80)
    
    # 随机选几只股票检查代码格式
    sample_codes = list(db_stock_codes)[:5] if len(db_stock_codes) >=5 else list(db_stock_codes)
    
    for code in sample_codes:
        logger.info(f"数据库中的股票代码: '{code}' (长度: {len(code)})")
    
    # 6. 检查智能数据预加载器的逻辑
    logger.info("\n" + "=" * 80)
    logger.info("步骤6: 检查智能数据预加载器")
    logger.info("=" * 80)
    
    # 检查预加载器使用的股票池
    from src.core.strategy_backtest.smart_data_preloader import SmartDataPreloader
    from stock_selector.tushare_data_downloader import get_tushare_downloader
    
    tushare_downloader = get_tushare_downloader()
    preloader = SmartDataPreloader(
        db_manager=db_manager,
        tushare_downloader=tushare_downloader
    )
    
    # 检查预加载器的数据覆盖检查逻辑
    test_start_date = date(2025, 4, 1)
    test_end_date = date(2026, 4, 1)
    
    logger.info(f"测试日期范围: {test_start_date} ~ {test_end_date}")
    
    # 对几个缺失的股票检查数据覆盖情况
    for test_code in ['001390', '000001']:
        coverage = preloader._check_stock_data_coverage(test_code, test_start_date, test_end_date)
        logger.info(f"股票 {test_code} 数据覆盖: {coverage}")
    
    # 7. 总结
    logger.info("\n" + "=" * 80)
    logger.info("诊断总结")
    logger.info("=" * 80)
    
    logger.info(f"股票池总数: {len(stock_pool_set)}")
    logger.info(f"数据库中的股票数: {len(db_stock_codes)}")
    logger.info(f"缺失的股票数: {len(missing_in_db)}")
    logger.info(f"覆盖率: {len(common)/len(stock_pool_set)*100:.1f}%")
    
    if len(missing_in_db) > 0:
        logger.warning("发现数据缺失问题!")
        logger.warning("可能的原因:")
        logger.warning("  1. 智能数据预加载器没有下载这些股票的数据")
        logger.warning("  2. 股票代码格式不匹配")
        logger.warning("  3. 数据保存失败")
        logger.warning("  4. 预加载器的股票池和回测时使用的股票池不一致")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("数据库数据完整性诊断工具")
    print("=" * 80 + "\n")
    
    diagnose_database_data()
    
    print("\n" + "=" * 80)
    print("诊断完成!")
    print("=" * 80 + "\n")
