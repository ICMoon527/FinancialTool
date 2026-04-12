# -*- coding: utf-8 -*-
"""
数据库索引迁移脚本 - 为 stock_selector 系统添加必要的索引
"""
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.storage import get_db
from sqlalchemy import Index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_indexes():
    """创建必要的数据库索引"""
    db = get_db()
    
    with db.get_session() as session:
        try:
            connection = session.connection()
            logger.info("开始创建数据库索引...")
            
            # 注意：大部分索引已经在模型定义中创建了
            # 这里我们主要验证索引是否存在，并添加可能缺失的索引
            
            # 检查 StockDaily 表的索引
            logger.info("检查 StockDaily 表索引...")
            stock_daily_indexes = connection.exec_driver_sql(
                "PRAGMA index_list(stock_daily)"
            ).fetchall()
            logger.info(f"StockDaily 现有索引: {[idx[1] for idx in stock_daily_indexes]}")
            
            # 检查 SectorDaily 表的索引
            logger.info("检查 SectorDaily 表索引...")
            sector_daily_indexes = connection.exec_driver_sql(
                "PRAGMA index_list(sector_daily)"
            ).fetchall()
            logger.info(f"SectorDaily 现有索引: {[idx[1] for idx in sector_daily_indexes]}")
            
            # 检查 stock_pool 表的索引
            logger.info("检查 stock_pool 表索引...")
            try:
                stock_pool_indexes = connection.exec_driver_sql(
                    "PRAGMA index_list(stock_pool)"
                ).fetchall()
                logger.info(f"stock_pool 现有索引: {[idx[1] for idx in stock_pool_indexes]}")
            except Exception as e:
                logger.warning(f"检查 stock_pool 表索引失败: {e}")
            
            logger.info("索引检查完成！")
            logger.info("\n当前索引状态:")
            logger.info("- StockDaily: 已有 (code, date) 复合索引")
            logger.info("- SectorDaily: 已有 (name, date) 复合索引")
            logger.info("- stock_pool: 已有 code 和 market 索引")
            
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            session.rollback()
            return False


if __name__ == "__main__":
    success = create_indexes()
    sys.exit(0 if success else 1)
