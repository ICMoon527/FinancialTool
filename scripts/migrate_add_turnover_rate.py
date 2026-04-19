# -*- coding: utf-8 -*-
"""
数据库迁移脚本：添加换手率列和股票基础信息表

1. 为 stock_daily 表添加 turnover_rate 列
2. 创建 stock_basic 表（如果不存在）
"""
import os
import sys
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text, inspect, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base

from src.storage import DatabaseManager

Base = declarative_base()


class StockBasic(Base):
    """
    股票基础信息表（仅用于迁移检查）
    """
    __tablename__ = "stock_basic"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100))
    list_date = Column(String(10))
    delist_date = Column(String(10))
    market = Column(String(20))
    industry = Column(String(50))
    list_status = Column(String(10))
    circulating_shares = Column(Float)  # 流通股本（万股）
    total_shares = Column(Float)  # 总股本（万股）
    total_market_cap = Column(Float)  # 总市值（万元）
    free_market_cap = Column(Float)  # 流通市值（万元）
    data_source = Column(String(50))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


def print_step(step_num: int, description: str):
    """打印步骤信息"""
    print(f"\n{'=' * 80}")
    print(f"步骤 {step_num}: {description}")
    print(f"{'=' * 80}")


def check_column_exists(engine, table_name: str, column_name: str) -> bool:
    """检查表中是否存在指定列"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def check_table_exists(engine, table_name: str) -> bool:
    """检查表是否存在"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def migrate_add_turnover_rate():
    """迁移：为 stock_daily 表添加 turnover_rate 列"""
    print_step(1, "检查并添加 turnover_rate 列")
    
    db = DatabaseManager.get_instance()
    
    if check_column_exists(db._engine, 'stock_daily', 'turnover_rate'):
        print("  ✓ turnover_rate 列已存在，跳过")
        return True
    
    try:
        with db.get_session() as session:
            # 添加 turnover_rate 列
            sql = text("ALTER TABLE stock_daily ADD COLUMN turnover_rate FLOAT")
            session.execute(sql)
            session.commit()
            print("  ✓ 成功添加 turnover_rate 列")
            return True
    except Exception as e:
        if "duplicate column name" in str(e).lower():
            print("  ✓ turnover_rate 列已存在（由其他会话创建）")
            return True
        logger.error(f"添加 turnover_rate 列失败: {e}")
        raise


def migrate_create_stock_basic_table():
    """迁移：创建 stock_basic 表"""
    print_step(2, "创建 stock_basic 表")
    
    db = DatabaseManager.get_instance()
    
    if check_table_exists(db._engine, 'stock_basic'):
        print("  ✓ stock_basic 表已存在，跳过")
        return True
    
    try:
        Base.metadata.create_all(db._engine, tables=[StockBasic.__table__])
        print("  ✓ 成功创建 stock_basic 表")
        return True
    except Exception as e:
        logger.error(f"创建 stock_basic 表失败: {e}")
        raise


def main():
    """主迁移函数"""
    print("\n" + "=" * 80)
    print("数据库迁移：添加换手率和股票基础信息表")
    print("=" * 80)
    
    try:
        # 步骤1：添加 turnover_rate 列
        if not migrate_add_turnover_rate():
            return
        
        # 步骤2：创建 stock_basic 表
        if not migrate_create_stock_basic_table():
            return
        
        print("\n" + "=" * 80)
        print("✓ 迁移成功完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 迁移失败: {e}")
        logger.exception("迁移过程中发生错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
