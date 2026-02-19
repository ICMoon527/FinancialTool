"""数据库管理模块"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from config import Config
from utils.progress_bar import create_progress_bar

try:
    from sqlalchemy import create_engine, Column, String, Float, Integer, Date
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    declarative_base = None


if HAS_SQLALCHEMY:
    Base = declarative_base()

    class StockBasic(Base):
        """股票基本信息表"""
        __tablename__ = 'stock_basic'
        
        ts_code = Column(String, primary_key=True)
        name = Column(String)
        area = Column(String)
        industry = Column(String)
        market = Column(String)
        list_date = Column(String)

    class StockDaily(Base):
        """股票日线数据表"""
        __tablename__ = 'stock_daily'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        ts_code = Column(String, index=True)
        trade_date = Column(Date, index=True)
        open = Column(Float)
        high = Column(Float)
        low = Column(Float)
        close = Column(Float)
        pre_close = Column(Float)
        change = Column(Float)
        pct_chg = Column(Float)
        vol = Column(Float)
        amount = Column(Float)
    
    class StockRecommendation(Base):
        """股票推荐表"""
        __tablename__ = 'stock_recommendation'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        ts_code = Column(String, index=True)
        name = Column(String)
        horizon = Column(String, index=True)
        score = Column(Float)
        rank = Column(Integer)
        reason = Column(String)
        target_price = Column(Float)
        stop_loss = Column(Float)
        holding_period = Column(String)
        entry_suggestion = Column(String)
        risk_control = Column(String)
        recommendation_date = Column(Date, index=True)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: str = None):
        """初始化数据库管理器"""
        if not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy is not installed")
        
        if database_url is None:
            database_url = Config.DATABASE_URL
        
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """创建数据表"""
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """获取数据库会话"""
        return self.Session()
    
    def get_stock_list(self) -> List[Dict]:
        """获取股票列表"""
        session = self.get_session()
        try:
            stocks = session.query(StockBasic).all()
            return [{'ts_code': s.ts_code, 'name': s.name} for s in stocks]
        finally:
            session.close()
    
    def load_stock_daily(self, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载单支股票的日线数据"""
        session = self.get_session()
        try:
            query = session.query(StockDaily).filter(StockDaily.ts_code == ts_code)
            
            if start_date:
                query = query.filter(StockDaily.trade_date >= start_date)
            if end_date:
                query = query.filter(StockDaily.trade_date <= end_date)
            
            records = query.order_by(StockDaily.trade_date).all()
            
            if not records:
                return pd.DataFrame()
            
            data = []
            for r in records:
                data.append({
                    'ts_code': r.ts_code,
                    'trade_date': r.trade_date,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'pre_close': r.pre_close,
                    'change': r.change,
                    'pct_chg': r.pct_chg,
                    'vol': r.vol,
                    'amount': r.amount,
                    'volume': r.vol
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_available_stocks(self) -> List[str]:
        """获取数据库中有数据的股票代码列表"""
        session = self.get_session()
        try:
            stocks = session.query(StockDaily.ts_code).distinct().all()
            return [s[0] for s in stocks]
        finally:
            session.close()
    
    def load_all_stock_data(self, stock_codes: List[str] = None, max_stocks: int = 100) -> Dict[str, pd.DataFrame]:
        """加载多支股票的数据"""
        if stock_codes is None:
            stock_codes = self.get_available_stocks()
        
        if max_stocks:
            stock_codes = stock_codes[:max_stocks]
        
        stock_data = {}
        for ts_code in stock_codes:
            df = self.load_stock_daily(ts_code)
            if not df.empty:
                stock_data[ts_code] = df
        
        return stock_data
    
    def upsert_daily_data(self, model_class, records: List[Dict], unique_keys: List[str], batch_size: int = 2000):
        """
        批量插入或更新数据 - 优化版
        
        Args:
            model_class: 数据模型类
            records: 数据记录列表
            unique_keys: 唯一键列表
            batch_size: 每批保存的记录数，默认 2000
        """
        # 首先验证 model_class
        if model_class is None:
            from logger import log
            log.error("model_class 为 None，无法保存数据")
            raise ValueError("model_class 不能为 None")
        
        # 验证 unique_keys
        if not unique_keys or len(unique_keys) == 0:
            from logger import log
            log.error("unique_keys 为空，无法保存数据")
            raise ValueError("unique_keys 不能为空")
        
        if not records:
            return
        
        valid_records = []
        invalid_count = 0
        
        # 先验证所有记录
        for i, record in enumerate(records):
            # 检查 record 是否为 None
            if record is None:
                invalid_count += 1
                continue
            
            # 检查 record 是否包含所有唯一键
            has_all_keys = True
            for key in unique_keys:
                if key not in record or record[key] is None:
                    has_all_keys = False
                    break
            
            if has_all_keys:
                valid_records.append(record)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            from logger import log
            log.warning(f"跳过 {invalid_count} 条无效记录")
        
        if not valid_records:
            return
        
        # 使用单个会话处理所有批次，这样更快
        session = self.get_session()
        
        try:
            # 分批保存
            total_batches = (len(valid_records) + batch_size - 1) // batch_size
            
            pb = None
            if total_batches > 1:
                pb = create_progress_bar(total_batches, '保存数据')
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(valid_records))
                batch_records = valid_records[start_idx:end_idx]
                
                # 先查询此批次所有可能的现有记录
                # 提取唯一键值
                key_values_list = []
                for record in batch_records:
                    key_values = {key: record[key] for key in unique_keys}
                    key_values_list.append(key_values)
                
                # 构建一个查询，一次性获取所有可能的现有记录
                # 为了简单，我们还是逐个处理，但使用同一个会话
                for record in batch_records:
                    # 构建查询条件
                    query = session.query(model_class)
                    for key in unique_keys:
                        if key in record and record[key] is not None:
                            query = query.filter(getattr(model_class, key) == record[key])
                    
                    # 查找现有记录
                    existing = query.first()
                    
                    if existing:
                        # 更新现有记录
                        for field, value in record.items():
                            if hasattr(existing, field):
                                setattr(existing, field, value)
                    else:
                        # 插入新记录
                        new_record = model_class(**record)
                        session.add(new_record)
                
                # 每批提交一次
                session.commit()
                
                # 更新进度条
                if pb:
                    pb.update(batch_idx + 1)
            
            if pb:
                pb.finish()
                
        except Exception as e:
            session.rollback()
            from logger import log
            log.error(f"保存数据失败: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise e
        finally:
            session.close()


if HAS_SQLALCHEMY:
    db_manager = DatabaseManager()
else:
    db_manager = None
