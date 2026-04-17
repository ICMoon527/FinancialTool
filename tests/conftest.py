# -*- coding: utf-8 -*-
"""
pytest 配置文件
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import date, datetime
from src.storage import DatabaseManager, Base


@pytest.fixture(scope="function")
def temp_db(tmp_path):
    """
    临时数据库 Fixture
    
    为每个测试创建一个干净的数据库
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # 创建临时数据库文件
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    # 创建数据库引擎
    engine = create_engine(db_url, echo=False)
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    
    # 创建 Session 工厂
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestingSessionLocal, engine
    
    # 清理
    engine.dispose()
    if db_path.exists():
        db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_stock_data():
    """
    示例股票数据 Fixture
    """
    return [
        {
            "code": "600519",
            "date": date(2025, 1, 1),
            "open": 1500.0,
            "high": 1550.0,
            "low": 1480.0,
            "close": 1520.0,
            "volume": 1000000,
            "amount": 1520000000.0,
            "pct_chg": 2.0,
        },
        {
            "code": "600519",
            "date": date(2025, 1, 2),
            "open": 1525.0,
            "high": 1580.0,
            "low": 1510.0,
            "close": 1560.0,
            "volume": 1200000,
            "amount": 1872000000.0,
            "pct_chg": 2.63,
        },
    ]


@pytest.fixture
def sample_indicator_data():
    """
    示例指标数据 Fixture
    """
    return {
        "code": "600519",
        "date": date(2025, 1, 2),
        "indicator_type": "test_indicator",
        "data": {"ma5": 1530.0, "ma10": 1520.0, "signal": "buy"},
    }
