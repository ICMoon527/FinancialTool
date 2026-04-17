# -*- coding: utf-8 -*-
"""
存储模块测试
"""
import pytest
from datetime import date
from src.storage import StockDaily, StockIndicator


def test_stock_daily_model():
    """测试 StockDaily 模型"""
    stock = StockDaily(
        code="600519",
        date=date(2025, 1, 1),
        open=1500.0,
        high=1550.0,
        low=1480.0,
        close=1520.0,
        volume=1000000,
        amount=1520000000.0,
        pct_chg=2.0,
    )
    
    assert stock.code == "600519"
    assert stock.date == date(2025, 1, 1)
    assert stock.close == 1520.0


def test_stock_daily_to_dict():
    """测试 StockDaily to_dict 方法"""
    stock = StockDaily(
        code="600519",
        date=date(2025, 1, 1),
        open=1500.0,
        high=1550.0,
        low=1480.0,
        close=1520.0,
        volume=1000000,
        amount=1520000000.0,
    )
    
    data = stock.to_dict()
    
    assert data["code"] == "600519"
    assert data["close"] == 1520.0
    assert "date" in data


def test_stock_indicator_model():
    """测试 StockIndicator 模型"""
    indicator = StockIndicator.create_from_dict(
        code="600519",
        date=date(2025, 1, 2),
        indicator_type="test_indicator",
        data={"ma5": 1530.0, "signal": "buy"},
    )
    
    assert indicator.code == "600519"
    assert indicator.indicator_type == "test_indicator"
    assert indicator.get_indicator_data() == {"ma5": 1530.0, "signal": "buy"}


def test_stock_daily_db_operations(temp_db, sample_stock_data):
    """测试 StockDaily 数据库操作"""
    SessionLocal, engine = temp_db
    
    with SessionLocal() as session:
        # 插入数据
        for data in sample_stock_data:
            stock = StockDaily(**data)
            session.add(stock)
        session.commit()
        
        # 查询数据
        result = session.query(StockDaily).filter(StockDaily.code == "600519").all()
        assert len(result) == 2
        assert result[0].close == 1520.0
        assert result[1].close == 1560.0


def test_stock_indicator_db_operations(temp_db, sample_indicator_data):
    """测试 StockIndicator 数据库操作"""
    SessionLocal, engine = temp_db
    
    with SessionLocal() as session:
        # 插入数据
        indicator = StockIndicator.create_from_dict(**sample_indicator_data)
        session.add(indicator)
        session.commit()
        
        # 查询数据
        result = session.query(StockIndicator).filter(
            StockIndicator.code == sample_indicator_data["code"],
            StockIndicator.indicator_type == sample_indicator_data["indicator_type"],
        ).first()
        
        assert result is not None
        assert result.get_indicator_data() == sample_indicator_data["data"]
