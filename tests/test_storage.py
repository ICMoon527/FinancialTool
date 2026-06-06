# -*- coding: utf-8 -*-
"""
存储模块测试
"""
import pytest
from datetime import date
from src.storage import StockDaily, StockIndicator, DatabaseManager, IntradayKline1Min


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


# =============================================================================
# save_intraday_klines 测试
# =============================================================================


@pytest.fixture
def temp_db_manager(tmp_path):
    """创建临时的 DatabaseManager 实例（用于测试 save_intraday_klines）"""
    # 重置 DatabaseManager 单例
    DatabaseManager._instance = None
    DatabaseManager._initialized = False

    db_path = tmp_path / "test_intraday.db"
    db_url = f"sqlite:///{db_path}"
    db = DatabaseManager(db_url=db_url)
    yield db
    # 清理：关闭引擎并重置单例
    if hasattr(db, "_engine"):
        db._engine.dispose()
    DatabaseManager._instance = None
    DatabaseManager._initialized = False


def test_save_intraday_klines_insert(temp_db_manager):
    """测试 save_intraday_klines 插入新记录"""
    db = temp_db_manager
    code = "600519"
    date_obj = date(2025, 6, 2)
    klines = [
        {"timestamp": "2025-06-02 09:30:00", "Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.5, "Volume": 1000, "Amount": 100500, "AvgPrice": 100.5},
        {"timestamp": "2025-06-02 09:31:00", "Open": 100.5, "High": 102.0, "Low": 100.0, "Close": 101.5, "Volume": 2000, "Amount": 203000, "AvgPrice": 101.5},
    ]

    count = db.save_intraday_klines(code, date_obj, klines)
    assert count == 2

    # 验证数据已写入
    with db.get_session() as session:
        rows = session.query(IntradayKline1Min).filter(
            IntradayKline1Min.code == code,
            IntradayKline1Min.date == date_obj,
        ).order_by(IntradayKline1Min.time).all()
        assert len(rows) == 2
        assert rows[0].time == "09:30"
        assert rows[0].close == 100.5
        assert rows[1].time == "09:31"
        assert rows[1].close == 101.5


def test_save_intraday_klines_upsert(temp_db_manager):
    """测试 save_intraday_klines UPSERT（冲突时更新）"""
    db = temp_db_manager
    code = "600519"
    date_obj = date(2025, 6, 2)

    # 第一次插入
    klines_v1 = [
        {"timestamp": "2025-06-02 09:30:00", "Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.5, "Volume": 1000, "Amount": 100500, "AvgPrice": 100.5},
    ]
    count = db.save_intraday_klines(code, date_obj, klines_v1)
    assert count == 1

    # 第二次插入相同 code/date/time，但数据不同（模拟获取到更新的数据）
    klines_v2 = [
        {"timestamp": "2025-06-02 09:30:00", "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 3000, "Amount": 303000, "AvgPrice": 101.0},
    ]
    count = db.save_intraday_klines(code, date_obj, klines_v2)
    assert count == 1  # 仍然只算一条（UPSERT，不是新增）

    # 验证数据已更新为最新值
    with db.get_session() as session:
        rows = session.query(IntradayKline1Min).filter(
            IntradayKline1Min.code == code,
            IntradayKline1Min.date == date_obj,
        ).all()
        assert len(rows) == 1  # 只有一条记录，没有重复
        assert rows[0].close == 101.0  # 使用更新后的值
        assert rows[0].volume == 3000
        assert rows[0].high == 102.0


def test_save_intraday_klines_skip_empty_timestamp(temp_db_manager):
    """测试 save_intraday_klines 跳过没有时间戳的记录"""
    db = temp_db_manager
    code = "600519"
    date_obj = date(2025, 6, 2)
    klines = [
        {"timestamp": "2025-06-02 09:30:00", "Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.5, "Volume": 1000, "Amount": 100500},
        {"timestamp": "", "Open": 101.0, "High": 102.0, "Low": 100.0, "Close": 101.5, "Volume": 2000, "Amount": 203000},  # 空时间戳，应跳过
        {"timestamp": "2025-06-02 09:32:00", "Open": 101.5, "High": 103.0, "Low": 101.0, "Close": 102.5, "Volume": 1500, "Amount": 153750},
    ]

    count = db.save_intraday_klines(code, date_obj, klines)
    assert count == 2  # 空时间戳的被跳过

    with db.get_session() as session:
        rows = session.query(IntradayKline1Min).filter(
            IntradayKline1Min.code == code,
            IntradayKline1Min.date == date_obj,
        ).order_by(IntradayKline1Min.time).all()
        assert len(rows) == 2
        assert rows[0].time == "09:30"
        assert rows[1].time == "09:32"
