# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 存储层
===================================

职责：
1. 管理 SQLite 数据库连接（单例模式）
2. 定义 ORM 数据模型
3. 提供数据存取接口
4. 实现智能更新逻辑（断点续传）
"""

import atexit
from contextlib import contextmanager
import hashlib
import json
import logging
import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Tuple

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Boolean,
    Date,
    DateTime,
    Integer,
    ForeignKey,
    Index,
    UniqueConstraint,
    Text,
    select,
    and_,
    delete,
    desc,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
)
from sqlalchemy.exc import IntegrityError

from src.config import get_config

logger = logging.getLogger(__name__)

# SQLAlchemy ORM 基类
Base = declarative_base()

if TYPE_CHECKING:
    from src.search_service import SearchResponse


# === 数据模型定义 ===


class StockDaily(Base):
    """
    股票日线数据模型

    存储每日行情数据和计算的技术指标
    支持多股票、多日期的唯一约束
    """

    __tablename__ = "stock_daily"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 股票代码（如 600519, 000001）
    code = Column(String(10), nullable=False, index=True)

    # 交易日期
    date = Column(Date, nullable=False, index=True)

    # OHLC 数据
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)

    # 成交数据
    volume = Column(Float)  # 成交量（股）
    amount = Column(Float)  # 成交额（元）
    pct_chg = Column(Float)  # 涨跌幅（%）

    # 技术指标
    ma5 = Column(Float)
    ma10 = Column(Float)
    ma20 = Column(Float)
    volume_ratio = Column(Float)  # 量比

    # 数据来源
    data_source = Column(String(50))  # 记录数据来源（如 AkshareFetcher）

    # 更新时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 唯一约束：同一股票同一日期只能有一条数据
    __table_args__ = (
        UniqueConstraint("code", "date", name="uix_code_date"),
        Index("ix_code_date", "code", "date"),
    )

    def __repr__(self):
        return f"<StockDaily(code={self.code}, date={self.date}, close={self.close})>"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "pct_chg": self.pct_chg,
            "ma5": self.ma5,
            "ma10": self.ma10,
            "ma20": self.ma20,
            "volume_ratio": self.volume_ratio,
            "data_source": self.data_source,
        }


class StockIndicator(Base):
    """
    股票技术指标数据模型

    存储计算后的技术指标数据，支持多股票、多日期的唯一约束
    """

    __tablename__ = "stock_indicator"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 股票代码（如 600519, 000001）
    code = Column(String(10), nullable=False, index=True)

    # 交易日期
    date = Column(Date, nullable=False, index=True)

    # 指标类型（如 'banker_control', 'main_capital_absorption' 等）
    indicator_type = Column(String(50), nullable=False, index=True)

    # 指标数据（JSON 格式存储）
    indicator_data = Column(Text, nullable=False)

    # 更新时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 唯一约束：同一股票同一日期同一指标只能有一条数据
    __table_args__ = (
        UniqueConstraint("code", "date", "indicator_type", name="uix_code_date_indicator"),
        Index("ix_code_date_indicator", "code", "date", "indicator_type"),
    )

    def __repr__(self):
        return f"<StockIndicator(code={self.code}, date={self.date}, type={self.indicator_type})>"

    def get_indicator_data(self) -> Dict[str, Any]:
        """获取指标数据字典"""
        try:
            return json.loads(self.indicator_data)
        except (json.JSONDecodeError, TypeError):
            return {}

    @classmethod
    def create_from_dict(cls, code: str, date: date, indicator_type: str, data: Dict[str, Any]) -> 'StockIndicator':
        """从字典创建实例"""
        return cls(
            code=code,
            date=date,
            indicator_type=indicator_type,
            indicator_data=json.dumps(data, ensure_ascii=False),
        )


class VisualizationSearchHistory(Base):
    """
    可视化搜索历史记录

    专门用于可视化页面的搜索历史记录
    """

    __tablename__ = "visualization_search_history"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 股票代码
    stock_code = Column(String(10), nullable=False, index=True)

    # 股票名称
    stock_name = Column(String(50))

    # 搜索时间
    searched_at = Column(DateTime, default=datetime.now, index=True)

    # 选中的指标列表（JSON 格式）
    selected_indicators = Column(Text)

    def __repr__(self):
        return f"<VisualizationSearchHistory(code={self.stock_code}, time={self.searched_at})>"

    def get_selected_indicators(self) -> List[str]:
        """获取选中的指标列表"""
        try:
            return json.loads(self.selected_indicators) if self.selected_indicators else []
        except (json.JSONDecodeError, TypeError):
            return []

    @classmethod
    def create(cls, stock_code: str, stock_name: Optional[str] = None, selected_indicators: Optional[List[str]] = None) -> 'VisualizationSearchHistory':
        """创建搜索历史记录"""
        return cls(
            stock_code=stock_code,
            stock_name=stock_name,
            selected_indicators=json.dumps(selected_indicators or [], ensure_ascii=False) if selected_indicators else None,
        )


class StockSector(Base):
    """
    股票板块信息数据模型

    存储股票所属板块信息，避免重复联网获取
    """

    __tablename__ = "stock_sector"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 股票代码（如 600519, 000001）
    code = Column(String(10), nullable=False, index=True)

    # 板块名称（JSON 格式存储列表）
    sectors = Column(Text, nullable=False)

    # 数据来源
    data_source = Column(String(50))

    # 更新时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 唯一约束：同一股票只能有一条板块记录
    __table_args__ = (
        UniqueConstraint("code", name="uix_stock_sector_code"),
    )

    def __repr__(self):
        return f"<StockSector(code={self.code}, sectors={self.sectors})>"

    def get_sectors(self) -> List[str]:
        """获取板块列表"""
        try:
            return json.loads(self.sectors)
        except (json.JSONDecodeError, TypeError):
            return []

    @classmethod
    def create_from_list(cls, code: str, sectors: List[str], data_source: str) -> 'StockSector':
        """从板块列表创建实例"""
        return cls(
            code=code,
            sectors=json.dumps(sectors, ensure_ascii=False),
            data_source=data_source
        )


class SectorDaily(Base):
    """
    板块每日数据模型

    存储板块每日涨跌幅等数据，用于历史回测
    """

    __tablename__ = "sector_daily"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 板块名称
    name = Column(String(100), nullable=False, index=True)

    # 交易日期
    date = Column(Date, nullable=False, index=True)

    # 涨跌幅 (%)
    change_pct = Column(Float)

    # 板块内股票数量
    stock_count = Column(Integer, default=0)

    # 涨停股票数量
    limit_up_count = Column(Integer, default=0)

    # 数据来源
    data_source = Column(String(50))

    # 更新时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 唯一约束：同一板块同一日期只能有一条数据
    __table_args__ = (
        UniqueConstraint("name", "date", name="uix_sector_date"),
        Index("ix_sector_date", "name", "date"),
    )

    def __repr__(self):
        return f"<SectorDaily(name={self.name}, date={self.date}, change_pct={self.change_pct})>"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "date": self.date,
            "change_pct": self.change_pct,
            "stock_count": self.stock_count,
            "limit_up_count": self.limit_up_count,
            "data_source": self.data_source,
        }


class NewsIntel(Base):
    """
    新闻情报数据模型

    存储搜索到的新闻情报条目，用于后续分析与查询
    """

    __tablename__ = "news_intel"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联用户查询操作
    query_id = Column(String(64), index=True)

    # 股票信息
    code = Column(String(10), nullable=False, index=True)
    name = Column(String(50))

    # 搜索上下文
    dimension = Column(String(32), index=True)  # latest_news / risk_check / earnings / market_analysis / industry
    query = Column(String(255))
    provider = Column(String(32), index=True)

    # 新闻内容
    title = Column(String(300), nullable=False)
    snippet = Column(Text)
    url = Column(String(1000), nullable=False)
    source = Column(String(100))
    published_date = Column(DateTime, index=True)

    # 入库时间
    fetched_at = Column(DateTime, default=datetime.now, index=True)
    query_source = Column(String(32), index=True)  # bot/web/cli/system
    requester_platform = Column(String(20))
    requester_user_id = Column(String(64))
    requester_user_name = Column(String(64))
    requester_chat_id = Column(String(64))
    requester_message_id = Column(String(64))
    requester_query = Column(String(255))

    __table_args__ = (
        UniqueConstraint("url", name="uix_news_url"),
        Index("ix_news_code_pub", "code", "published_date"),
    )

    def __repr__(self) -> str:
        return f"<NewsIntel(code={self.code}, title={self.title[:20]}...)>"


class AnalysisHistory(Base):
    """
    分析结果历史记录模型

    保存每次分析结果，支持按 query_id/股票代码检索
    """

    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联查询链路
    query_id = Column(String(64), index=True)

    # 股票信息
    code = Column(String(10), nullable=False, index=True)
    name = Column(String(50))
    report_type = Column(String(16), index=True)

    # 核心结论
    sentiment_score = Column(Integer)
    operation_advice = Column(String(20))
    trend_prediction = Column(String(50))
    analysis_summary = Column(Text)

    # 详细数据
    raw_result = Column(Text)
    news_content = Column(Text)
    context_snapshot = Column(Text)

    # 狙击点位（用于回测）
    ideal_buy = Column(Float)
    secondary_buy = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)

    created_at = Column(DateTime, default=datetime.now, index=True)

    __table_args__ = (Index("ix_analysis_code_time", "code", "created_at"),)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "code": self.code,
            "name": self.name,
            "report_type": self.report_type,
            "sentiment_score": self.sentiment_score,
            "operation_advice": self.operation_advice,
            "trend_prediction": self.trend_prediction,
            "analysis_summary": self.analysis_summary,
            "raw_result": self.raw_result,
            "news_content": self.news_content,
            "context_snapshot": self.context_snapshot,
            "ideal_buy": self.ideal_buy,
            "secondary_buy": self.secondary_buy,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestResult(Base):
    """单条分析记录的回测结果。"""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analysis_history_id = Column(
        Integer,
        ForeignKey("analysis_history.id"),
        nullable=False,
        index=True,
    )

    # 冗余字段，便于按股票筛选
    code = Column(String(10), nullable=False, index=True)
    analysis_date = Column(Date, index=True)

    # 回测参数
    eval_window_days = Column(Integer, nullable=False, default=10)
    engine_version = Column(String(16), nullable=False, default="v1")

    # 状态
    eval_status = Column(String(16), nullable=False, default="pending")
    evaluated_at = Column(DateTime, default=datetime.now, index=True)

    # 建议快照（避免未来分析字段变化导致回测不可解释）
    operation_advice = Column(String(20))
    position_recommendation = Column(String(8))  # long/cash

    # 价格与收益
    start_price = Column(Float)
    end_close = Column(Float)
    max_high = Column(Float)
    min_low = Column(Float)
    stock_return_pct = Column(Float)

    # 方向与结果
    direction_expected = Column(String(16))  # up/down/flat/not_down
    direction_correct = Column(Boolean, nullable=True)
    outcome = Column(String(16))  # win/loss/neutral

    # 目标价命中（仅 long 且配置了止盈/止损时有意义）
    stop_loss = Column(Float)
    take_profit = Column(Float)
    hit_stop_loss = Column(Boolean)
    hit_take_profit = Column(Boolean)
    first_hit = Column(String(16))  # take_profit/stop_loss/ambiguous/neither/not_applicable
    first_hit_date = Column(Date)
    first_hit_trading_days = Column(Integer)

    # 模拟执行（long-only）
    simulated_entry_price = Column(Float)
    simulated_exit_price = Column(Float)
    simulated_exit_reason = Column(String(24))  # stop_loss/take_profit/window_end/cash/ambiguous_stop_loss
    simulated_return_pct = Column(Float)

    __table_args__ = (
        UniqueConstraint(
            "analysis_history_id",
            "eval_window_days",
            "engine_version",
            name="uix_backtest_analysis_window_version",
        ),
        Index("ix_backtest_code_date", "code", "analysis_date"),
    )


class BacktestSummary(Base):
    """回测汇总指标（按股票或全局）。"""

    __tablename__ = "backtest_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)

    scope = Column(String(16), nullable=False, index=True)  # overall/stock
    code = Column(String(16), index=True)

    eval_window_days = Column(Integer, nullable=False, default=10)
    engine_version = Column(String(16), nullable=False, default="v1")
    computed_at = Column(DateTime, default=datetime.now, index=True)

    # 计数
    total_evaluations = Column(Integer, default=0)
    completed_count = Column(Integer, default=0)
    insufficient_count = Column(Integer, default=0)
    long_count = Column(Integer, default=0)
    cash_count = Column(Integer, default=0)

    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)

    # 准确率/胜率
    direction_accuracy_pct = Column(Float)
    win_rate_pct = Column(Float)
    neutral_rate_pct = Column(Float)

    # 收益
    avg_stock_return_pct = Column(Float)
    avg_simulated_return_pct = Column(Float)

    # 目标价触发统计（仅 long 且配置止盈/止损时统计）
    stop_loss_trigger_rate = Column(Float)
    take_profit_trigger_rate = Column(Float)
    ambiguous_rate = Column(Float)
    avg_days_to_first_hit = Column(Float)

    # 诊断字段（JSON 字符串）
    advice_breakdown_json = Column(Text)
    diagnostics_json = Column(Text)

    __table_args__ = (
        UniqueConstraint(
            "scope",
            "code",
            "eval_window_days",
            "engine_version",
            name="uix_backtest_summary_scope_code_window_version",
        ),
    )


class ConversationMessage(Base):
    """
    Agent 对话历史记录表
    """

    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now, index=True)


class DatabaseManager:
    """
    数据库管理器 - 单例模式

    职责：
    1. 管理数据库连接池
    2. 提供 Session 上下文管理
    3. 封装数据存取操作
    """

    _instance: Optional["DatabaseManager"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_url: Optional[str] = None):
        """
        初始化数据库管理器

        Args:
            db_url: 数据库连接 URL（可选，默认从配置读取）
        """
        if getattr(self, "_initialized", False):
            return

        if db_url is None:
            config = get_config()
            db_url = config.get_db_url()

        # 创建数据库引擎
        self._engine = create_engine(
            db_url,
            echo=False,  # 设为 True 可查看 SQL 语句
            pool_pre_ping=True,  # 连接健康检查
            pool_size=20,  # 连接池大小，匹配线程池数量
            max_overflow=10,  # 最大溢出连接数
            pool_recycle=3600,  # 连接回收时间（秒），避免长时间连接
        )

        # 创建 Session 工厂
        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

        # 创建所有表
        Base.metadata.create_all(self._engine)

        # 初始化 LRU 缓存（用于性能优化）
        from collections import OrderedDict
        self._data_cache = OrderedDict()
        self.CACHE_MAX_SIZE = 1000  # 最大缓存条目数
        self.CACHE_TTL_SECONDS = 300  # 缓存 TTL: 5 分钟

        self._initialized = True
        logger.info(f"数据库初始化完成: {db_url}")

        # 注册退出钩子，确保程序退出时关闭数据库连接
        atexit.register(DatabaseManager._cleanup_engine, self._engine)

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（用于测试）"""
        if cls._instance is not None:
            if hasattr(cls._instance, "_engine") and cls._instance._engine is not None:
                cls._instance._engine.dispose()
            cls._instance._initialized = False
            cls._instance = None

    @classmethod
    def _cleanup_engine(cls, engine) -> None:
        """
        清理数据库引擎（atexit 钩子）

        确保程序退出时关闭所有数据库连接，避免 ResourceWarning

        Args:
            engine: SQLAlchemy 引擎对象
        """
        try:
            if engine is not None:
                engine.dispose()
                logger.debug("数据库引擎已清理")
        except Exception as e:
            logger.warning(f"清理数据库引擎时出错: {e}")

    def get_session(self) -> Session:
        """
        获取数据库 Session

        使用示例:
            with db.get_session() as session:
                # 执行查询
                session.commit()  # 如果需要
        """
        if not getattr(self, "_initialized", False) or not hasattr(self, "_SessionLocal"):
            raise RuntimeError("DatabaseManager 未正确初始化。" "请确保通过 DatabaseManager.get_instance() 获取实例。")
        session = self._SessionLocal()
        try:
            return session
        except Exception:
            session.close()
            raise

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def has_today_data(self, code: str, target_date: Optional[date] = None) -> bool:
        """
        检查是否已有指定日期的数据
        
        用于断点续传逻辑：如果已有数据则跳过网络请求
        
        Args:
            code: 股票代码
            target_date: 目标日期（默认今天）
            
        Returns:
            是否存在数据
        """
        if target_date is None:
            target_date = date.today()
        
        # 检查指定日期是否有数据
        with self.get_session() as session:
            result = session.execute(
                select(StockDaily).where(
                    and_(
                        StockDaily.code == code,
                        StockDaily.date == target_date
                    )
                )
            ).scalar_one_or_none()
            
            if result is not None:
                return True
            
            # 检查是否有最近的数据
            latest_data = session.execute(
                select(StockDaily)
                .where(StockDaily.code == code)
                .order_by(desc(StockDaily.date))
                .limit(1)
            ).scalar_one_or_none()
            
            if not latest_data:
                return False
            
            # 计算最近数据与今天的天数差
            days_diff = (target_date - latest_data.date).days
            
            # 如果最近数据是今天或昨天的，直接使用
            if days_diff <= 1:
                logger.debug(f"{code} 今天 {target_date} 数据未更新，使用最近数据 {latest_data.date}")
                return True
            
            # 检查是否是非交易日
            from src.core.trading_calendar import get_market_for_stock, is_market_open
            market = get_market_for_stock(code)
            
            # 检查是否是非交易日
            is_non_trading_day = False
            
            # 首先检查是否是周末
            if target_date.weekday() >= 5:  # 0-4 是工作日，5-6 是周末
                is_non_trading_day = True
                logger.debug(f"{code} 在周末 {target_date}")
            
            # 然后检查是否是节假日（如果有 exchange-calendars 包）
            elif market:
                is_open = is_market_open(market, target_date)
                if not is_open:
                    is_non_trading_day = True
                    logger.debug(f"{code} 在非交易日 {target_date}")
            
            if is_non_trading_day:
                # 非交易日，只要有最近的数据就返回 True
                logger.debug(f"{code} 在非交易日 {target_date}，使用最近数据 {latest_data.date}")
                return True
        
        return False

    def get_latest_data(self, code: str, days: int = 2) -> List[StockDaily]:
        """
        获取最近 N 天的数据

        用于计算"相比昨日"的变化

        Args:
            code: 股票代码
            days: 获取天数

        Returns:
            StockDaily 对象列表（按日期降序）
        """
        with self.get_session() as session:
            results = (
                session.execute(
                    select(StockDaily).where(StockDaily.code == code).order_by(desc(StockDaily.date)).limit(days)
                )
                .scalars()
                .all()
            )

            return list(results)

    def save_news_intel(
        self,
        code: str,
        name: str,
        dimension: str,
        query: str,
        response: "SearchResponse",
        query_context: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        保存新闻情报到数据库

        去重策略：
        - 优先按 URL 去重（唯一约束）
        - URL 缺失时按 title + source + published_date 进行软去重

        关联策略：
        - query_context 记录用户查询信息（平台、用户、会话、原始指令等）
        """
        if not response or not response.results:
            return 0

        saved_count = 0
        query_ctx = query_context or {}
        current_query_id = (query_ctx.get("query_id") or "").strip()

        with self.get_session() as session:
            try:
                for item in response.results:
                    title = (item.title or "").strip()
                    url = (item.url or "").strip()
                    source = (item.source or "").strip()
                    snippet = (item.snippet or "").strip()
                    published_date = self._parse_published_date(item.published_date)

                    if not title and not url:
                        continue

                    url_key = url or self._build_fallback_url_key(
                        code=code, title=title, source=source, published_date=published_date
                    )

                    # 优先按 URL 或兜底键去重
                    existing = session.execute(select(NewsIntel).where(NewsIntel.url == url_key)).scalar_one_or_none()

                    if existing:
                        existing.name = name or existing.name
                        existing.dimension = dimension or existing.dimension
                        existing.query = query or existing.query
                        existing.provider = response.provider or existing.provider
                        existing.snippet = snippet or existing.snippet
                        existing.source = source or existing.source
                        existing.published_date = published_date or existing.published_date
                        existing.fetched_at = datetime.now()

                        if query_context:
                            # Keep the first query_id to avoid overwriting historical links.
                            if not existing.query_id and current_query_id:
                                existing.query_id = current_query_id
                            existing.query_source = query_context.get("query_source") or existing.query_source
                            existing.requester_platform = (
                                query_context.get("requester_platform") or existing.requester_platform
                            )
                            existing.requester_user_id = (
                                query_context.get("requester_user_id") or existing.requester_user_id
                            )
                            existing.requester_user_name = (
                                query_context.get("requester_user_name") or existing.requester_user_name
                            )
                            existing.requester_chat_id = (
                                query_context.get("requester_chat_id") or existing.requester_chat_id
                            )
                            existing.requester_message_id = (
                                query_context.get("requester_message_id") or existing.requester_message_id
                            )
                            existing.requester_query = query_context.get("requester_query") or existing.requester_query
                    else:
                        try:
                            with session.begin_nested():
                                record = NewsIntel(
                                    code=code,
                                    name=name,
                                    dimension=dimension,
                                    query=query,
                                    provider=response.provider,
                                    title=title,
                                    snippet=snippet,
                                    url=url_key,
                                    source=source,
                                    published_date=published_date,
                                    fetched_at=datetime.now(),
                                    query_id=current_query_id or None,
                                    query_source=query_ctx.get("query_source"),
                                    requester_platform=query_ctx.get("requester_platform"),
                                    requester_user_id=query_ctx.get("requester_user_id"),
                                    requester_user_name=query_ctx.get("requester_user_name"),
                                    requester_chat_id=query_ctx.get("requester_chat_id"),
                                    requester_message_id=query_ctx.get("requester_message_id"),
                                    requester_query=query_ctx.get("requester_query"),
                                )
                                session.add(record)
                                session.flush()
                            saved_count += 1
                        except IntegrityError:
                            # 单条 URL 唯一约束冲突（如并发插入），仅跳过本条，保留本批其余成功项
                            logger.debug("新闻情报重复（已跳过）: %s %s", code, url_key)

                session.commit()
                logger.info(f"保存新闻情报成功: {code}, 新增 {saved_count} 条")

            except Exception as e:
                session.rollback()
                logger.error(f"保存新闻情报失败: {e}")
                raise

        return saved_count

    def get_recent_news(self, code: str, days: int = 7, limit: int = 20) -> List[NewsIntel]:
        """
        获取指定股票最近 N 天的新闻情报
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.get_session() as session:
            results = (
                session.execute(
                    select(NewsIntel)
                    .where(and_(NewsIntel.code == code, NewsIntel.fetched_at >= cutoff_date))
                    .order_by(desc(NewsIntel.fetched_at))
                    .limit(limit)
                )
                .scalars()
                .all()
            )

            return list(results)

    def get_news_intel_by_query_id(self, query_id: str, limit: int = 20) -> List[NewsIntel]:
        """
        根据 query_id 获取新闻情报列表

        Args:
            query_id: 分析记录唯一标识
            limit: 返回数量限制

        Returns:
            NewsIntel 列表（按发布时间或抓取时间倒序）
        """
        from sqlalchemy import func

        with self.get_session() as session:
            results = (
                session.execute(
                    select(NewsIntel)
                    .where(NewsIntel.query_id == query_id)
                    .order_by(
                        desc(func.coalesce(NewsIntel.published_date, NewsIntel.fetched_at)), desc(NewsIntel.fetched_at)
                    )
                    .limit(limit)
                )
                .scalars()
                .all()
            )

            return list(results)

    def save_analysis_history(
        self,
        result: Any,
        query_id: str,
        report_type: str,
        news_content: Optional[str],
        context_snapshot: Optional[Dict[str, Any]] = None,
        save_snapshot: bool = True,
    ) -> int:
        """
        保存分析结果历史记录
        """
        if result is None:
            return 0

        sniper_points = self._extract_sniper_points(result)
        raw_result = self._build_raw_result(result)
        context_text = None
        if save_snapshot and context_snapshot is not None:
            context_text = self._safe_json_dumps(context_snapshot)

        record = AnalysisHistory(
            query_id=query_id,
            code=result.code,
            name=result.name,
            report_type=report_type,
            sentiment_score=result.sentiment_score,
            operation_advice=result.operation_advice,
            trend_prediction=result.trend_prediction,
            analysis_summary=result.analysis_summary,
            raw_result=self._safe_json_dumps(raw_result),
            news_content=news_content,
            context_snapshot=context_text,
            ideal_buy=sniper_points.get("ideal_buy"),
            secondary_buy=sniper_points.get("secondary_buy"),
            stop_loss=sniper_points.get("stop_loss"),
            take_profit=sniper_points.get("take_profit"),
            created_at=datetime.now(),
        )

        with self.get_session() as session:
            try:
                session.add(record)
                session.commit()
                return 1
            except Exception as e:
                session.rollback()
                logger.error(f"保存分析历史失败: {e}")
                return 0

    def get_analysis_history(
        self, code: Optional[str] = None, query_id: Optional[str] = None, days: int = 30, limit: int = 50
    ) -> List[AnalysisHistory]:
        """
        Query analysis history records.

        Notes:
        - If query_id is provided, perform exact lookup and ignore days window.
        - If query_id is not provided, apply days-based time filtering.
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.get_session() as session:
            conditions = []

            if query_id:
                conditions.append(AnalysisHistory.query_id == query_id)
            else:
                conditions.append(AnalysisHistory.created_at >= cutoff_date)

            if code:
                conditions.append(AnalysisHistory.code == code)

            results = (
                session.execute(
                    select(AnalysisHistory)
                    .where(and_(*conditions))
                    .order_by(desc(AnalysisHistory.created_at))
                    .limit(limit)
                )
                .scalars()
                .all()
            )

            return list(results)

    def get_analysis_history_paginated(
        self,
        code: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> Tuple[List[AnalysisHistory], int]:
        """
        分页查询分析历史记录（带总数）

        Args:
            code: 股票代码筛选
            start_date: 开始日期（含）
            end_date: 结束日期（含）
            offset: 偏移量（跳过前 N 条）
            limit: 每页数量

        Returns:
            Tuple[List[AnalysisHistory], int]: (记录列表, 总数)
        """
        from sqlalchemy import func

        with self.get_session() as session:
            conditions = []

            if code:
                conditions.append(AnalysisHistory.code == code)
            if start_date:
                # created_at >= start_date 00:00:00
                conditions.append(AnalysisHistory.created_at >= datetime.combine(start_date, datetime.min.time()))
            if end_date:
                # created_at < end_date+1 00:00:00 (即 <= end_date 23:59:59)
                conditions.append(
                    AnalysisHistory.created_at < datetime.combine(end_date + timedelta(days=1), datetime.min.time())
                )

            # 构建 where 子句
            where_clause = and_(*conditions) if conditions else True

            # 查询总数
            total_query = select(func.count(AnalysisHistory.id)).where(where_clause)
            total = session.execute(total_query).scalar() or 0

            # 查询分页数据
            data_query = (
                select(AnalysisHistory)
                .where(where_clause)
                .order_by(desc(AnalysisHistory.created_at))
                .offset(offset)
                .limit(limit)
            )
            results = session.execute(data_query).scalars().all()

            return list(results), total

    def get_analysis_history_by_id(self, record_id: int) -> Optional[AnalysisHistory]:
        """
        根据数据库主键 ID 查询单条分析历史记录

        由于 query_id 可能重复（批量分析时多条记录共享同一 query_id），
        使用主键 ID 确保精确查询唯一记录。

        Args:
            record_id: 分析历史记录的主键 ID

        Returns:
            AnalysisHistory 对象，不存在返回 None
        """
        with self.get_session() as session:
            result = session.execute(select(AnalysisHistory).where(AnalysisHistory.id == record_id)).scalars().first()
            return result

    def get_latest_analysis_by_query_id(self, query_id: str) -> Optional[AnalysisHistory]:
        """
        根据 query_id 查询最新一条分析历史记录

        query_id 在批量分析时可能重复，故返回最近创建的一条。

        Args:
            query_id: 分析记录关联的 query_id

        Returns:
            AnalysisHistory 对象，不存在返回 None
        """
        with self.get_session() as session:
            result = (
                session.execute(
                    select(AnalysisHistory)
                    .where(AnalysisHistory.query_id == query_id)
                    .order_by(desc(AnalysisHistory.created_at))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            return result

    def get_data_range(self, code: str, start_date: date, end_date: date) -> List[StockDaily]:
        """
        获取指定日期范围的数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            StockDaily 对象列表
        """
        with self.get_session() as session:
            results = (
                session.execute(
                    select(StockDaily)
                    .where(and_(StockDaily.code == code, StockDaily.date >= start_date, StockDaily.date <= end_date))
                    .order_by(StockDaily.date)
                )
                .scalars()
                .all()
            )

            return list(results)

    def get_data_range_optimized(
        self, 
        code: str, 
        start_date: date, 
        end_date: date,
        use_cache: bool = True
    ) -> List[StockDaily]:
        """
        优化的数据范围查询 - 带 LRU 缓存

        优化点：
        1. LRU 缓存，避免重复查询
        2. 批量加载
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            StockDaily 对象列表
        """
        cache_key = (code, start_date, end_date)
        
        if use_cache and cache_key in self._data_cache:
            cached_data, timestamp = self._data_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < self.CACHE_TTL_SECONDS:
                logger.debug(f"缓存命中：{code} ({age:.1f}s old)")
                self._data_cache.move_to_end(cache_key)
                return cached_data
        
        with self.get_session() as session:
            results = (
                session.execute(
                    select(StockDaily)
                    .where(and_(StockDaily.code == code, StockDaily.date >= start_date, StockDaily.date <= end_date))
                    .order_by(StockDaily.date)
                )
                .scalars()
                .all()
            )
            
            data = list(results)
            
            if use_cache:
                self._update_cache(cache_key, data)
            
            return data

    def get_data_range_with_fields(
        self, 
        code: str, 
        start_date: date, 
        end_date: date,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        优化的数据范围查询 - 只检索指定字段
        
        优化点：
        1. 只选择需要的字段，减少数据传输量
        2. 直接返回 DataFrame，方便后续处理
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要检索的字段列表，默认使用核心字段
            
        Returns:
            pandas DataFrame
        """
        if fields is None:
            fields = [
                'date', 'open', 'high', 'low', 'close', 
                'volume', 'pct_chg'
            ]
        
        with self.get_session() as session:
            columns = [getattr(StockDaily, field) for field in fields if hasattr(StockDaily, field)]
            
            results = (
                session.execute(
                    select(*columns)
                    .where(and_(StockDaily.code == code, StockDaily.date >= start_date, StockDaily.date <= end_date))
                    .order_by(StockDaily.date)
                )
            )
            
            df = pd.DataFrame(results.fetchall(), columns=fields)
            
            return df

    def save_daily_data_bulk(
        self, 
        df: pd.DataFrame, 
        code: str, 
        data_source: str = "Unknown"
    ) -> int:
        """
        批量保存日线数据（优化版本）
        
        优化点：
        1. 批量查询已存在的记录（解决 N+1 问题）
        2. 批量插入新记录
        3. 批量更新现有记录
        4. 减少数据库交互次数
        
        Args:
            df: 包含日线数据的 DataFrame
            code: 股票代码
            data_source: 数据来源名称
            
        Returns:
            新增/更新的记录数
        """
        if df is None or df.empty:
            logger.warning(f"保存数据为空，跳过 {code}")
            return 0
        
        with self.get_session() as session:
            try:
                parsed_dates = []
                for _, row in df.iterrows():
                    row_date = self._parse_date(row.get("date"))
                    if row_date:
                        parsed_dates.append((row_date, row))
                
                if not parsed_dates:
                    logger.warning(f"没有有效的日期数据，跳过 {code}")
                    return 0
                
                date_set = {d for d, _ in parsed_dates}
                
                existing_records = (
                    session.execute(
                        select(StockDaily)
                        .where(and_(StockDaily.code == code, StockDaily.date.in_(date_set)))
                    )
                    .scalars()
                    .all()
                )
                
                existing_map = {rec.date: rec for rec in existing_records}
                
                records_to_add = []
                records_to_update = []
                
                for row_date, row in parsed_dates:
                    if row_date in existing_map:
                        records_to_update.append((row_date, row))
                    else:
                        records_to_add.append((row_date, row))
                
                if records_to_add:
                    for row_date, row in records_to_add:
                        record = StockDaily(
                            code=code,
                            date=row_date,
                            open=row.get("open"),
                            high=row.get("high"),
                            low=row.get("low"),
                            close=row.get("close"),
                            volume=row.get("volume"),
                            amount=row.get("amount"),
                            pct_chg=row.get("pct_chg"),
                            ma5=row.get("ma5"),
                            ma10=row.get("ma10"),
                            ma20=row.get("ma20"),
                            volume_ratio=row.get("volume_ratio"),
                            data_source=data_source,
                        )
                        session.add(record)
                    
                    logger.info(f"批量插入 {code} 数据：{len(records_to_add)} 条")
                
                if records_to_update:
                    for row_date, row in records_to_update:
                        existing_record = existing_map[row_date]
                        existing_record.open = row.get("open")
                        existing_record.high = row.get("high")
                        existing_record.low = row.get("low")
                        existing_record.close = row.get("close")
                        existing_record.volume = row.get("volume")
                        existing_record.amount = row.get("amount")
                        existing_record.pct_chg = row.get("pct_chg")
                        existing_record.ma5 = row.get("ma5")
                        existing_record.ma10 = row.get("ma10")
                        existing_record.ma20 = row.get("ma20")
                        existing_record.volume_ratio = row.get("volume_ratio")
                        existing_record.data_source = data_source
                        existing_record.updated_at = datetime.now()
                    
                    logger.info(f"批量更新 {code} 数据：{len(records_to_update)} 条")
                
                session.commit()
                
                total_count = len(records_to_add) + len(records_to_update)
                logger.info(f"保存 {code} 数据成功，共 {total_count} 条（新增 {len(records_to_add)}, 更新 {len(records_to_update)}）")
                
                self._invalidate_cache(code)
                
                return total_count
                
            except Exception as e:
                session.rollback()
                logger.error(f"保存 {code} 数据失败：{e}")
                raise

    def _parse_date(self, date_value) -> Optional[date]:
        """
        解析日期值为 date 对象
        
        Args:
            date_value: 可以是字符串、datetime、pd.Timestamp 等
            
        Returns:
            date 对象或 None
        """
        if date_value is None:
            return None
        
        if isinstance(date_value, str):
            try:
                return datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                return None
        elif isinstance(date_value, datetime):
            return date_value.date()
        elif isinstance(date_value, pd.Timestamp):
            return date_value.date()
        elif isinstance(date_value, date):
            return date_value
        
        return None

    def _update_cache(self, key: Tuple, data: List[StockDaily]) -> None:
        """
        更新 LRU 缓存
        
        Args:
            key: 缓存键
            data: 数据
        """
        if len(self._data_cache) >= self.CACHE_MAX_SIZE:
            self._data_cache.popitem(last=False)
        
        self._data_cache[key] = (data, datetime.now())

    def _invalidate_cache(self, code: str) -> None:
        """
        清除指定股票的所有缓存
        
        Args:
            code: 股票代码
        """
        keys_to_remove = [
            key for key in self._data_cache.keys() 
            if key[0] == code
        ]
        
        for key in keys_to_remove:
            del self._data_cache[key]
        
        logger.debug(f"清除 {code} 的缓存，共 {len(keys_to_remove)} 条")

    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._data_cache.clear()
        logger.info("数据缓存已清空")

    def save_daily_data(self, df: pd.DataFrame, code: str, data_source: str = "Unknown") -> int:
        """
        保存日线数据到数据库

        策略：
        - 使用 UPSERT 逻辑（存在则更新，不存在则插入）
        - 跳过已存在的数据，避免重复

        Args:
            df: 包含日线数据的 DataFrame
            code: 股票代码
            data_source: 数据来源名称

        Returns:
            新增的记录数
        """
        if df is None or df.empty:
            logger.warning(f"保存数据为空，跳过 {code}")
            return 0

        saved_count = 0
        updated_count = 0

        with self.get_session() as session:
            try:
                for _, row in df.iterrows():
                    # 解析日期
                    row_date = row.get("date")
                    if isinstance(row_date, str):
                        row_date = datetime.strptime(row_date, "%Y-%m-%d").date()
                    elif isinstance(row_date, datetime):
                        row_date = row_date.date()
                    elif isinstance(row_date, pd.Timestamp):
                        row_date = row_date.date()

                    # 检查是否已存在
                    existing = session.execute(
                        select(StockDaily).where(and_(StockDaily.code == code, StockDaily.date == row_date))
                    ).scalar_one_or_none()

                    if existing:
                        # 更新现有记录
                        existing.open = row.get("open")
                        existing.high = row.get("high")
                        existing.low = row.get("low")
                        existing.close = row.get("close")
                        existing.volume = row.get("volume")
                        existing.amount = row.get("amount")
                        existing.pct_chg = row.get("pct_chg")
                        existing.ma5 = row.get("ma5")
                        existing.ma10 = row.get("ma10")
                        existing.ma20 = row.get("ma20")
                        existing.volume_ratio = row.get("volume_ratio")
                        existing.data_source = data_source
                        existing.updated_at = datetime.now()
                        updated_count += 1
                    else:
                        # 创建新记录
                        record = StockDaily(
                            code=code,
                            date=row_date,
                            open=row.get("open"),
                            high=row.get("high"),
                            low=row.get("low"),
                            close=row.get("close"),
                            volume=row.get("volume"),
                            amount=row.get("amount"),
                            pct_chg=row.get("pct_chg"),
                            ma5=row.get("ma5"),
                            ma10=row.get("ma10"),
                            ma20=row.get("ma20"),
                            volume_ratio=row.get("volume_ratio"),
                            data_source=data_source,
                        )
                        session.add(record)
                        saved_count += 1

                session.commit()
                logger.info(f"保存 {code} 数据成功，新增 {saved_count} 条，更新 {updated_count} 条")

            except Exception as e:
                session.rollback()
                logger.error(f"保存 {code} 数据失败: {e}")
                raise

        return saved_count

    def check_data_coverage(
        self,
        stock_codes: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Tuple[Optional[date], Optional[date]]]:
        """
        批量检查多只股票的数据覆盖范围
        
        Args:
            stock_codes: 股票代码列表
            start_date: 可选，目标开始日期
            end_date: 可选，目标结束日期
            
        Returns:
            {stock_code: (min_date, max_date)}
            - min_date: 数据库中该股票的最早数据日期
            - max_date: 数据库中该股票的最晚数据日期
            - 没有数据时返回 (None, None)
        """
        from sqlalchemy import func
        
        result = {code: (None, None) for code in stock_codes}
        
        if not stock_codes:
            return result
        
        with self.get_session() as session:
            # 使用 GROUP BY 和聚合函数批量查询
            query = (
                select(
                    StockDaily.code,
                    func.min(StockDaily.date).label('min_date'),
                    func.max(StockDaily.date).label('max_date')
                )
                .where(StockDaily.code.in_(stock_codes))
                .group_by(StockDaily.code)
            )
            
            rows = session.execute(query).all()
            
            for row in rows:
                code, min_date, max_date = row
                result[code] = (min_date, max_date)
        
        return result

    def get_analysis_context(self, code: str, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """
        获取分析所需的上下文数据

        返回今日数据 + 昨日数据的对比信息

        Args:
            code: 股票代码
            target_date: 目标日期（默认今天）

        Returns:
            包含今日数据、昨日对比等信息的字典
        """
        if target_date is None:
            target_date = date.today()

        # 获取最近2天数据
        recent_data = self.get_latest_data(code, days=2)

        if not recent_data:
            logger.warning(f"未找到 {code} 的数据")
            return None

        today_data = recent_data[0]
        yesterday_data = recent_data[1] if len(recent_data) > 1 else None

        # 检查是否是非交易日，如果是，确保使用的是最近的交易日数据
        from src.core.trading_calendar import get_market_for_stock, is_market_open

        market = get_market_for_stock(code)
        if market and not is_market_open(market, target_date):
            # 非交易日，确认我们使用的是最近的交易日数据
            latest_date = today_data.date
            if is_market_open(market, latest_date):
                logger.debug(f"在非交易日 {target_date} 分析 {code}，使用最近交易日 {latest_date} 的数据")

        context = {
            "code": code,
            "date": today_data.date.isoformat(),
            "today": today_data.to_dict(),
        }

        if yesterday_data:
            context["yesterday"] = yesterday_data.to_dict()

            # 计算相比昨日的变化
            if yesterday_data.volume and yesterday_data.volume > 0:
                context["volume_change_ratio"] = round(today_data.volume / yesterday_data.volume, 2)

            if yesterday_data.close and yesterday_data.close > 0:
                context["price_change_ratio"] = round(
                    (today_data.close - yesterday_data.close) / yesterday_data.close * 100, 2
                )

            # 均线形态判断
            context["ma_status"] = self._analyze_ma_status(today_data)

        return context

    def _analyze_ma_status(self, data: StockDaily) -> str:
        """
        分析均线形态

        判断条件：
        - 多头排列：close > ma5 > ma10 > ma20
        - 空头排列：close < ma5 < ma10 < ma20
        - 震荡整理：其他情况
        """
        # 注意：这里的均线形态判断基于“close/ma5/ma10/ma20”静态比较，
        # 未考虑均线拐点、斜率、或不同数据源复权口径差异。
        # 该行为目前保留（按需求不改逻辑）。
        close = data.close or 0
        ma5 = data.ma5 or 0
        ma10 = data.ma10 or 0
        ma20 = data.ma20 or 0

        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        elif close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        elif close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        elif close < ma5 and ma5 < ma10:
            return "短期走弱 🔽"
        else:
            return "震荡整理 ↔️"

    def get_stock_sectors(self, code: str) -> Optional[List[str]]:
        """
        获取股票的板块信息

        Args:
            code: 股票代码

        Returns:
            板块列表，如果数据库中没有则返回 None
        """
        with self.session_scope() as session:
            stmt = select(StockSector).where(StockSector.code == code)
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                return result.get_sectors()
            return None

    def save_stock_sectors(self, code: str, sectors: List[str], data_source: str = "unknown") -> bool:
        """
        保存股票的板块信息

        Args:
            code: 股票代码
            sectors: 板块列表
            data_source: 数据来源

        Returns:
            是否保存成功
        """
        try:
            with self.session_scope() as session:
                stmt = select(StockSector).where(StockSector.code == code)
                existing = session.execute(stmt).scalar_one_or_none()

                if existing:
                    existing.sectors = json.dumps(sectors, ensure_ascii=False)
                    existing.data_source = data_source
                    existing.updated_at = datetime.now()
                    logger.info(f"更新股票 {code} 的板块信息: {sectors}")
                else:
                    record = StockSector.create_from_list(code, sectors, data_source)
                    session.add(record)
                    logger.info(f"保存股票 {code} 的板块信息: {sectors}")

                session.commit()
                return True
        except Exception as e:
            logger.error(f"保存股票 {code} 的板块信息失败: {e}")
            return False

    def get_sector_daily(self, name: str, date: date) -> Optional[SectorDaily]:
        """
        获取指定板块在指定日期的数据

        Args:
            name: 板块名称
            date: 日期

        Returns:
            SectorDaily 实例，如果没有则返回 None
        """
        from sqlalchemy.orm import make_transient
        
        with self.session_scope() as session:
            stmt = select(SectorDaily).where(
                and_(SectorDaily.name == name, SectorDaily.date == date)
            )
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                # 分离对象，使其在 session 关闭后仍可访问
                session.expunge(result)
                make_transient(result)
            return result

    def get_sector_change_pct(self, name: str, date: date) -> Optional[float]:
        """
        获取指定板块在指定日期的涨跌幅

        Args:
            name: 板块名称
            date: 日期

        Returns:
            涨跌幅（%），如果没有则返回 None
        """
        sector_data = self.get_sector_daily(name, date)
        if sector_data:
            return sector_data.change_pct
        return None

    def get_sector_history(self, name: str, end_date: date, days: int = 15) -> List[SectorDaily]:
        """
        获取指定板块近N天的历史数据

        Args:
            name: 板块名称
            end_date: 结束日期
            days: 获取天数，默认15天

        Returns:
            板块历史数据列表，按日期降序排列
        """
        from datetime import timedelta
        from sqlalchemy.orm import make_transient
        
        start_date = end_date - timedelta(days=days)
        
        with self.session_scope() as session:
            stmt = select(SectorDaily).where(
                and_(
                    SectorDaily.name == name,
                    SectorDaily.date >= start_date,
                    SectorDaily.date <= end_date
                )
            ).order_by(SectorDaily.date.desc())
            
            results = session.execute(stmt).scalars().all()
            if results:
                # 分离对象，使其在 session 关闭后仍可访问
                for obj in results:
                    session.expunge(obj)
                    make_transient(obj)
            return list(results) if results else []

    def get_sector_avg_change_pct(self, name: str, end_date: date, days: int = 15) -> Optional[float]:
        """
        计算指定板块近N天的平均涨跌幅

        Args:
            name: 板块名称
            end_date: 结束日期
            days: 统计天数，默认15天

        Returns:
            平均涨跌幅（%），如果没有数据则返回 None
        """
        from datetime import timedelta
        
        start_date = end_date - timedelta(days=days)
        
        with self.session_scope() as session:
            stmt = select(SectorDaily).where(
                and_(
                    SectorDaily.name == name,
                    SectorDaily.date >= start_date,
                    SectorDaily.date <= end_date
                )
            ).order_by(SectorDaily.date.desc())
            
            results = session.execute(stmt).scalars().all()
            if not results:
                return None
            
            change_pcts = [data.change_pct for data in results if data.change_pct is not None]
            if not change_pcts:
                return None
            
            return sum(change_pcts) / len(change_pcts)

    def get_all_sectors(self) -> List[str]:
        """
        获取所有有数据的板块名称列表

        Returns:
            板块名称列表
        """
        with self.session_scope() as session:
            stmt = select(SectorDaily.name).distinct()
            results = session.execute(stmt).scalars().all()
            return list(results) if results else []

    def get_sector_daily_batch(
        self, 
        sector_names: List[str], 
        date: date
    ) -> Dict[str, Optional[SectorDaily]]:
        """
        批量获取多个板块在指定日期的数据

        Args:
            sector_names: 板块名称列表
            date: 日期

        Returns:
            字典: {板块名称: SectorDaily实例, ...}
        """
        from sqlalchemy.orm import make_transient
        
        if not sector_names:
            return {}
        
        with self.session_scope() as session:
            stmt = select(SectorDaily).where(
                and_(
                    SectorDaily.name.in_(sector_names),
                    SectorDaily.date == date
                )
            )
            results = session.execute(stmt).scalars().all()
            
            result_dict = {}
            for result in results:
                session.expunge(result)
                make_transient(result)
                result_dict[result.name] = result
            
            for name in sector_names:
                if name not in result_dict:
                    result_dict[name] = None
            
            return result_dict

    def get_sector_avg_change_pct_batch(
        self, 
        sector_names: List[str], 
        end_date: date, 
        days: int = 15
    ) -> Dict[str, Optional[float]]:
        """
        批量计算多个板块近N天的平均涨跌幅

        Args:
            sector_names: 板块名称列表
            end_date: 结束日期
            days: 统计天数，默认15天

        Returns:
            字典: {板块名称: 平均涨跌幅, ...}
        """
        from datetime import timedelta
        from sqlalchemy import func
        
        if not sector_names:
            return {}
        
        start_date = end_date - timedelta(days=days)
        
        with self.session_scope() as session:
            stmt = select(
                SectorDaily.name,
                func.avg(SectorDaily.change_pct).label('avg_change')
            ).where(
                and_(
                    SectorDaily.name.in_(sector_names),
                    SectorDaily.date >= start_date,
                    SectorDaily.date <= end_date,
                    SectorDaily.change_pct.isnot(None)
                )
            ).group_by(SectorDaily.name)
            
            results = session.execute(stmt).all()
            
            result_dict = {}
            for row in results:
                result_dict[row.name] = float(row.avg_change) if row.avg_change is not None else None
            
            for name in sector_names:
                if name not in result_dict:
                    result_dict[name] = None
            
            return result_dict

    def save_sector_daily(
        self,
        name: str,
        date: date,
        change_pct: Optional[float] = None,
        stock_count: int = 0,
        limit_up_count: int = 0,
        data_source: str = "unknown"
    ) -> bool:
        """
        保存板块每日数据

        Args:
            name: 板块名称
            date: 日期
            change_pct: 涨跌幅
            stock_count: 板块内股票数量
            limit_up_count: 涨停股票数量
            data_source: 数据来源

        Returns:
            是否保存成功
        """
        try:
            with self.session_scope() as session:
                stmt = select(SectorDaily).where(
                    and_(SectorDaily.name == name, SectorDaily.date == date)
                )
                existing = session.execute(stmt).scalar_one_or_none()

                if existing:
                    if change_pct is not None:
                        existing.change_pct = change_pct
                    existing.stock_count = stock_count
                    existing.limit_up_count = limit_up_count
                    existing.data_source = data_source
                    existing.updated_at = datetime.now()
                    logger.info(f"更新板块 {name} 在 {date} 的数据: change_pct={change_pct}")
                else:
                    record = SectorDaily(
                        name=name,
                        date=date,
                        change_pct=change_pct,
                        stock_count=stock_count,
                        limit_up_count=limit_up_count,
                        data_source=data_source
                    )
                    session.add(record)
                    logger.info(f"保存板块 {name} 在 {date} 的数据: change_pct={change_pct}")

                session.commit()
                return True
        except Exception as e:
            logger.error(f"保存板块 {name} 在 {date} 的数据失败: {e}")
            return False

    def save_sector_dailies_batch(
        self,
        sectors: List[Dict],
        date: date,
        data_source: str = "unknown"
    ) -> int:
        """
        批量保存板块每日数据（优化版）

        Args:
            sectors: 板块数据列表，每个元素是包含 'name' 和 'change_pct' 的字典
            date: 日期
            data_source: 数据来源

        Returns:
            保存成功的数量
        """
        if not sectors:
            return 0
        
        saved_count = 0
        
        try:
            with self.session_scope() as session:
                # 1. 批量查询已有记录
                sector_names = [s.get('name') for s in sectors if s.get('name')]
                
                if not sector_names:
                    return 0
                
                stmt = select(SectorDaily).where(
                    and_(
                        SectorDaily.name.in_(sector_names),
                        SectorDaily.date == date
                    )
                )
                existing_records = session.execute(stmt).scalars().all()
                
                # 2. 构建已有记录的字典
                existing_dict = {rec.name: rec for rec in existing_records}
                
                # 3. 分别更新和插入
                new_records = []
                for sector in sectors:
                    name = sector.get('name')
                    change_pct = sector.get('change_pct')
                    
                    if not name:
                        continue
                    
                    if name in existing_dict:
                        # 更新现有记录
                        existing = existing_dict[name]
                        if change_pct is not None:
                            existing.change_pct = change_pct
                        existing.stock_count = sector.get('stock_count', 0)
                        existing.limit_up_count = sector.get('limit_up_count', 0)
                        existing.data_source = data_source
                        existing.updated_at = datetime.now()
                        logger.info(f"更新板块 {name} 在 {date} 的数据: change_pct={change_pct}")
                    else:
                        # 添加新记录
                        record = SectorDaily(
                            name=name,
                            date=date,
                            change_pct=change_pct,
                            stock_count=sector.get('stock_count', 0),
                            limit_up_count=sector.get('limit_up_count', 0),
                            data_source=data_source
                        )
                        new_records.append(record)
                        logger.info(f"保存板块 {name} 在 {date} 的数据: change_pct={change_pct}")
                
                if new_records:
                    session.add_all(new_records)
                
                session.commit()
                saved_count = len(sector_names)
                
        except Exception as e:
            logger.error(f"批量保存板块数据失败: {e}")
        
        return saved_count

    def save_sector_rankings(
        self,
        top_sectors: List[Dict],
        bottom_sectors: List[Dict],
        date: Optional[date] = None,
        data_source: str = "unknown"
    ) -> int:
        """
        批量保存板块排行数据

        Args:
            top_sectors: 领涨板块列表
            bottom_sectors: 领跌板块列表
            date: 日期（默认为今天）
            data_source: 数据来源

        Returns:
            保存成功的数量
        """
        if date is None:
            date = datetime.now().date()

        all_sectors = top_sectors + bottom_sectors
        
        # 使用批量保存方法，性能更优
        return self.save_sector_dailies_batch(
            sectors=all_sectors,
            date=date,
            data_source=data_source
        )

    @staticmethod
    def _parse_published_date(value: Optional[str]) -> Optional[datetime]:
        """
        解析发布时间字符串（失败返回 None）
        """
        if not value:
            return None

        if isinstance(value, datetime):
            return value

        text = str(value).strip()
        if not text:
            return None

        # 优先尝试 ISO 格式
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

        return None

    @staticmethod
    def _safe_json_dumps(data: Any) -> str:
        """
        安全序列化为 JSON 字符串
        """
        try:
            return json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps(str(data), ensure_ascii=False)

    @staticmethod
    def _build_raw_result(result: Any) -> Dict[str, Any]:
        """
        生成完整分析结果字典
        """
        data = result.to_dict() if hasattr(result, "to_dict") else {}
        data.update(
            {
                "data_sources": getattr(result, "data_sources", ""),
                "raw_response": getattr(result, "raw_response", None),
            }
        )
        return data

    @staticmethod
    def _parse_sniper_value(value: Any) -> Optional[float]:
        """
        Parse a sniper point value from various formats to float.

        Handles: numeric types, plain number strings, Chinese price formats
        like "18.50元", range formats like "18.50-19.00", and text with
        embedded numbers while filtering out MA indicators.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            v = float(value)
            return v if v > 0 else None

        text = str(value).replace(",", "").replace("，", "").strip()
        if not text or text == "-" or text == "—" or text == "N/A":
            return None

        # 尝试直接解析纯数字字符串
        try:
            return float(text)
        except ValueError:
            pass

        # 优先截取 "：" 到 "元" 之间的价格，避免误提取 MA5/MA10 等技术指标数字
        colon_pos = max(text.rfind("："), text.rfind(":"))
        yuan_pos = text.find("元", colon_pos + 1 if colon_pos != -1 else 0)
        if yuan_pos != -1:
            segment_start = colon_pos + 1 if colon_pos != -1 else 0
            segment = text[segment_start:yuan_pos]

            # 使用 finditer 并过滤掉 MA 开头的数字
            matches = list(re.finditer(r"-?\d+(?:\.\d+)?", segment))
            valid_numbers = []
            for m in matches:
                # 检查前面是否是 "MA" (忽略大小写)
                start_idx = m.start()
                if start_idx >= 2:
                    prefix = segment[start_idx - 2 : start_idx].upper()
                    if prefix == "MA":
                        continue
                valid_numbers.append(m.group())

            if valid_numbers:
                try:
                    return abs(float(valid_numbers[-1]))
                except ValueError:
                    pass

        # 兜底：无"元"字时（如 "102.10-103.00（MA5附近）"），
        # 提取最后一个非 MA 前缀的数字
        valid_numbers = []
        for m in re.finditer(r"\d+(?:\.\d+)?", text):
            start_idx = m.start()
            if start_idx >= 2 and text[start_idx - 2 : start_idx].upper() == "MA":
                continue
            valid_numbers.append(m.group())
        if valid_numbers:
            try:
                return float(valid_numbers[-1])
            except ValueError:
                pass
        return None

    def _extract_sniper_points(self, result: Any) -> Dict[str, Optional[float]]:
        """
        Extract sniper point values from an AnalysisResult.

        Tries multiple extraction paths to handle different dashboard structures:
        1. result.get_sniper_points() (standard path)
        2. Direct dashboard dict traversal with various nesting levels
        3. Fallback from raw_result dict if available
        """
        raw_points = {}

        # Path 1: standard method
        if hasattr(result, "get_sniper_points"):
            raw_points = result.get_sniper_points() or {}

        # Path 2: direct dashboard traversal when standard path yields empty values
        if not any(raw_points.get(k) for k in ("ideal_buy", "secondary_buy", "stop_loss", "take_profit")):
            dashboard = getattr(result, "dashboard", None)
            if isinstance(dashboard, dict):
                raw_points = self._find_sniper_in_dashboard(dashboard) or raw_points

        # Path 3: try raw_result for agent mode results
        if not any(raw_points.get(k) for k in ("ideal_buy", "secondary_buy", "stop_loss", "take_profit")):
            raw_response = getattr(result, "raw_response", None)
            if isinstance(raw_response, dict):
                raw_points = self._find_sniper_in_dashboard(raw_response) or raw_points

        return {
            "ideal_buy": self._parse_sniper_value(raw_points.get("ideal_buy")),
            "secondary_buy": self._parse_sniper_value(raw_points.get("secondary_buy")),
            "stop_loss": self._parse_sniper_value(raw_points.get("stop_loss")),
            "take_profit": self._parse_sniper_value(raw_points.get("take_profit")),
        }

    @staticmethod
    def _find_sniper_in_dashboard(d: dict) -> Optional[Dict[str, Any]]:
        """
        Recursively search for sniper_points in a dashboard dict.
        Handles various nesting: dashboard.battle_plan.sniper_points,
        dashboard.dashboard.battle_plan.sniper_points, etc.
        """
        if not isinstance(d, dict):
            return None

        # Direct: d has sniper_points keys at top level
        if "ideal_buy" in d:
            return d

        # d.sniper_points
        sp = d.get("sniper_points")
        if isinstance(sp, dict) and sp:
            return sp

        # d.battle_plan.sniper_points
        bp = d.get("battle_plan")
        if isinstance(bp, dict):
            sp = bp.get("sniper_points")
            if isinstance(sp, dict) and sp:
                return sp

        # d.dashboard.battle_plan.sniper_points (double-nested)
        inner = d.get("dashboard")
        if isinstance(inner, dict):
            bp = inner.get("battle_plan")
            if isinstance(bp, dict):
                sp = bp.get("sniper_points")
                if isinstance(sp, dict) and sp:
                    return sp

        return None

    @staticmethod
    def _build_fallback_url_key(code: str, title: str, source: str, published_date: Optional[datetime]) -> str:
        """
        生成无 URL 时的去重键（确保稳定且较短）
        """
        date_str = published_date.isoformat() if published_date else ""
        raw_key = f"{code}|{title}|{source}|{date_str}"
        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
        return f"no-url:{code}:{digest}"

    def save_conversation_message(self, session_id: str, role: str, content: str) -> None:
        """
        保存 Agent 对话消息
        """
        with self.session_scope() as session:
            msg = ConversationMessage(session_id=session_id, role=role, content=content)
            session.add(msg)

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取 Agent 对话历史
        """
        with self.session_scope() as session:
            stmt = (
                select(ConversationMessage)
                .filter(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(limit)
            )
            messages = session.execute(stmt).scalars().all()

            # 倒序返回，保证时间顺序
            return [{"role": msg.role, "content": msg.content} for msg in reversed(messages)]

    def get_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取聊天会话列表（从 conversation_messages 聚合）

        Returns:
            按最近活跃时间倒序的会话列表，每条包含 session_id, title, message_count, last_active
        """
        from sqlalchemy import func

        with self.session_scope() as session:
            # 聚合每个 session 的消息数和最后活跃时间
            stmt = (
                select(
                    ConversationMessage.session_id,
                    func.count(ConversationMessage.id).label("message_count"),
                    func.min(ConversationMessage.created_at).label("created_at"),
                    func.max(ConversationMessage.created_at).label("last_active"),
                )
                .group_by(ConversationMessage.session_id)
                .order_by(desc(func.max(ConversationMessage.created_at)))
                .limit(limit)
            )
            rows = session.execute(stmt).all()

            results = []
            for row in rows:
                sid = row.session_id
                # 取该会话第一条 user 消息作为标题
                first_user_msg = session.execute(
                    select(ConversationMessage.content)
                    .where(
                        and_(
                            ConversationMessage.session_id == sid,
                            ConversationMessage.role == "user",
                        )
                    )
                    .order_by(ConversationMessage.created_at)
                    .limit(1)
                ).scalar()
                title = (first_user_msg or "新对话")[:60]

                results.append(
                    {
                        "session_id": sid,
                        "title": title,
                        "message_count": row.message_count,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "last_active": row.last_active.isoformat() if row.last_active else None,
                    }
                )
            return results

    def get_conversation_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取单个会话的完整消息列表（用于前端恢复历史）
        """
        with self.session_scope() as session:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at)
                .limit(limit)
            )
            messages = session.execute(stmt).scalars().all()
            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                }
                for msg in messages
            ]

    def delete_conversation_session(self, session_id: str) -> int:
        """
        删除指定会话的所有消息

        Returns:
            删除的消息数
        """
        with self.session_scope() as session:
            result = session.execute(delete(ConversationMessage).where(ConversationMessage.session_id == session_id))
            return result.rowcount

    def save_stock_indicators(
        self,
        code: str,
        indicator_type: str,
        df: pd.DataFrame,
        data_source: str = "Unknown"
    ) -> int:
        """
        批量保存股票技术指标数据

        Args:
            code: 股票代码
            indicator_type: 指标类型
            df: 包含指标数据的 DataFrame
            data_source: 数据来源

        Returns:
            保存的记录数（新增+更新）
        """
        if df is None or df.empty:
            logger.warning(f"保存指标数据为空，跳过 {code} {indicator_type}")
            return 0

        saved_count = 0
        updated_count = 0

        with self.get_session() as session:
            try:
                for _, row in df.iterrows():
                    row_date = self._parse_date(row.get("date"))
                    if not row_date:
                        continue

                    existing = session.execute(
                        select(StockIndicator).where(
                            and_(
                                StockIndicator.code == code,
                                StockIndicator.date == row_date,
                                StockIndicator.indicator_type == indicator_type
                            )
                        )
                    ).scalar_one_or_none()

                    indicator_data_dict = row.to_dict()
                    if 'date' in indicator_data_dict:
                        del indicator_data_dict['date']

                    if existing:
                        existing.indicator_data = json.dumps(indicator_data_dict, ensure_ascii=False)
                        existing.updated_at = datetime.now()
                        updated_count += 1
                    else:
                        record = StockIndicator.create_from_dict(
                            code=code,
                            date=row_date,
                            indicator_type=indicator_type,
                            data=indicator_data_dict
                        )
                        session.add(record)
                        saved_count += 1

                session.commit()
                logger.info(f"保存 {code} {indicator_type} 指标数据成功，新增 {saved_count} 条，更新 {updated_count} 条")
                return saved_count + updated_count
            except Exception as e:
                session.rollback()
                logger.error(f"保存 {code} {indicator_type} 指标数据失败: {e}")
                raise

    def get_stock_indicators(
        self,
        code: str,
        indicator_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[StockIndicator]:
        """
        获取股票技术指标数据

        Args:
            code: 股票代码
            indicator_type: 指标类型（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            StockIndicator 对象列表
        """
        with self.get_session() as session:
            conditions = [StockIndicator.code == code]

            if indicator_type:
                conditions.append(StockIndicator.indicator_type == indicator_type)
            if start_date:
                conditions.append(StockIndicator.date >= start_date)
            if end_date:
                conditions.append(StockIndicator.date <= end_date)

            results = (
                session.execute(
                    select(StockIndicator)
                    .where(and_(*conditions))
                    .order_by(StockIndicator.date)
                )
                .scalars()
                .all()
            )

            return list(results)

    def get_stock_indicators_as_df(
        self,
        code: str,
        indicator_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        获取股票技术指标数据并转换为 DataFrame

        Args:
            code: 股票代码
            indicator_type: 指标类型
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            pandas DataFrame
        """
        indicators = self.get_stock_indicators(code, indicator_type, start_date, end_date)

        if not indicators:
            return pd.DataFrame()

        data_list = []
        for ind in indicators:
            ind_data = ind.get_indicator_data()
            ind_data['date'] = ind.date
            data_list.append(ind_data)

        return pd.DataFrame(data_list)

    def save_visualization_search_history(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        selected_indicators: Optional[List[str]] = None
    ) -> int:
        """
        保存可视化搜索历史记录

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            selected_indicators: 选中的指标列表

        Returns:
            保存的记录ID
        """
        with self.get_session() as session:
            try:
                record = VisualizationSearchHistory.create(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    selected_indicators=selected_indicators
                )
                session.add(record)
                session.commit()
                logger.info(f"保存可视化搜索历史成功: {stock_code}")
                return record.id
            except Exception as e:
                session.rollback()
                logger.error(f"保存可视化搜索历史失败: {e}")
                raise

    def get_visualization_search_history(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        获取可视化搜索历史记录

        Args:
            limit: 返回数量限制

        Returns:
            搜索历史记录列表（已去重）
        """
        with self.get_session() as session:
            results = (
                session.execute(
                    select(VisualizationSearchHistory)
                    .order_by(desc(VisualizationSearchHistory.searched_at))
                )
                .scalars()
                .all()
            )

            # 去重：只保留每个股票代码的最新记录
            seen_stock_codes = set()
            unique_results = []
            for record in results:
                if record.stock_code not in seen_stock_codes:
                    seen_stock_codes.add(record.stock_code)
                    unique_results.append(record)
                    # 达到 limit 限制后停止
                    if len(unique_results) >= limit:
                        break

            return [
                {
                    "id": record.id,
                    "stock_code": record.stock_code,
                    "stock_name": record.stock_name,
                    "searched_at": record.searched_at.isoformat() if record.searched_at else None,
                    "selected_indicators": record.get_selected_indicators()
                }
                for record in unique_results
            ]

    def delete_duplicate_visualization_history(self, stock_code: str) -> int:
        """
        删除重复的可视化搜索历史记录，只保留最新的一条

        Args:
            stock_code: 股票代码

        Returns:
            删除的记录数
        """
        with self.get_session() as session:
            try:
                latest_record = (
                    session.execute(
                        select(VisualizationSearchHistory)
                        .where(VisualizationSearchHistory.stock_code == stock_code)
                        .order_by(desc(VisualizationSearchHistory.searched_at))
                        .limit(1)
                    )
                    .scalar_one_or_none()
                )

                if not latest_record:
                    return 0

                result = session.execute(
                    delete(VisualizationSearchHistory).where(
                        and_(
                            VisualizationSearchHistory.stock_code == stock_code,
                            VisualizationSearchHistory.id != latest_record.id
                        )
                    )
                )
                session.commit()
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"删除 {stock_code} 的重复搜索历史: {deleted_count} 条")
                return deleted_count
            except Exception as e:
                session.rollback()
                logger.error(f"删除重复搜索历史失败: {e}")
                return 0

    def delete_visualization_search_history(self, record_id: int) -> bool:
        """
        删除单条可视化搜索历史记录

        Args:
            record_id: 记录ID

        Returns:
            是否成功删除
        """
        with self.get_session() as session:
            try:
                result = session.execute(
                    delete(VisualizationSearchHistory).where(
                        VisualizationSearchHistory.id == record_id
                    )
                )
                session.commit()
                success = result.rowcount > 0
                if success:
                    logger.info(f"删除可视化搜索历史记录: {record_id}")
                return success
            except Exception as e:
                session.rollback()
                logger.error(f"删除可视化搜索历史失败: {e}")
                return False

    def update_visualization_search_history_timestamp(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        更新可视化搜索历史记录的时间戳为当前时间

        Args:
            record_id: 记录ID

        Returns:
            更新后的记录信息，失败返回 None
        """
        with self.get_session() as session:
            try:
                record = session.execute(
                    select(VisualizationSearchHistory).where(
                        VisualizationSearchHistory.id == record_id
                    )
                ).scalar_one_or_none()

                if not record:
                    logger.warning(f"未找到可视化搜索历史记录: {record_id}")
                    return None

                record.searched_at = datetime.now()
                session.commit()
                logger.info(f"更新可视化搜索历史记录时间戳: {record_id}, stock_code: {record.stock_code}")

                return {
                    "id": record.id,
                    "stock_code": record.stock_code,
                    "stock_name": record.stock_name,
                    "searched_at": record.searched_at.isoformat() if record.searched_at else None,
                    "selected_indicators": record.get_selected_indicators()
                }
            except Exception as e:
                session.rollback()
                logger.error(f"更新可视化搜索历史记录时间戳失败: {e}")
                return None


# 便捷函数
def get_db() -> DatabaseManager:
    """获取数据库管理器实例的快捷方式"""
    return DatabaseManager.get_instance()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    db = get_db()

    print("=== 数据库测试 ===")
    print(f"数据库初始化成功")

    # 测试检查今日数据
    has_data = db.has_today_data("600519")
    print(f"茅台今日是否有数据: {has_data}")

    # 测试保存数据
    test_df = pd.DataFrame(
        {
            "date": [date.today()],
            "open": [1800.0],
            "high": [1850.0],
            "low": [1780.0],
            "close": [1820.0],
            "volume": [10000000],
            "amount": [18200000000],
            "pct_chg": [1.5],
            "ma5": [1810.0],
            "ma10": [1800.0],
            "ma20": [1790.0],
            "volume_ratio": [1.2],
        }
    )

    saved = db.save_daily_data(test_df, "600519", "TestSource")
    print(f"保存测试数据: {saved} 条")

    # 测试获取上下文
    context = db.get_analysis_context("600519")
    print(f"分析上下文: {context}")
