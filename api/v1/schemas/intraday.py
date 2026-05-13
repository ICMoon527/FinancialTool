# -*- coding: utf-8 -*-
"""分时做T API 响应模型"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IntradayKlinePoint(BaseModel):
    """分时K线数据点"""

    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Amount: Optional[float] = None
    AvgPrice: Optional[float] = None  # 累计分时均价（成交额/成交量）
    timestamp: str  # ISO格式时间字符串


class WeightContribution(BaseModel):
    """单项权重贡献明细"""

    key: str  # 权重键名
    label: str  # 中文标签
    weight: float  # 满分数
    triggered: bool = False  # 是否触发
    score: float = 0.0  # 实际得分


class IntradaySignal(BaseModel):
    """做T信号"""

    stock_code: str
    signal_type: str  # "buy" | "sell"
    trigger_time: str
    price: float
    score: float
    max_score: int = 10
    confidence: float
    position_advice: str
    reasoning: str = ""
    gravity_adjustment: float = 0.0  # 引力场修正分
    support_force: float = 0.0
    pressure_force: float = 0.0
    buy_weight_details: List[WeightContribution] = Field(default_factory=list)
    sell_weight_details: List[WeightContribution] = Field(default_factory=list)


class ReferenceLine(BaseModel):
    """支撑/压力参考线"""

    id: str
    label: str
    price: float
    category: str
    color: str
    style: str = "dashed"
    base_weight: float = 1.0


# ── 指标子图模型 ──


class IndicatorLinePoint(BaseModel):
    """指标线数据点"""

    time: str  # 时间标签 "HH:MM"
    value: float


class IndicatorLine(BaseModel):
    """指标图中的一条线"""

    name: str
    label: str
    color: str
    data: List[IndicatorLinePoint] = Field(default_factory=list)


class IndicatorSubChart(BaseModel):
    """单个指标子图"""

    id: str  # "absorption" / "main_in_out" / "cyw"
    label: str  # 中文标题 "主力吸筹"
    height: int = 120
    lines: List[IndicatorLine] = Field(default_factory=list)
    signal_text: str = ""  # 最新信号文本，如 "正T买入 ↑"、"卖出 ↓" 等


class IntradayDataResponse(BaseModel):
    """分时数据完整响应"""

    stock_code: str
    stock_name: str = ""
    date: str
    kline_data: List[IntradayKlinePoint] = Field(default_factory=list)
    signals: List[IntradaySignal] = Field(default_factory=list)
    reference_lines: List[ReferenceLine] = Field(default_factory=list)
    indicator_sub_charts: List[IndicatorSubChart] = Field(default_factory=list)
    signal_summary: Dict[str, Any] = Field(default_factory=dict)
    warm_up_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="前一个交易日的分时终值快照，用于开盘时指标预热（含价格、成交量、技术指标终值）",
    )


class SearchHistoryItem(BaseModel):
    """搜索历史条目"""

    id: int
    stock_code: str
    stock_name: str = ""
    date: str = ""
    search_time: str = ""


class SearchHistoryResponse(BaseModel):
    """搜索历史响应"""

    items: List[SearchHistoryItem] = Field(default_factory=list)
    total: int = 0


class SearchHistoryRequest(BaseModel):
    """保存搜索历史请求"""

    stock_code: str
    stock_name: str = ""
    date: str = ""


class DeleteHistoryResponse(BaseModel):
    """删除历史响应"""

    success: bool = True
    message: str = ""


# ── 批量状态查询 ──


class StockSnapshot(BaseModel):
    """单只股票实时快照"""

    stock_code: str
    stock_name: str = ""
    latest_price: float = 0.0
    change_pct: float = 0.0  # 涨跌幅百分比
    open_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    timestamp: str = ""  # "HH:MM:SS"


class BatchStatusRequest(BaseModel):
    """批量状态查询请求"""

    stock_codes: List[str] = Field(..., description="需要查询的股票代码列表")
    current_code: str = Field("", description="当前展示的股票代码")


class BatchStatusResponse(BaseModel):
    """批量状态查询响应"""

    snapshots: Dict[str, StockSnapshot] = Field(default_factory=dict)
    current_updated: bool = False  # 当前展示股票是否有新数据
    current_full_data: Optional[IntradayDataResponse] = None  # 仅 current_updated=True 时有值
