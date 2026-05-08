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
    timestamp: str  # ISO格式时间字符串


class IntradaySignal(BaseModel):
    """做T信号"""

    stock_code: str
    signal_type: str  # "buy" | "sell"
    trigger_time: str
    price: float
    score: int
    max_score: int = 10
    confidence: float
    position_advice: str
    reasoning: str = ""
    # 引力场修正信息
    gravity_adjustment: float = 0.0
    support_force: float = 0.0
    pressure_force: float = 0.0


class ReferenceLine(BaseModel):
    """支撑/压力参考线"""

    id: str
    label: str
    price: float
    category: str
    color: str
    style: str = "dashed"
    base_weight: float = 1.0


class IntradayDataResponse(BaseModel):
    """分时数据完整响应"""

    stock_code: str
    stock_name: str = ""
    date: str
    kline_data: List[IntradayKlinePoint] = Field(default_factory=list)
    signals: List[IntradaySignal] = Field(default_factory=list)
    reference_lines: List[ReferenceLine] = Field(default_factory=list)
    signal_summary: Dict[str, Any] = Field(default_factory=dict)


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
