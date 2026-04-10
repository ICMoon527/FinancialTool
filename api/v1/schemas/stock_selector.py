# -*- coding: utf-8 -*-
"""
Stock Selector API Schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrategyInfo(BaseModel):
    """Strategy metadata."""

    id: str
    name: str
    display_name: str
    description: str
    strategy_type: str  # "NATURAL_LANGUAGE" or "PYTHON"
    category: str
    source: str
    version: str
    created_at: datetime
    is_active: bool


class StrategyMatchInfo(BaseModel):
    """Single strategy match result."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    strategy_id: str
    strategy_name: str
    matched: bool
    score: float = 0.0
    reason: Optional[str] = None
    match_details: Dict[str, Any] = Field(default_factory=dict)


class StockCandidateInfo(BaseModel):
    """Stock candidate with strategy matches."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    stock_code: str
    stock_name: str
    current_price: float
    overall_score: float
    strategy_matches: List[StrategyMatchInfo]
    created_at: datetime
    sectors: List[str] = Field(default_factory=list)
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class StockSelectorRequest(BaseModel):
    """Request to screen stocks."""

    stock_codes: Optional[List[str]] = None
    strategy_ids: Optional[List[str]] = None
    top_n: int = 5
    update_data: bool = False
    update_realtime: bool = False


class StockSelectorResponse(BaseModel):
    """Response from stock screening."""

    success: bool
    candidates: List[StockCandidateInfo]
    total_screened: int
    execution_time_ms: float = 0.0
    error: Optional[str] = None


class StrategiesResponse(BaseModel):
    """Response with available strategies."""

    success: bool
    strategies: List[StrategyInfo]
    active_strategy_ids: List[str]


class ActivateStrategyRequest(BaseModel):
    """Request to activate a strategy."""

    strategy_id: str


class DeactivateStrategyRequest(BaseModel):
    """Request to deactivate a strategy."""

    strategy_id: str


class StockSelectorConfigResponse(BaseModel):
    """选股器配置响应."""

    success: bool
    default_top_n: int
    error: Optional[str] = None
