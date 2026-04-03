# -*- coding: utf-8 -*-
"""Strategy Backtest API schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyInfo(BaseModel):
    id: str = Field(..., description="策略ID")
    name: str = Field(..., description="策略名称")
    description: str = Field("", description="策略描述")
    type: str = Field("UNKNOWN", description="策略类型")


class StrategyListResponse(BaseModel):
    strategies: List[StrategyInfo] = Field(default_factory=list)


class StrategyBacktestRunRequest(BaseModel):
    strategy_id: str = Field(..., description="策略ID")
    start_date: Optional[str] = Field(None, description="开始日期 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="结束日期 (YYYY-MM-DD)")
    stock_pool: Optional[List[str]] = Field(None, description="股票池")
    max_positions: Optional[int] = Field(None, description="最高持仓数（例如：3表示最多同时持有3只股票）")


class StrategyBacktestRunResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    message: str = Field("", description="消息")
    results: Optional[Dict[str, Any]] = Field(None, description="回测结果")
    metrics: Optional[Dict[str, Any]] = Field(None, description="绩效指标")
    reports: Optional[Dict[str, Any]] = Field(None, description="报告路径")


class StrategyBacktestRunAsyncRequest(BaseModel):
    """策略回测异步运行请求"""
    strategy_id: str = Field(..., description="策略ID")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    max_positions: int = Field(3, ge=1, description="最高持仓数（例如：3表示最多同时持有3只股票）")


class StrategyBacktestRunAsyncResponse(BaseModel):
    """策略回测异步运行响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    task_id: str = Field(..., description="任务ID")


class StrategyBacktestTaskStatusResponse(BaseModel):
    """策略回测任务状态响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    task: Optional[Dict[str, Any]] = Field(None, description="任务信息")


class StrategyBacktestStopByTaskIdRequest(BaseModel):
    """通过任务ID停止策略回测请求"""
    task_id: str = Field(..., description="任务ID")
