# -*- coding: utf-8 -*-
"""
===================================
可视化数据相关模型
===================================

职责：
1. 定义可视化相关的请求和响应模型
"""

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class VisualizationIndicatorData(BaseModel):
    """单个指标数据"""
    
    indicator_type: str = Field(..., description="指标类型")
    data: List[Dict[str, Any]] = Field(..., description="指标数据列表")


class VisualizationResponse(BaseModel):
    """可视化数据响应"""
    
    stock_code: str = Field(..., description="股票代码")
    stock_name: Optional[str] = Field(None, description="股票名称")
    kline_data: List[Dict[str, Any]] = Field(..., description="K线数据")
    indicators: List[VisualizationIndicatorData] = Field(default_factory=list, description="指标数据列表")
    chip_distribution: Optional[Dict[str, Any]] = Field(None, description="筹码分布数据")
    circulating_shares_updated: bool = Field(False, description="流通股本是否已更新")
    circulating_shares: Optional[float] = Field(None, description="最新流通股本")
    turnover_filled_count: int = Field(0, description="填充的历史换手率记录数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stock_code": "600519",
                "stock_name": "贵州茅台",
                "kline_data": [],
                "indicators": []
            }
        }


class VisualizationSearchHistoryItem(BaseModel):
    """可视化搜索历史项"""
    
    id: int = Field(..., description="记录ID")
    stock_code: str = Field(..., description="股票代码")
    stock_name: Optional[str] = Field(None, description="股票名称")
    searched_at: str = Field(..., description="搜索时间")
    selected_indicators: List[str] = Field(default_factory=list, description="选中的指标列表")


class VisualizationSearchHistoryResponse(BaseModel):
    """可视化搜索历史响应"""
    
    items: List[VisualizationSearchHistoryItem] = Field(default_factory=list, description="搜索历史列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": []
            }
        }


class ChipDistributionResponse(BaseModel):
    """筹码分布响应"""
    
    stock_code: str = Field(..., description="股票代码")
    price_bins: List[float] = Field(..., description="价格区间列表")
    chip_volumes: List[float] = Field(..., description="筹码数量列表")
    profit_volumes: List[float] = Field(..., description="获利盘数量列表")
    loss_volumes: List[float] = Field(..., description="套牢盘数量列表")
    avg_cost: Optional[float] = Field(None, description="平均成本价")
    max_chip_price: Optional[float] = Field(None, description="筹码集中度最高价格")
    current_price: Optional[float] = Field(None, description="当前价格")


class VisualizationSearchRequest(BaseModel):
    """可视化搜索请求"""

    stock_code: str = Field(..., description="股票代码")
    stock_name: Optional[str] = Field(None, description="股票名称")
    days: int = Field(60, ge=1, le=3650, description="获取天数")
    indicator_types: Optional[List[str]] = Field(None, description="指标类型列表，不填则返回所有可用指标")
