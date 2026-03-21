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


class VisualizationSearchRequest(BaseModel):
    """可视化搜索请求"""
    
    stock_code: str = Field(..., description="股票代码")
    stock_name: Optional[str] = Field(None, description="股票名称")
    days: int = Field(60, ge=1, le=3650, description="获取天数")
    indicator_types: Optional[List[str]] = Field(None, description="指标类型列表，不填则返回所有可用指标")
