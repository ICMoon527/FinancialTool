# -*- coding: utf-8 -*-
"""
===================================
通用响应模型
===================================

职责：
1. 定义通用的响应模型（HealthResponse, ErrorResponse 等）
2. 提供统一的响应格式
3. 提供响应工具函数
"""

import uuid
from datetime import datetime
from typing import Optional, Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar('T')


class RootResponse(BaseModel):
    """API 根路由响应"""
    
    message: str = Field(..., description="API 运行状态消息", example="Daily Stock Analysis API is running")
    version: Optional[str] = Field(None, description="API 版本", example="1.0.0")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Daily Stock Analysis API is running",
                "version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""
    
    status: str = Field(..., description="服务状态", example="ok")
    timestamp: Optional[str] = Field(None, description="时间戳")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class ApiResponseMeta(BaseModel):
    """API 响应元数据"""
    
    timestamp: str = Field(..., description="响应时间戳", default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = Field(None, description="请求追踪 ID")
    api_version: str = Field("1.0.0", description="API 版本")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-01T12:00:00",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "api_version": "1.0.0"
            }
        }


class ApiResponse(BaseModel, Generic[T]):
    """标准 API 成功响应格式"""
    
    code: int = Field(200, description="响应状态码", example=200)
    message: str = Field("success", description="响应消息", example="success")
    data: Optional[T] = Field(None, description="响应数据")
    meta: ApiResponseMeta = Field(default_factory=ApiResponseMeta, description="响应元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {"key": "value"},
                "meta": {
                    "timestamp": "2024-01-01T12:00:00",
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "api_version": "1.0.0"
                }
            }
        }


class ApiError(BaseModel):
    """API 错误响应格式"""
    
    code: int = Field(..., description="错误状态码", example=400)
    error: str = Field(..., description="错误类型", example="validation_error")
    message: str = Field(..., description="错误详情", example="请求参数错误")
    detail: Optional[Any] = Field(None, description="附加错误信息")
    meta: ApiResponseMeta = Field(default_factory=ApiResponseMeta, description="响应元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 404,
                "error": "not_found",
                "message": "资源不存在",
                "detail": None,
                "meta": {
                    "timestamp": "2024-01-01T12:00:00",
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "api_version": "1.0.0"
                }
            }
        }


class ErrorResponse(BaseModel):
    """错误响应（向后兼容）"""
    
    error: str = Field(..., description="错误类型", example="validation_error")
    message: str = Field(..., description="错误详情", example="请求参数错误")
    detail: Optional[Any] = Field(None, description="附加错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "not_found",
                "message": "资源不存在",
                "detail": None
            }
        }


class SuccessResponse(BaseModel):
    """通用成功响应（向后兼容）"""
    
    success: bool = Field(True, description="是否成功")
    message: Optional[str] = Field(None, description="成功消息")
    data: Optional[Any] = Field(None, description="响应数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "操作成功",
                "data": None
            }
        }


def success_response(
    data: Any = None,
    message: str = "success",
    code: int = 200,
    request_id: Optional[str] = None
) -> ApiResponse:
    """
    生成标准成功响应
    
    Args:
        data: 响应数据
        message: 响应消息
        code: 状态码
        request_id: 请求追踪 ID
    
    Returns:
        ApiResponse: 标准响应对象
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    return ApiResponse(
        code=code,
        message=message,
        data=data,
        meta=ApiResponseMeta(
            request_id=request_id
        )
    )


def error_response(
    error: str,
    message: str,
    code: int = 400,
    detail: Any = None,
    request_id: Optional[str] = None
) -> ApiError:
    """
    生成标准错误响应
    
    Args:
        error: 错误类型
        message: 错误消息
        code: 状态码
        detail: 详细错误信息
        request_id: 请求追踪 ID
    
    Returns:
        ApiError: 标准错误对象
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    return ApiError(
        code=code,
        error=error,
        message=message,
        detail=detail,
        meta=ApiResponseMeta(
            request_id=request_id
        )
    )

