# -*- coding: utf-8 -*-
"""
===================================
全局异常处理中间件
===================================

职责：
1. 捕获未处理的异常
2. 统一错误响应格式
3. 记录错误日志
"""

import logging
import traceback
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> str:
    """
    从请求头获取或生成请求追踪 ID
    
    Args:
        request: FastAPI 请求对象
    
    Returns:
        请求追踪 ID
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    全局异常处理中间件
    
    捕获所有未处理的异常，返回统一格式的错误响应
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        处理请求，捕获异常
        
        Args:
            request: 请求对象
            call_next: 下一个处理器
            
        Returns:
            Response: 响应对象
        """
        request_id = get_request_id(request)
        
        try:
            response = await call_next(request)
            # 将 request_id 添加到响应头
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            # 记录错误日志
            logger.error(
                f"未处理的异常: {e}\n"
                f"请求 ID: {request_id}\n"
                f"请求路径: {request.url.path}\n"
                f"请求方法: {request.method}\n"
                f"堆栈: {traceback.format_exc()}"
            )
            
            # 返回统一格式的错误响应
            from api.v1.schemas.common import error_response
            error_resp = error_response(
                error="internal_error",
                message="服务器内部错误，请稍后重试",
                code=500,
                detail=str(e) if logger.isEnabledFor(logging.DEBUG) else None,
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_resp.model_dump()
            )


def add_error_handlers(app) -> None:
    """
    添加全局异常处理器
    
    为 FastAPI 应用添加各类异常的处理器
    
    Args:
        app: FastAPI 应用实例
    """
    from fastapi import HTTPException
    from fastapi.exceptions import RequestValidationError
    from api.v1.schemas.common import error_response
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理 HTTP 异常"""
        request_id = get_request_id(request)
        
        # 如果 detail 已经是标准格式的 dict，直接使用
        if isinstance(exc.detail, dict) and "code" in exc.detail and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
                headers={"X-Request-ID": request_id}
            )
        
        # 使用新的标准错误格式
        error_resp = error_response(
            error="http_error",
            message=str(exc.detail) if exc.detail else "HTTP Error",
            code=exc.status_code,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_resp.model_dump(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """处理请求验证异常"""
        request_id = get_request_id(request)
        
        error_resp = error_response(
            error="validation_error",
            message="请求参数验证失败",
            code=422,
            detail=exc.errors(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_resp.model_dump(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理通用异常"""
        request_id = get_request_id(request)
        
        logger.error(
            f"未处理的异常: {exc}\n"
            f"请求 ID: {request_id}\n"
            f"请求路径: {request.url.path}\n"
            f"堆栈: {traceback.format_exc()}"
        )
        
        error_resp = error_response(
            error="internal_error",
            message="服务器内部错误",
            code=500,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_resp.model_dump(),
            headers={"X-Request-ID": request_id}
        )

