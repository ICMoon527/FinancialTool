# -*- coding: utf-8 -*-
"""
===================================
可视化数据接口
===================================

职责：
1. GET /api/v1/visualization/{stock_code} 获取股票可视化数据
2. GET /api/v1/visualization/history 获取搜索历史
3. POST /api/v1/visualization/history 保存搜索历史
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends

from api.deps import get_database_manager
from api.v1.schemas.visualization import (
    VisualizationResponse,
    VisualizationSearchHistoryResponse,
    VisualizationSearchHistoryItem,
    VisualizationSearchRequest
)
from api.v1.schemas.common import ErrorResponse
from src.storage import DatabaseManager
from src.services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/history",
    response_model=VisualizationSearchHistoryResponse,
    responses={
        200: {"description": "搜索历史列表"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取可视化搜索历史",
    description="获取用户的可视化搜索历史记录"
)
def get_search_history(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> VisualizationSearchHistoryResponse:
    """
    获取可视化搜索历史

    Args:
        limit: 返回数量限制
        db_manager: 数据库管理器依赖

    Returns:
        VisualizationSearchHistoryResponse: 搜索历史响应
    """
    try:
        service = VisualizationService(db_manager)
        history = service.get_search_history(limit)

        items = [
            VisualizationSearchHistoryItem(
                id=item['id'],
                stock_code=item['stock_code'],
                stock_name=item.get('stock_name'),
                searched_at=item['searched_at'],
                selected_indicators=item['selected_indicators']
            )
            for item in history
        ]

        return VisualizationSearchHistoryResponse(items=items)

    except Exception as e:
        logger.error(f"获取搜索历史失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"获取搜索历史失败: {str(e)}"
            }
        )


@router.post(
    "/history",
    response_model=VisualizationSearchHistoryItem,
    responses={
        200: {"description": "保存成功"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="保存可视化搜索历史",
    description="保存用户的可视化搜索历史记录"
)
def save_search_history(
    request: VisualizationSearchRequest,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> VisualizationSearchHistoryItem:
    """
    保存可视化搜索历史

    Args:
        request: 搜索请求
        db_manager: 数据库管理器依赖

    Returns:
        VisualizationSearchHistoryItem: 保存的记录
    """
    try:
        service = VisualizationService(db_manager)

        record_id = service.save_search_history(
            stock_code=request.stock_code,
            stock_name=request.stock_name,
            selected_indicators=request.indicator_types
        )

        # 获取刚保存的记录返回
        history = service.get_search_history(1)
        if history:
            item = history[0]
            return VisualizationSearchHistoryItem(
                id=item['id'],
                stock_code=item['stock_code'],
                stock_name=item.get('stock_name'),
                searched_at=item['searched_at'],
                selected_indicators=item['selected_indicators']
            )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "保存搜索历史后无法获取记录"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存搜索历史失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"保存搜索历史失败: {str(e)}"
            }
        )


@router.delete(
    "/history/{record_id}",
    responses={
        204: {"description": "删除成功"},
        404: {"description": "记录不存在", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    status_code=204,
    summary="删除可视化搜索历史",
    description="删除指定的可视化搜索历史记录"
)
def delete_search_history(
    record_id: int,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    删除可视化搜索历史

    Args:
        record_id: 记录ID
        db_manager: 数据库管理器依赖
    """
    try:
        success = db_manager.delete_visualization_search_history(record_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "not_found",
                    "message": f"未找到ID为 {record_id} 的记录"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除搜索历史失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"删除搜索历史失败: {str(e)}"
            }
        )


@router.get(
    "/{stock_code}",
    response_model=VisualizationResponse,
    responses={
        200: {"description": "可视化数据"},
        404: {"description": "股票不存在", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取股票可视化数据",
    description="获取指定股票的K线数据和技术指标数据，自动计算缺失的指标"
)
def get_visualization_data(
    stock_code: str,
    days: int = Query(365, ge=1, le=3650, description="获取天数"),
    indicator_types: Optional[str] = Query(None, description="指标类型列表，逗号分隔"),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> VisualizationResponse:
    """
    获取股票可视化数据

    Args:
        stock_code: 股票代码
        days: 获取天数
        indicator_types: 指标类型列表（逗号分隔）
        db_manager: 数据库管理器依赖

    Returns:
        VisualizationResponse: 可视化数据
    """
    try:
        service = VisualizationService(db_manager)

        # 解析指标类型
        indicator_list = None
        if indicator_types:
            indicator_list = [t.strip() for t in indicator_types.split(',') if t.strip()]

        result = service.get_visualization_data(
            stock_code=stock_code,
            days=days,
            indicator_types=indicator_list
        )

        if not result.get('kline_data'):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "not_found",
                    "message": f"未找到股票 {stock_code} 的数据"
                }
            )

        return VisualizationResponse(
            stock_code=result['stock_code'],
            stock_name=result.get('stock_name'),
            kline_data=result['kline_data'],
            indicators=result['indicators']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取可视化数据失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"获取可视化数据失败: {str(e)}"
            }
        )
