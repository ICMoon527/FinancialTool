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
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends

from api.deps import get_database_manager
from api.v1.schemas.visualization import (
    VisualizationResponse,
    VisualizationSearchHistoryResponse,
    VisualizationSearchHistoryItem,
    VisualizationSearchRequest,
    ChipDistributionResponse
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
    limit: int = Query(20, ge=1, le=1000, description="返回数量限制"),
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


@router.patch(
    "/history/{record_id}/timestamp",
    response_model=VisualizationSearchHistoryItem,
    responses={
        200: {"description": "更新成功"},
        404: {"description": "记录不存在", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="更新搜索历史记录时间戳",
    description="将指定搜索历史记录的时间戳更新为当前时间，使其置顶显示"
)
def update_search_history_timestamp(
    record_id: int,
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> VisualizationSearchHistoryItem:
    """
    更新搜索历史记录时间戳为当前时间

    Args:
        record_id: 记录ID
        db_manager: 数据库管理器依赖

    Returns:
        VisualizationSearchHistoryItem: 更新后的记录
    """
    try:
        service = VisualizationService(db_manager)
        updated_record = service.update_search_history_timestamp(record_id)
        
        if not updated_record:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "not_found",
                    "message": f"未找到ID为 {record_id} 的记录"
                }
            )
        
        return VisualizationSearchHistoryItem(
            id=updated_record['id'],
            stock_code=updated_record['stock_code'],
            stock_name=updated_record.get('stock_name'),
            searched_at=updated_record['searched_at'],
            selected_indicators=updated_record['selected_indicators']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新搜索历史时间戳失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"更新搜索历史时间戳失败: {str(e)}"
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
    days: int = Query(3650, ge=1, le=3650, description="获取天数（如果提供了 start_date，则此参数被忽略）"),
    indicator_types: Optional[str] = Query(None, description="指标类型列表，逗号分隔"),
    start_date: Optional[str] = Query(None, description="起始日期，格式为 YYYY-MM-DD"),
    force_update: bool = Query(False, description="是否强制更新，跳过交易时间和已有数据的检查"),
    update_circulating_shares: bool = Query(True, description="是否更新流通股本和换手率"),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> VisualizationResponse:
    """
    获取股票可视化数据

    Args:
        stock_code: 股票代码
        days: 获取天数（如果提供了 start_date，则此参数被忽略）
        indicator_types: 指标类型列表（逗号分隔）
        start_date: 起始日期，格式为 YYYY-MM-DD
        force_update: 是否强制更新，跳过交易时间和已有数据的检查
        update_circulating_shares: 是否更新流通股本和换手率
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

        # 解析起始日期
        start_date_obj = None
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"无效的起始日期格式: {start_date}，使用默认天数")

        result = service.get_visualization_data(
            stock_code=stock_code,
            days=days,
            indicator_types=None,
            start_date=start_date_obj,
            force_update=force_update,
            update_circulating_shares=update_circulating_shares
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
            indicators=result['indicators'],
            chip_distribution=result.get('chip_distribution')
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


@router.get(
    "/chip-distribution/{stock_code}",
    response_model=ChipDistributionResponse,
    responses={
        200: {"description": "筹码分布数据"},
        404: {"description": "股票不存在"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取筹码分布数据",
    description="获取指定股票的筹码分布数据，包括获利盘、套牢盘、平均成本等"
)
def get_chip_distribution(
    stock_code: str,
    days: int = Query(365, ge=1, le=3650, description="获取天数"),
    start_date: Optional[str] = Query(None, description="起始日期，格式为 YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="截止日期，格式为 YYYY-MM-DD"),
    end_date_idx: Optional[int] = Query(None, description="截止日期索引"),
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> ChipDistributionResponse:
    """
    获取筹码分布数据

    Args:
        stock_code: 股票代码
        days: 获取天数
        start_date: 起始日期
        end_date: 截止日期
        end_date_idx: 截止日期索引
        db_manager: 数据库管理器依赖

    Returns:
        筹码分布响应
    """
    try:
        service = VisualizationService(db_manager)
        
        start_date_obj = None
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"无效的起始日期格式: {start_date}，使用默认天数")
        
        end_date_obj = None
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"无效的截止日期格式: {end_date}")
        
        result = service.get_chip_distribution(
            stock_code=stock_code,
            days=days,
            start_date=start_date_obj,
            end_date_idx=end_date_idx,
            end_date=end_date_obj
        )
        
        return ChipDistributionResponse(
            stock_code=result['stock_code'],
            price_bins=result['price_bins'],
            chip_volumes=result['chip_volumes'],
            profit_volumes=result['profit_volumes'],
            loss_volumes=result['loss_volumes'],
            avg_cost=result['avg_cost'],
            max_chip_price=result['max_chip_price'],
            current_price=result['current_price']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取筹码分布数据失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"获取筹码分布数据失败: {str(e)}"
            }
        )
