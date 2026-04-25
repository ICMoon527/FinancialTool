# -*- coding: utf-8 -*-
"""Backtest endpoints."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.deps import get_database_manager
from api.v1.schemas.backtest import (
    BacktestRunRequest,
    BacktestRunResponse,
    BacktestResultItem,
    BacktestResultsResponse,
    PerformanceMetrics,
)
from api.v1.schemas.strategy_backtest import (
    StrategyInfo,
    StrategyListResponse,
    StrategyBacktestRunRequest,
    StrategyBacktestRunResponse,
    StrategyBacktestRunAsyncRequest,
    StrategyBacktestRunAsyncResponse,
    StrategyBacktestTaskStatusResponse,
    StrategyBacktestStopByTaskIdRequest,
)
from api.v1.schemas.common import ErrorResponse
from src.services.backtest_service import BacktestService
from src.services.strategy_backtest_service import StrategyBacktestService
from src.core.strategy_backtest.log_capture import get_log_capturer
from src.storage import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()

# 策略回测服务单例
_strategy_backtest_service: Optional[StrategyBacktestService] = None
# 回测运行状态
_is_backtest_running = False


def get_strategy_backtest_service() -> StrategyBacktestService:
    """获取策略回测服务单例。"""
    global _strategy_backtest_service
    if _strategy_backtest_service is None:
        _strategy_backtest_service = StrategyBacktestService()
    return _strategy_backtest_service


def log_generator():
    """SSE日志生成器"""
    capturer = get_log_capturer()
    while True:
        log = capturer.get_logs(timeout=0.5)
        if log:
            yield f"data: {log}\n\n"
        # 检查回测是否还在运行
        global _is_backtest_running
        if not _is_backtest_running:
            # 再等一会儿，确保所有日志都被发送
            time.sleep(0.5)
            # 发送剩余的日志
            while True:
                log = capturer.get_logs(timeout=0.1)
                if log:
                    yield f"data: {log}\n\n"
                else:
                    break
            yield "data: [DONE]\n\n"
            break


# ============ 策略回测端点 ============

@router.get(
    "/strategies",
    response_model=StrategyListResponse,
    responses={
        200: {"description": "策略列表"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取策略列表",
    description="获取所有可用的策略列表",
)
def get_strategies() -> StrategyListResponse:
    try:
        service = get_strategy_backtest_service()
        strategies = service.get_available_strategies()
        return StrategyListResponse(
            strategies=[StrategyInfo(**s) for s in strategies]
        )
    except Exception as exc:
        logger.error(f"获取策略列表失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"获取策略列表失败: {str(exc)}"},
        )


@router.get(
    "/strategy/logs",
    summary="获取回测实时日志 (SSE)",
    description="使用Server-Sent Events获取回测的实时日志",
)
async def get_backtest_logs():
    """SSE端点，用于实时推送回测日志"""
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/strategy/stop",
    summary="停止策略回测",
    description="停止当前正在运行的策略回测",
)
def stop_strategy_backtest():
    """停止当前运行的策略回测"""
    try:
        service = get_strategy_backtest_service()
        service.stop_backtest()
        return {"success": True, "message": "已发送停止回测信号"}
    except Exception as exc:
        logger.error(f"停止回测失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"停止回测失败: {str(exc)}"},
        )


@router.post(
    "/strategy/run-async",
    response_model=StrategyBacktestRunAsyncResponse,
    responses={
        200: {"description": "异步回测任务已提交"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="异步运行策略回测",
    description="异步运行指定策略的回测，立即返回task_id",
)
def run_strategy_backtest_async(
    request: StrategyBacktestRunAsyncRequest,
) -> StrategyBacktestRunAsyncResponse:
    """异步运行策略回测"""
    global _is_backtest_running
    capturer = get_log_capturer()
    
    try:
        # 调试：打印接收到的请求
        logger.info(f"收到异步回测请求: {request}")
        logger.info(f"  strategy_id: {request.strategy_id}")
        logger.info(f"  start_date: {request.start_date}")
        logger.info(f"  end_date: {request.end_date}")
        logger.info(f"  max_positions: {request.max_positions}")
        
        # 清空之前的日志
        capturer.clear()
        # 开始捕获日志
        capturer.start_capture([
            "src.core.strategy_backtest",
            "src.services.strategy_backtest_service",
        ])
        
        _is_backtest_running = True
        
        service = get_strategy_backtest_service()
        task_id = service.run_backtest_async(
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            max_positions=request.max_positions,
            on_complete=lambda: _update_running_state(False),
            on_error=lambda: _update_running_state(False),
        )
        
        return StrategyBacktestRunAsyncResponse(
            success=True,
            message="异步回测任务已提交",
            task_id=task_id,
        )
    except Exception as exc:
        logger.error(f"异步回测任务提交失败: {exc}", exc_info=True)
        _is_backtest_running = False
        capturer.stop_capture()
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"异步回测任务提交失败: {str(exc)}"},
        )


def _update_running_state(is_running: bool):
    """更新全局运行状态"""
    global _is_backtest_running
    _is_backtest_running = is_running
    if not is_running:
        capturer = get_log_capturer()
        capturer.stop_capture()


@router.get(
    "/strategy/task/{task_id}",
    response_model=StrategyBacktestTaskStatusResponse,
    responses={
        200: {"description": "获取任务状态成功"},
        404: {"description": "任务不存在"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取回测任务状态",
    description="获取指定task_id的回测任务状态",
)
def get_strategy_backtest_task_status(
    task_id: str,
) -> StrategyBacktestTaskStatusResponse:
    """获取回测任务状态"""
    try:
        # 防护：检查 task_id 是否有效
        if not task_id or task_id == 'undefined' or task_id == 'null':
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_task_id", "message": "无效的任务ID"},
            )
        
        service = get_strategy_backtest_service()
        task = service.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=404,
                detail={"error": "task_not_found", "message": "任务不存在"},
            )
        
        return StrategyBacktestTaskStatusResponse(
            success=True,
            message="获取任务状态成功",
            task=task,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"获取任务状态失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"获取任务状态失败: {str(exc)}"},
        )


@router.post(
    "/strategy/stop-by-task-id",
    summary="通过任务ID停止策略回测",
    description="通过task_id停止指定的回测任务",
)
def stop_strategy_backtest_by_task_id(
    request: StrategyBacktestStopByTaskIdRequest,
):
    """通过任务ID停止策略回测"""
    try:
        service = get_strategy_backtest_service()
        success = service.stop_backtest_by_task_id(request.task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "task_not_found", "message": "任务不存在或无法停止"},
            )
        
        return {"success": True, "message": "已发送停止回测信号"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"停止回测失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"停止回测失败: {str(exc)}"},
        )


@router.post(
    "/strategy/run",
    response_model=StrategyBacktestRunResponse,
    responses={
        200: {"description": "策略回测执行完成"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="运行策略回测",
    description="运行指定策略的回测",
)
def run_strategy_backtest(
    request: StrategyBacktestRunRequest,
) -> StrategyBacktestRunResponse:
    global _is_backtest_running
    capturer = get_log_capturer()
    
    try:
        # 清空之前的日志
        capturer.clear()
        # 开始捕获日志
        capturer.start_capture([
            "src.core.strategy_backtest",
            "src.services.strategy_backtest_service",
        ])
        _is_backtest_running = True
        
        service = get_strategy_backtest_service()
        result = service.run_backtest(
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            stock_pool=request.stock_pool,
            max_positions=request.max_positions,
        )
        return StrategyBacktestRunResponse(
            success=True,
            message="策略回测执行完成",
            results=result.get("results"),
            metrics=result.get("metrics"),
            reports=result.get("reports"),
        )
    except Exception as exc:
        logger.error(f"策略回测执行失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"策略回测执行失败: {str(exc)}"},
        )
    finally:
        _is_backtest_running = False
        # 停止捕获日志
        capturer.stop_capture()


# ============ 历史AI分析回测端点 (已废弃) ============

@router.post(
    "/run",
    response_model=BacktestRunResponse,
    responses={
        200: {"description": "回测执行完成"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="触发回测 (已废弃)",
    description="对历史分析记录进行回测评估，并写入 backtest_results/backtest_summaries",
    deprecated=True,
)
def run_backtest(
    request: BacktestRunRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BacktestRunResponse:
    try:
        service = BacktestService(db_manager)
        stats = service.run_backtest(
            code=request.code,
            force=request.force,
            eval_window_days=request.eval_window_days,
            min_age_days=request.min_age_days,
            limit=request.limit,
        )
        return BacktestRunResponse(**stats)
    except Exception as exc:
        logger.error(f"回测执行失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"回测执行失败: {str(exc)}"},
        )


@router.get(
    "/results",
    response_model=BacktestResultsResponse,
    responses={
        200: {"description": "回测结果列表"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取回测结果 (已废弃)",
    description="分页获取回测结果，支持按股票代码过滤",
    deprecated=True,
)
def get_backtest_results(
    code: Optional[str] = Query(None, description="股票代码筛选"),
    eval_window_days: Optional[int] = Query(None, ge=1, le=120, description="评估窗口过滤"),
    page: int = Query(1, ge=1, description="页码"),
    limit: int = Query(20, ge=1, le=200, description="每页数量"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BacktestResultsResponse:
    try:
        service = BacktestService(db_manager)
        data = service.get_recent_evaluations(code=code, eval_window_days=eval_window_days, limit=limit, page=page)
        items = [BacktestResultItem(**item) for item in data.get("items", [])]
        return BacktestResultsResponse(
            total=int(data.get("total", 0)),
            page=page,
            limit=limit,
            items=items,
        )
    except Exception as exc:
        logger.error(f"查询回测结果失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"查询回测结果失败: {str(exc)}"},
        )


@router.get(
    "/performance",
    response_model=PerformanceMetrics,
    responses={
        200: {"description": "整体回测表现"},
        404: {"description": "无回测汇总", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取整体回测表现 (已废弃)",
    deprecated=True,
)
def get_overall_performance(
    eval_window_days: Optional[int] = Query(None, ge=1, le=120, description="评估窗口过滤"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> PerformanceMetrics:
    try:
        service = BacktestService(db_manager)
        summary = service.get_summary(scope="overall", code=None, eval_window_days=eval_window_days)
        if summary is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": "未找到整体回测汇总"},
            )
        return PerformanceMetrics(**summary)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询整体表现失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"查询整体表现失败: {str(exc)}"},
        )


@router.get(
    "/performance/{code}",
    response_model=PerformanceMetrics,
    responses={
        200: {"description": "单股回测表现"},
        404: {"description": "无回测汇总", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取单股回测表现 (已废弃)",
    deprecated=True,
)
def get_stock_performance(
    code: str,
    eval_window_days: Optional[int] = Query(None, ge=1, le=120, description="评估窗口过滤"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> PerformanceMetrics:
    try:
        service = BacktestService(db_manager)
        summary = service.get_summary(scope="stock", code=code, eval_window_days=eval_window_days)
        if summary is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"未找到 {code} 的回测汇总"},
            )
        return PerformanceMetrics(**summary)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询单股表现失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"查询单股表现失败: {str(exc)}"},
        )

