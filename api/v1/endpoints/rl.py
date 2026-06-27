# -*- coding: utf-8 -*-
"""RL 训练 API 端点"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_database_manager
from api.v1.schemas.rl import (
    DailyReplayResponse,
    EvaluateRequest,
    EvaluateResultResponse,
    ModelInfo,
    ModelListResponse,
    TaskStatusResponse,
    TrainRequest,
    TrainResponse,
    TrainingProgressResponse,
)
from src.services.rl_service import RLService
from src.storage import DatabaseManager

router = APIRouter()

# 全局 RLService 实例（懒加载）
_rl_service: Optional[RLService] = None


def get_rl_service() -> RLService:
    """获取 RLService 单例"""
    global _rl_service
    if _rl_service is None:
        from rl.config import RLConfig
        config = RLConfig.from_env()
        db = DatabaseManager.get_instance()
        _rl_service = RLService(config, db)
    return _rl_service


@router.get("/models", response_model=ModelListResponse)
async def list_models(service: RLService = Depends(get_rl_service)):
    """获取已训练的模型列表"""
    models = service.get_models()
    return ModelListResponse(
        models=[
            ModelInfo(
                model_id=m["model_id"],
                algorithm=m["algorithm"],
                created_at=m["created_at"],
                metrics=m.get("metrics"),
            )
            for m in models
        ]
    )


@router.post("/train", response_model=TrainResponse)
async def start_training(
    request: TrainRequest, service: RLService = Depends(get_rl_service)
):
    """启动异步训练任务"""
    params = {}
    if request.algorithm is not None:
        params["default_algorithm"] = request.algorithm
    if request.episodes is not None:
        params["training_episodes"] = request.episodes
    if request.batch_size is not None:
        params["batch_size"] = request.batch_size
    if request.learning_rate is not None:
        params["learning_rate"] = request.learning_rate

    task_id = service.start_training(params)
    return TrainResponse(task_id=task_id, status="pending", message="训练任务已创建")


@router.get("/train/{task_id}/status", response_model=TaskStatusResponse)
async def get_training_status(
    task_id: str, service: RLService = Depends(get_rl_service)
):
    """查询训练任务状态"""
    status = service.get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**status)


@router.post("/train/{task_id}/stop")
async def stop_training(
    task_id: str, service: RLService = Depends(get_rl_service)
):
    """停止训练任务"""
    success = service.stop_training(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "训练已停止"}


@router.get("/train/{task_id}/progress", response_model=TrainingProgressResponse)
async def get_training_progress(
    task_id: str, service: RLService = Depends(get_rl_service)
):
    """获取训练进度（供前端轮询）"""
    progress = service.get_training_progress(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TrainingProgressResponse(**progress)


@router.post("/evaluate", response_model=EvaluateResultResponse)
async def evaluate_model(
    request: EvaluateRequest, service: RLService = Depends(get_rl_service)
):
    """评估模型"""
    result = service.evaluate(request.model_id, request.stock_codes)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.get("/evaluate/{model_id}/daily/{stock_code}/{date}", response_model=DailyReplayResponse)
async def get_daily_replay(
    model_id: str,
    stock_code: str,
    date: str,
    service: RLService = Depends(get_rl_service),
):
    """获取单日回放数据"""
    result = service.evaluate_daily(model_id, stock_code, date)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return DailyReplayResponse(**result)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str, service: RLService = Depends(get_rl_service)
):
    """删除模型"""
    success = service.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": "模型已删除"}