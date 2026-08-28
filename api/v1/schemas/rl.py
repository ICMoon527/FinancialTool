# -*- coding: utf-8 -*-
"""RL 模块 Pydantic Schema"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """训练请求"""

    algorithm: Optional[str] = Field(None, description="算法: dqn | ppo")
    episodes: Optional[int] = Field(None, description="训练轮数（总轮数，续训时需大于已完成轮数）")
    batch_size: Optional[int] = Field(None, description="批次大小")
    learning_rate: Optional[float] = Field(None, description="学习率")
    resume_from: Optional[str] = Field(None, description="断点续训来源：模型 ID、checkpoint 目录名或 latest")


class TrainResponse(BaseModel):
    """训练响应"""

    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""

    task_id: str
    status: str  # pending | running | completed | failed | stopped
    progress: int
    message: str
    created_at: str


class TrainingProgressResponse(BaseModel):
    """训练进度响应"""

    current_episode: int
    latest_reward: float
    metrics: Dict[str, List[float]]
    status: str
    progress: float = Field(0.0, description="进度百分比 0-100")
    total_episodes: int = Field(0, description="总训练轮数")
    message: str = Field("", description="当前阶段说明")
    paused: bool = Field(False, description="是否已暂停")


class ModelInfo(BaseModel):
    """模型信息"""

    model_id: str
    algorithm: str
    created_at: str
    metrics: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    """模型列表响应"""

    models: List[ModelInfo]


class EvaluateRequest(BaseModel):
    """评估请求"""

    model_id: str
    stock_codes: Optional[List[str]] = None
    max_days: Optional[int] = Field(
        100,
        description="抽样评估的最大交易日数；None 或 <=0 表示全量评估，默认 100",
    )


class EvaluateTaskResponse(BaseModel):
    """评估任务创建响应"""

    task_id: str
    status: str
    message: str


class DailySummary(BaseModel):
    """每日摘要"""

    date: str
    stock_code: str
    daily_return: float
    trade_count: int
    avg_reward: float
    trades: List[Dict[str, Any]]


class SummaryMetrics(BaseModel):
    """总体评估指标"""

    sharpe_ratio: float
    total_return: float
    win_rate: float
    max_drawdown: float
    total_trades: int


class EvaluateResultResponse(BaseModel):
    """评估结果响应"""

    cumulative_returns: List[float]
    benchmark_returns: List[float]
    daily_summaries: List[DailySummary]
    summary_metrics: SummaryMetrics


class EvaluateProgressResponse(BaseModel):
    """评估任务进度响应（供前端轮询）"""

    task_id: str
    status: str = Field(..., description="pending | running | paused | completed | failed | stopped")
    progress: float = Field(0.0, description="进度百分比 0-100")
    done: int = Field(0, description="已完成样本数")
    total: int = Field(0, description="总样本数")
    message: str = Field("", description="当前阶段说明（当前回放股票/日期、做T收益等）")
    paused: bool = Field(False, description="是否已暂停")
    result: Optional[EvaluateResultResponse] = Field(None, description="评估结果（仅终态返回）")


class DailyDecision(BaseModel):
    """单步决策"""

    step: int
    action: str
    reward: float
    position: int


class DailyReplayResponse(BaseModel):
    """单日回放响应"""

    stock_code: str
    date: str
    klines: List[Dict[str, Any]]
    decisions: List[DailyDecision]
    reward_heatmap: List[float]
    trades: List[Dict[str, Any]]