# -*- coding: utf-8 -*-
"""RL 模块 Pydantic Schema"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """训练请求"""

    algorithm: Optional[str] = Field(None, description="算法: dqn | ppo")
    episodes: Optional[int] = Field(None, description="训练轮数（总轮数，续训时需大于已完成轮数）")
    batch_size: Optional[int] = Field(None, description="批次大小")
    learning_rate: Optional[float] = Field(None, description="学习率")
    resume_from: Optional[str] = Field(None, description="断点续训来源：模型 ID、checkpoint 目录名或 latest")
    use_signal_scores: Optional[bool] = Field(
        None,
        description="是否将规则买卖点评分接入状态特征（True→state_dim=52，False/None→state_dim=50）",
    )


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


class EvaluateCompareRequest(BaseModel):
    """多模型对比评估请求：所有模型在同一批抽样数据（同基准）上评估"""

    model_ids: List[str] = Field(..., min_length=1, description="待对比的模型 ID 列表（至少 1 个）")
    max_days: Optional[int] = Field(
        100,
        description="抽样评估的最大交易日数；None 或 <=0 表示全量评估，默认 100",
    )


class CompareModelResult(BaseModel):
    """对比评估中单个模型的结果（与其余模型共用同一批样本/基准）"""

    model_id: str
    cumulative_returns: List[float]
    summary_metrics: SummaryMetrics


class EvaluateCompareResultResponse(BaseModel):
    """多模型对比评估结果"""

    samples: List[Dict[str, str]] = Field(..., description="实际评估的样本列表（[{stock_code, date}]）")
    benchmark_returns: List[float] = Field(..., description="同一批样本的买入持有基准（所有模型共用）")
    models: List[CompareModelResult]


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
    current_model_idx: Optional[int] = Field(
        None, description="对比评估中当前正在评估的模型序号（0-based）；单模型评估为 None"
    )
    model_done: Optional[int] = Field(None, description="当前模型已完成的交易日数")
    model_total: Optional[int] = Field(None, description="每个模型的交易日总数（= 抽样天数）")
    result: Optional[Union[EvaluateResultResponse, EvaluateCompareResultResponse]] = Field(
        None, description="评估结果（仅终态返回）；单模型为 EvaluateResultResponse，对比评估为 EvaluateCompareResultResponse"
    )


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


# 由于文件顶部启用了 from __future__ import annotations，
# 字段注解在运行时为字符串（前向引用），需显式重建以解析 Union 成员类型，
# 否则报错：`EvaluateProgressResponse` is not fully defined
EvaluateProgressResponse.model_rebuild()