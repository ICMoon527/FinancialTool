# -*- coding: utf-8 -*-
"""
回测任务状态管理器

管理异步回测任务的状态和结果
"""

import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class BacktestTaskStatus(str, Enum):
    """回测任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class BacktestTask:
    """回测任务"""
    
    def __init__(
        self,
        task_id: str,
        status: BacktestTaskStatus = BacktestTaskStatus.PENDING,
        created_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        orchestrator: Optional[Any] = None
    ):
        self.task_id = task_id
        self.status = status
        self.created_at = created_at or datetime.now()
        self.started_at = started_at
        self.completed_at = completed_at
        self.result = result
        self.error = error
        self.orchestrator = orchestrator
        self._lock = threading.Lock()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 - 不包含不可序列化的对象"""
        # 确保 result 是可序列化的
        def make_serializable(value):
            import numpy as np
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: make_serializable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [make_serializable(v) for v in value]
            if isinstance(value, tuple):
                return tuple(make_serializable(v) for v in value)
            return value
        
        result = self.result
        if result is not None:
            result = make_serializable(result)
        
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": result,
            "error": self.error
        }


class BacktestTaskManager:
    """
    回测任务管理器
    
    管理异步回测任务的生命周期
    """
    
    def __init__(
        self,
        task_timeout_seconds: int = 3600,  # 1小时
        cleanup_interval_seconds: int = 600  # 10分钟
    ):
        """
        初始化任务管理器
        
        Args:
            task_timeout_seconds: 任务超时时间（秒）
            cleanup_interval_seconds: 清理间隔（秒）
        """
        self._tasks: Dict[str, BacktestTask] = {}
        self._lock = threading.Lock()
        self.task_timeout_seconds = task_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._last_cleanup_time = datetime.now()
        logger.info("回测任务管理器初始化完成")
    
    def create_task(self) -> str:
        """
        创建新任务
        
        Returns:
            task_id
        """
        task_id = str(uuid.uuid4())
        task = BacktestTask(task_id=task_id)
        
        with self._lock:
            self._tasks[task_id] = task
        
        logger.info(f"创建回测任务: {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[BacktestTask]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            BacktestTask 或 None
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: BacktestTaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        orchestrator: Optional[Any] = None
    ) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            result: 任务结果（完成时）
            error: 错误信息（失败时）
            orchestrator: 编排器引用（用于终止）
            
        Returns:
            是否成功更新
        """
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        with task._lock:
            task.status = status
            
            if status == BacktestTaskStatus.RUNNING:
                task.started_at = datetime.now()
            
            if status in [BacktestTaskStatus.COMPLETED, BacktestTaskStatus.FAILED, BacktestTaskStatus.STOPPED]:
                task.completed_at = datetime.now()
            
            if result is not None:
                task.result = result
            
            if error is not None:
                task.error = error
            
            if orchestrator is not None:
                task.orchestrator = orchestrator
        
        logger.info(f"更新任务状态: {task_id} -> {status.value}")
        return True
    
    def stop_task(self, task_id: str) -> bool:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功停止
        """
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        # 调用编排器的stop方法
        if task.orchestrator:
            try:
                task.orchestrator.stop()
                logger.info(f"已发送停止信号给任务: {task_id}")
            except Exception as e:
                logger.error(f"停止任务失败: {task_id}, 错误: {e}")
        
        # 更新任务状态
        self.update_task_status(task_id, BacktestTaskStatus.STOPPED)
        return True
    
    def cleanup_expired_tasks(self) -> int:
        """
        清理过期任务
        
        Returns:
            清理的任务数量
        """
        now = datetime.now()
        
        # 检查是否需要清理
        if (now - self._last_cleanup_time).total_seconds() < self.cleanup_interval_seconds:
            return 0
        
        with self._lock:
            expired_task_ids = []
            
            for task_id, task in self._tasks.items():
                # 检查是否超时
                if task.status in [BacktestTaskStatus.COMPLETED, BacktestTaskStatus.FAILED, BacktestTaskStatus.STOPPED]:
                    # 已完成的任务，检查是否超时
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > self.task_timeout_seconds:
                            expired_task_ids.append(task_id)
                elif task.status == BacktestTaskStatus.RUNNING:
                    # 运行中的任务，检查是否超时
                    if task.started_at:
                        age = (now - task.started_at).total_seconds()
                        if age > self.task_timeout_seconds:
                            expired_task_ids.append(task_id)
            
            # 删除过期任务
            for task_id in expired_task_ids:
                del self._tasks[task_id]
                logger.info(f"清理过期任务: {task_id}")
            
            self._last_cleanup_time = now
            return len(expired_task_ids)


# 全局任务管理器实例
_task_manager: Optional[BacktestTaskManager] = None
_task_manager_lock = threading.Lock()


def get_task_manager() -> BacktestTaskManager:
    """
    获取全局任务管理器实例
    
    Returns:
        BacktestTaskManager 实例
    """
    global _task_manager
    
    if _task_manager is None:
        with _task_manager_lock:
            if _task_manager is None:
                _task_manager = BacktestTaskManager()
    
    return _task_manager
